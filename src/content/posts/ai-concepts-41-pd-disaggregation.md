---
title: "41 | P/D 分离：把 prefill 从解码机器上搬走，ITL 从 430 掉到 21——账单换成了两个数字"
description: "本机没有 GPU，这篇用离散事件调度模拟（step_time = O + tokens×T）量 P/D 分离：8 副本、32 条解码流在跑、4 条 4096-token 长 prompt 同时到达。同机混跑时最大 ITL 是 430（chunk=4096）/ 225（chunk=2048）/ 71（chunk=512）；拆成 2 个 P + 6 个 D 之后是 21——去掉的是尖峰，不是平均。代价写在两个地方：中位 ITL 从 20 涨到 21（少了 2 个副本解码），以及每请求一次的 KV 传输税——4096 token 的 KV 是 1.250 GiB，单口 100GbE 上 107.4 ms（相当于 prefill 时间的 25%），400GbE 上 26.8 ms，NVLink 上 1.5 ms。交叉点很干脆：不分块时尖峰 410 ms，只要落点副本上有 1 条在跑的解码流，107 ms 的传输税就比它替下来的尖峰便宜。"
pubDate: 2026-10-04
tags: ["AI核心概念", "P/D分离", "Prefill-Decode Disaggregation", "KV传输", "推理优化", "DistServe"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 41
slug: ai-concepts-41-pd-disaggregation
---

# AI核心概念(41)：P/D 分离——把 prefill 搬走之后，你买到的是形状，不是速度

> 「P/D 分离」在文档里看起来像一个部署选项，但它其实是一次**账单转移**：你把 prefill 从解码机器上搬走，换掉的是 ITL 曲线的**形状**，付出去的是两笔新开销——每请求一次的 KV 传输税，和「少了几台机器做解码」的常数税。
>
> 这篇还是在同一台没有 GPU 的机器上跑离散事件调度模拟（成本模型和盲区都写在正文里）。一句话结果：同机混跑时最大 ITL 是 **430**（chunk=4096）/ **225**（chunk=2048）/ **71**（chunk=512）；拆成 2 个 P + 6 个 D 之后是 **21**（20.5×）。而代价：中位 ITL 从 **20 涨到 21**，每请求多一笔 KV 传输——4096 token 的 KV 是 **1.250 GiB**，单口 100GbE 上 **107.4 ms**（prefill 的 25.0%），400GbE 上 26.8 ms，NVLink 上 1.5 ms。

## 🎯 为什么要把这两个阶段拆开

先把「为什么」对齐到三篇原文（标题和 arXiv 号我逐条核过 arxiv.org 页面）：

- **Splitwise**（arXiv 2311.18677，2023-11-30，标题 *Efficient generative LLM inference using phase splitting*）的摘要把动机写得很清楚：一次请求里有**计算密集的 prompt 计算**和**访存密集的 token 生成**两个阶段，两者「延迟、吞吐、内存、功耗特性各不相同」；而且尽管有最先进的 batching 和调度，**token 生成阶段是在浪费算力**——它不需要最新 GPU 的算力，因此**可以用更便宜、更省电的硬件跑**。
- **DistServe**（arXiv 2401.09670，2024-01-18，*Disaggregating Prefill and Decoding for Goodput-optimized LLM Serving*）补的是另一半：把两者放在一起（colocate）会带来「**强烈的 prefill-decoding 干扰**」，而且会把两个阶段的**资源分配和并行计划绑在一起**。它指出应用关心的是两个独立指标——prefill 的 TTFT、decode 的每 token 时间（TPOT）——而在严格延迟要求下，「现有系统只能牺牲其中一个，或者多买算力同时满足两个」。
- **Mooncake**（arXiv 2407.00079，2024-06-24）走的是同一个方向但把重心放在 KV 上：标题就是 *A KVCache-centric Disaggregated Architecture for LLM Serving*——分离之后，KV 本身成了被调度的一等资源。

落地的旋钮也不神秘：vLLM 有一节 **disaggregated prefill** 的文档（上一期 40 就是在那节的末尾提到的它），SGLang 也把 PD-disaggregation 作为部署模式写进文档。真正的问题是**怎么切、切完值不值**。

## 🔬 一台没有 GPU 的机器怎么量「切法」

边界先说清楚：这台机器没有 GPU，也没装 vLLM，下面是**离散事件调度模拟，不是 GPU 实测**。成本模型还是那一行：

```
一步的耗时 = O + (这一步处理的 token 数) × T
```

上一期 40 用 `O=200、T=1` 是为了让 `最大 ITL = O + chunk` 这个恒等式看得干净；这一期要跟 KV 传输的**毫秒数**对话，所以把锚点换成 `O = 20 ms`（每步固定开销）、`T = 0.1 ms/token`（即单副本 prefill 吞吐 10k tokens/s 量级）。**这两个值是本文的建模选择，不是硬件测量值**——换掉它们，下面所有绝对值都会变。

工作负载：**8 个副本**，t=0 时 **32 条解码流**各还需 128 个 token，同时 **4 条 4096-token 的长 prompt** 到达。

**这个模型的已知盲区，要写在表前面：**

1. 它假定时间与 token 数**严格成正比**，所以能量「调度造成的排队和尖峰」，但**量不到** Splitwise 说的那部分收益——「decode 不需要最新 GPU 的算力，可以跑在更便宜的卡上」。那是硬件层面的账。
2. decode 步耗时对批量大小也是线性的。真实系统里 decode 是访存瓶颈，批量的边际成本更小，所以下面 P/D 那笔「常数税」在真机上**大概率更小**。
3. 它没有到达率模型，所以只能回答「同一次冲击怎么分摊」，**回答不了「P 副本该配几台」**。

## 💻 可运行代码

```python
O, T, REMAIN, PROMPT = 20.0, 0.1, 128, 4096
N_REP, N_STREAM, N_PROMPT = 8, 32, 4
KV_PER_TOKEN = 2 * 80 * 8 * 128 * 2      # Llama-3-70B GQA：K+V × 80 层 × 8 KV 头 × 128 维 × fp16
BW = [("100GbE 单口", 12.5e9), ("400GbE 单口", 50e9), ("TP8 折合 800GbE", 100e9),
      ("NVLink4 单口", 900e9), ("理想零税", float("inf"))]


def sim_mixed_replica(n0, prompt_len, chunk, t0=0.0):
    """同机混跑的一个副本：解码优先，剩余预算喂 prefill（装不下就切开）。"""
    streams = [REMAIN] * n0
    pending, t, itls, ttft = prompt_len, t0, [], None
    while any(s > 0 for s in streams) or pending:
        active = sum(1 for s in streams if s > 0)
        took = min(pending, max(chunk - active, 0)) if pending else 0
        pending -= took
        dt = O + (active + took) * T
        for i, s in enumerate(streams):
            if s > 0:
                streams[i] -= 1
                itls.append(dt)
        t += dt
        if took and pending == 0 and ttft is None:
            ttft = t
            streams.append(REMAIN)
    return itls, ttft, t


def sim_decode_replica(n0, arrivals, t0=0.0):
    """纯解码副本：只有解码流，没有 prefill，arrivals = 新流到达时刻。"""
    streams, t, itls, pend = [REMAIN] * n0, t0, [], sorted(arrivals)
    while any(s > 0 for s in streams) or pend:
        while pend and pend[0] <= t:
            streams.append(REMAIN); pend.pop(0)
        if not any(s > 0 for s in streams) and pend:
            t = pend.pop(0); streams.append(REMAIN); continue
        dt = O + sum(1 for s in streams if s > 0) * T
        for i, s in enumerate(streams):
            if s > 0:
                streams[i] -= 1
                itls.append(dt)
        t += dt
    return itls, t


# A 同机混跑：4 条 prompt 落在前 4 个副本
for chunk in (512, 2048, 4096):
    reps = [sim_mixed_replica(4, PROMPT if i < N_PROMPT else 0, chunk) for i in range(N_REP)]
    ...
# B P/D 分离：n_p 个 P + (8-n_p) 个 D；P 副本一整个 prompt 一步装完，TTFT += KV 传输
for n_p in (2, 4):
    n_d, pf = N_REP - n_p, O + PROMPT * T
    base, extra = divmod(N_STREAM, n_d)
    for label, bw in BW:
        tax = 0.0 if bw == float("inf") else PROMPT * KV_PER_TOKEN / bw * 1000
        reps = [sim_decode_replica(base + (1 if i < extra else 0),
                                   [pf + tax] if i < N_PROMPT else []) for i in range(n_d)]
```

（完整脚本 `41-pd-disaggregation.py` 连同真实输出存在 `~/scripts/blog_daily/`。）

## 🔬 结果：尖峰没了，平均数变贵了一点点

| 调度 | 最大 ITL | 中位 ITL | 均值 ITL | 最大 TTFT | 最晚完成 |
|---|---|---|---|---|---|
| 同机 chunk=512 | **71** | 20 | 22 | 593 | 3214 |
| 同机 chunk=2048 | **225** | 20 | 22 | 471 | 3094 |
| 同机 chunk=4096 | **430** | 20 | 22 | 450 | 3074 |
| P/D 2P6D（KV 税 107 ms） | **21** | 21 | 21 | 537 | 3190 |
| P/D 2P6D（KV 税 13 ms） | **21** | 21 | 21 | 443 | 3090 |
| P/D 2P6D（KV 税 0） | **21** | 21 | 21 | 430 | 3070 |
| P/D 4P4D（KV 税 107 ms） | **21** | 21 | 21 | 537 | 3195 |

本机真实输出（节选，与上表逐格一致）：

```
colocated chunk=512        最大ITL    71  中位   20  均值     22  最大TTFT   593  最晚完成  3214
colocated chunk=2048       最大ITL   225  中位   20  均值     22  最大TTFT   471  最晚完成  3094
colocated chunk=4096       最大ITL   430  中位   20  均值     22  最大TTFT   450  最晚完成  3074
P/D 2P6D  100GbE 单口        最大ITL    21  中位   21  均值     21  最大TTFT   537  最晚完成  3190  (prefill 430 + KV税 107)
P/D 2P6D  NVLink4 单口       最大ITL    21  中位   21  均值     21  最大TTFT   431  最晚完成  3077  (prefill 430 + KV税 1)
P/D 2P6D  理想零税             最大ITL    21  中位   21  均值     21  最大TTFT   430  最晚完成  3070  (prefill 430 + KV税 0)
```

## 🔬 从表里读出来的三件事

**1. P/D 清掉的是尖峰，ITL 从 430 变 21 是 20.5 倍。** 而且 21 这个数已经不是「无干扰」了——它是 `O + 5.3 个 token`（32 条流摊到 6 个 D 副本上），说明**干扰只是换了个名字：从 prefill 尖峰换成了批量常数**。这一期最该记住的对比其实不是 430→21，而是**中位 ITL 20→21**：你为了拿到一条平直的线，先在每个 token 上多付了约 5%（8 副本里抽出 2 台只做 prefill）。真实系统里这笔税多半更小（盲区 2），但它是存在的，且方向和量级都该报出来。

**2. 最好的一格是「P/D + 快链路」：ITL 21 对 430，TTFT 443 对 450。** 注意同机不分块的 TTFT 是 450 而不是 430——因为那一步里 4 个预算被解码流吃掉了，4096 个 token 要分两步装完（这一步 `O` 得付两次）；而 P 副本上没有解码流，一个 prompt 就是干净的一步 430。所以当 KV 税 ≤ 13 ms 时，**P/D 在两个轴上都赢**，这也解释了 400GbE/NVLink 场景下大家为什么毫不犹豫地拆。

**3. 单口 100GbE 下，P/D 的 TTFT 从 450 涨到 537（+19.3%），「最晚完成」也从 3074 涨到 3190（+3.8%）。** 后者值得单独说：新流晚 87 ms 进场，但它进场后要在 D 副本上占 128 步的位置，于是这个延迟一路传到了收尾。**P/D 是延迟整形，不是免费的吞吐**——在这套线性成本模型里，总 token 数没变，总时间也就不该指望它变好（顺带提醒：这一列本来就不能当吞吐结论读，见盲区 1）。

## 🔬 KV 传输税：1.250 GiB 一次，走哪条线值多少

KV 尺寸是**算术，不是测量**：Llama-3-70B（GQA，80 层、8 个 KV 头、head_dim 128、fp16）每 token 的 KV = `2 × 80 × 8 × 128 × 2 B = 327,680 B = 320 KiB`，一条 4096-token 的 prompt 就是 **1.250 GiB**。

| 链路（整份 KV 走一条） | 传输耗时 | 相当于 prefill（430 ms）的 |
|---|---|---|
| 100GbE 单口（12.5 GB/s） | **107.4 ms** | 25.0% |
| 400GbE 单口（50 GB/s） | 26.8 ms | 6.2% |
| TP8 分片折合（100 GB/s） | 13.4 ms | 3.1% |
| NVLink4 单口（900 GB/s） | 1.5 ms | 0.3% |

实务上的数字落在哪一行，取决于你的并行配置和链路——TP 分片会把有效带宽乘以卡数（副本内 8 卡各走一路），这也是为什么「KV 传输税」在不同团队嘴里能差两个数量级。**报 P/D 收益而不报链路配置，等于没报。**

交叉点可以直接算出来——同机混跑下，每来一条长 prompt，**落点副本上正在跑的解码流就多挨一次尖峰**：

```
chunk=512   尖峰   51 ms  对 100GbE KV税  107.4 ms  → 落点副本上解码流数 >  2.11 时 P/D 的账更划算
chunk=512   尖峰   51 ms  对 400GbE KV税   26.8 ms  → 落点副本上解码流数 >  0.53 时 P/D 的账更划算
chunk=4096  尖峰  410 ms  对 100GbE KV税  107.4 ms  → 落点副本上解码流数 >  0.26 时 P/D 的账更划算
```

读法：**不分块时尖峰是 410 ms，只要落点副本上有 1 条在跑的解码流，107 ms 的传输税就比它替下来的那一刀便宜**。而如果你已经把 chunk 调到 512，尖峰只剩 51 ms，那要有 3 条以上的解码流同时在线，P/D 才在「单次冲击」这笔账上划算——**分块预填充和 P/D 分离是同一笔钱的两个花法**，先调 chunk 再考虑拆机器，顺序反了会白付传输税。

## 💡 落地清单

- P/D 分离买的是一条**平的 ITL**（430→21），付的是**每请求一次的 KV 税**和**每 token 一次的副本税**（中位 20→21）。三个数字都要报，只报第一个就是营销。
- 先量 KV 传输的真实带宽（含 TP 分片、含跨机架那段），再决定要不要拆：100GbE 单口那一行会把 TTFT 抬 19.3%，NVLink 那一行只抬 0.3%。
- 「chunk 调小」和「拆 P/D」解决的是同一个尖峰：不分块场景 P/D 稳赢（1 条解码流就够），chunk=512 场景要 3 条以上才划算。
- 别忘了 Splitwise 那条我这份模型**量不到**的收益：decode 不挑卡。如果你的 P 池用满血卡、D 池用便宜卡，省下来的钱是这张表看不见的那一栏。
- 最后是模型自己承认的空白：它算不出 **P 副本该配几台**——那要到达率分布，要排队论，要真机。

---

*本文的实验脚本 `41-pd-disaggregation.py` 与本机真实输出保存在 `~/scripts/blog_daily/`。`O=20 ms、T=0.1 ms/token` 是本文的建模选择，不是硬件测量值；KV 尺寸与传输耗时是从公开配置和链路规格算出来的算术结果，不是实测。*
