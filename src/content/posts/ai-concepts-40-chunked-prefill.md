---
title: "40 | 分块预填充：最大 ITL 恰好等于「固定开销 + chunk」——尾延迟是一张可以按 chunk 定价的价目表"
description: "本机没有 GPU，所以这篇用离散事件调度模拟（step_time = O + tokens·T）量分块预填充：一条 4096 token 的长 prompt 撞进正在解码的批次，不分块时最大 ITL 是 4499，chunk=512 时是 712——而表里每一行的最大 ITL 都精确等于 O + chunk。代价写在另外两列：TTFT 从 4296 涨到 5923（+38%），chunk=128 时 TTFT 直接 10792（+151%）。最反直觉的是中位 ITL：chunk=512 时只有 203（4096 个 prefill token 只占 9 步，剩下 23 步是纯解码的便宜步），而 chunk=256/128 时中位 ITL 涨到 456/328——分块太小，你买到的是「没有尖峰」，付的是「全程都在浪」。另附一处事实核对：常被引作 Sarathi-Serve 的 arXiv 2401.08671 其实是 DeepSpeed-FastGen。"
pubDate: 2026-10-03
tags: ["AI核心概念", "分块预填充", "Chunked Prefill", "vLLM", "推理优化", "尾延迟"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 40
slug: ai-concepts-40-chunked-prefill
---

# AI核心概念(40)：分块预填充——最大 ITL 恰好等于「固定开销 + chunk」

> 「把长 prompt 切开」听起来像个吞吐量技巧，但它真正卖给你的是延迟曲线的一个形状：**最大 ITL ≈ 每步固定开销 + chunk 大小**。这个式子意味着尾延迟不是被「优化」掉的，而是被你按 chunk 大小**定价**的——你写多少，就买多少。
>
> 这篇的实验是在这台没有 GPU 的机器上跑一个离散事件调度模拟（成本模型写在正文里，结论的适用边界也写在正文里）。它量出的第一件事不太像结论、更像一张价目表：一条 4096 token 的长 prompt 撞进正在解码的批次时，不分块的最大 ITL 是 **4499**，`chunk=512` 时是 **712**（6.32×），`chunk=128` 时是 **328**——而代价写在另外两列：TTFT 从 4296 一路涨到 10792。

## 🎯 旋钮在哪：vLLM 的两种调度策略

先把「分块预填充（chunked prefill）」这个词对齐到具体行为，以下读自 vLLM 官方文档（Performance and Tuning / Optimization 两版页面）：

- **默认调度**：优先 prefill，且**不把 prefill 和 decode 放进同一批**。官方原话是这套策略「optimizes the TTFT, but incurs slower ITL and inefficient GPU utilization」——**TTFT 最优、ITL 更慢、GPU 利用率更低**。
- **开启分块预填充后**，策略换成 **decode 优先**：先把所有待解码请求塞进这一批，再用剩下的 `max_num_batched_tokens` 预算调度 prefill；**装不下的那个 prefill 被切开**，这一轮先吃一部分，剩下的留给下一步。
- 官方给出的收益是两条：ITL 更好（因为解码被优先）、GPU 利用率更高（**compute-bound 的 prefill 和 memory-bound 的 decode 落在同一批里**）。
- 唯一的旋钮是 `max_num_batched_tokens`。文档原话：值越小 ITL 越好（prefill 打断解码越少），值越大 TTFT 越好（能塞进更多 prefill）。**默认值本身在版本之间移动过**：v0.4–v0.6 的文档写 512（「在 A100 上 ITL 最优」，基准是 Llama-70B 与 Mixtral 8x22B），并建议「>2048 以换取吞吐」；到 v0.8.1 文档，默认值已经写成 2048。文档还提示：`max_num_batched_tokens` 等于 `max_model_len` 时，几乎等价于默认策略（只是仍然 decode 优先）。

这套做法的源头是微软的两篇工作，vLLM 文档在 chunked prefill 一节末尾就链了它们：

- **SARATHI**（arXiv 2308.16369，2023-08-31，标题里的说法就是 "Piggybacking Decodes with Chunked Prefills"）。摘要写得很直白：prefill 阶段在**小批量下就能让 GPU 算力饱和**，而 decode 阶段「每请求一次只生成一个 token」，算力利用率低——两个阶段一个打满算力、一个打不满，把它们拆开跑就是浪费。
- **DeepSpeed-FastGen**（arXiv 2401.08671，2024-01-09），把这套思路做成了 Dynamic SplitFuse。顺手做一个事实核对：2401.08671 常被二手文章标成 "Sarathi-Serve"，我去 arXiv 页面核了标题——**它是 DeepSpeed-FastGen，不是 Sarathi-Serve**。

## 🔬 一台没有 GPU 的机器怎么量「调度」

先说清楚边界：这台机器没有 GPU，也没装 vLLM，所以下面**不是 GPU 实测**，而是一个**离散事件调度模拟**。成本模型只有一行：

```
一步的耗时 = O + (这一步处理的 token 数) × T
```

`O` 是每一步的固定开销（kernel launch、调度、抽样——那些「不管你这步干多少活都要付」的钱），取 **200**；`T` 是每个 token 的算力时间，取 **1**。工作负载是本文唯一一个场景：**t=0 时，3 条序列正在解码（各还需 32 个 token），同时到达一条 4096 token 的长 prompt**。调度规则严格照文档：decode 优先占预算，剩下的预算按先进先出喂 prefill，装不下的那个被切开。

**这个模型的已知盲区必须先写在表前面**：它假定时间与 token 数严格线性成正比，所以它能算清「调度造成的排队」，但**算不出「混合批次把 GPU 利用率填满」那部分收益**（那是硬件层面的效果，见上面 SARATHI 摘要）。因此下表的「总时间」列**不能当吞吐结论读**——后面我会单独说这一列为什么在这里几乎是常数。

## 🔬 结果：尖峰矮了 6.32 倍，代价写在另外两列

| chunk | 步数 | 总时间 | **最大 ITL** | 中位 ITL | 最大 TTFT |
|---|---|---|---|---|---|
| 128 | 33 | 10792 | **328** | 328 | 10792 |
| 256 | 32 | 10592 | **456** | 456 | 7547 |
| 512 | 32 | 10592 | **712** | 203 | 5923 |
| 2048 | 32 | 10592 | **2248** | 203 | 4705 |
| 4096 | 32 | 10592 | **4296** | 203 | 4502 |
| 不分块 | 33 | 10792 | **4499** | 203 | 4296 |

## 💻 可运行代码

```python
def simulate(chunk, prefills=(4096,), n_decode=3, decode_len=32, O=200.0, T=1.0):
    """chunk=None 表示不分块：一个 prefill 独占一整步，期间解码全部停摆。"""
    now, steps = 0.0, 0
    decodes = [[decode_len, [0.0]] for _ in range(n_decode)]   # [剩余 token, 输出时间戳]
    pending, ttft = list(prefills), []
    while any(d[0] > 0 for d in decodes) or pending:
        tokens = 0
        served_d, served_p = [], []
        if chunk is None:
            if pending:                      # 一整段 prefill 独占这一步
                tokens = pending.pop(0)
                served_p.append(0)
            else:                            # 没有 prefill 排队，纯解码
                served_d = [d for d in decodes if d[0] > 0]
                tokens = len(served_d)
        else:
            budget = chunk
            cand = [d for d in decodes if d[0] > 0]
            served_d = cand[:budget]         # ① 解码优先
            budget -= len(served_d)
            tokens = len(served_d)
            while budget > 0 and pending:    # ② 剩余预算喂给 prefill 分块
                take = min(pending[0], budget)
                pending[0] -= take
                budget -= take
                tokens += take
                if pending[0] == 0:
                    served_p.append(0)
                    pending.pop(0)
        now += O + tokens * T                # 一步的耗时 = 固定开销 + token 数
        steps += 1
        for d in served_d:
            d[0] -= 1
            d[1].append(now)
        ttft += [now] * len(served_p)
    itls = [b - a for d in decodes for a, b in zip(d[1], d[1][1:])]
    return steps, round(now), round(max(itls)), round(sorted(itls)[len(itls) // 2], 1), round(max(ttft))


for ch in (128, 256, 512, 2048, 4096, None):
    print(ch, simulate(ch))
```

本机真实输出（与上面表格逐格一致）：

```
    128    33    10792      328      328     10792
    256    32    10592      456      456      7547
    512    32    10592      712      203      5923
   2048    32    10592     2248      203      4705
   4096    32    10592     4296      203      4502
  不分块    33    10792     4499      203      4296
```

## 🔬 从表里能读出来的三件事

**1. 最大 ITL 精确等于 `O + chunk`。** 这不是拟合，是逐行对得上的恒等式：200+128=328、200+256=456、200+512=712、200+2048=2248、200+4096=4296。不分块那一行是 4499 = 200+4096+3——多出来的 3 是「抢占的那一步」里还捎带了 3 个正在解码的 token。所以**尾延迟的定价权完全在 chunk 手上**：想压尾延迟，先压每步的固定开销 `O`，再压 chunk，没有第三条路。反过来说，任何声称「开了分块预填充所以延迟变好了」而不报 chunk 大小的说法，等于没报数字。

**2. TTFT 是明码标价的代价，而且涨得比 ITL 掉得快。** 从 chunk=512 走到 4096：最大 ITL 涨到 **6.03 倍**（712→4296），换回的是 TTFT 只降了 **24.0%**（5923→4502）。反过来把 chunk 压到 128：最大 ITL 从 712 掉到 328（2.17×），TTFT 却从 5923 涨到 10792（**+82.2%**，相对不分块是 **+151.2%**）。原因很朴素——预算里解码先吃，chunk 越小、prefill 每步能分到的越少，它就得排更多步；而每一步都要再付一次 `O`。

**3. 最反直觉的是中位 ITL 那一列。** chunk=512 时中位 ITL 只有 203，跟不分块一样；但 chunk=256 涨到 456，chunk=128 涨到 328。区别在于：4096 个 prefill token 在 chunk=512 时只占 **9 步**，剩下 23 步是「纯解码的便宜步」（200+3=203）；而 chunk=256/128 时 prefill 把 32 步全占满了，每一步都在被抬高。**所以分块太小，你买到的不是「延迟变好」，而是「没有尖峰但全程都在浪」**——这正好解释了 vLLM 的默认值为什么会从 512 走到 2048：用一点尖峰，换回平峰。

## 🔬 场景 B：4 条长 prompt 同时到达

把工作负载换成 4 条 4096 token 的 prompt 同时到达（没有解码序列），看的就不是尖峰而是 TTFT 阶梯：

| chunk | 步数 | 总时间 | 4 条请求的 TTFT |
|---|---|---|---|
| 512 | 32 | 22784 | 5696 / 11392 / 17088 / 22784 |
| 2048 | 8 | 17984 | 4496 / 8992 / 13488 / 17984 |
| 4096 | 4 | 17184 | 4296 / 8592 / 12888 / 17184 |
| 不分块 | 4 | 17184 | 4296 / 8592 / 12888 / 17184 |

两点值得记下：**chunk=4096 与「不分块」逐格相同**——这跟文档那句「`max_num_batched_tokens` 等于 `max_model_len` 时几乎等价于默认策略」对上了，算是模型的一个自检；另外 chunk=512 时整条阶梯整体抬 **32.6%**（最大 TTFT 17184→22784），**没有一条请求变快**。分块预填充在纯 prefill 负载下是纯成本——它的收益必须靠「同时有解码在跑」才能兑现。

## 💡 落地清单

- 尾 ITL 的定价公式是 `O + chunk`。要在报告里写分块预填充的收益，就同时给出 chunk 值，否则数字没有意义。
- chunk 大小是一张汇率表：512↔4096 之间是「6.03× 的尖峰」换「24% 的 TTFT」。挑 chunk 就是决定这两条曲线的交换点在哪——而交换点取决于你更怕哪种投诉：用户说「它卡住了」，还是用户说「它半天没开始」。
- 这个模型里**总时间几乎是常数**（10592 vs 10792，差 1.9%，因为总 token 数没变）。所以**分块预填充是一种延迟整形，不是吞吐技巧**——至少在我这套线性成本模型里不是。
- 真实系统里确实有吞吐收益，但它来自另一个机制：SARATHI 摘要说的「prefill 小批量即打满算力、decode 一次一个 token 利用率低」，把 decode 搭进 prefill 那一步是搭便车。**我这份模型量不到它**，因为它假定 token 与时间严格成正比；要量这个得看硬件，得看 GPU。
- 如果你的目标只是「把尾 ITL 钉住」而**不想调参**：vLLM 自己的 disaggregated prefill 文档给的答案是把 prefill 和 decode 拆成两个实例（P/D 分离），原话是分块预填充配一个「合适的 chunk 大小」也能达到同样目的，「但实践中很难找到那个正确的值」。这句话本身就是今天这条赛道的现状注脚。

---

*本文的实验脚本 `40-chunked-prefill.py` 与本机真实输出保存在 `~/scripts/blog_daily/`。模型参数（O=200、T=1）是本文的建模选择，不是硬件测量值；换掉 `O` 会改变上面所有绝对值，但 `最大 ITL = O + chunk` 这个恒等式不变（文中矩阵已按 O=50/200/800 三档验证过）。*
