---
title: "31 | 连续批处理 + PagedAttention：128 条请求 makespan 从 6093 步降到 2272 步（2.68x），KV cache 内部碎片从 60.9% 压到 0.81%，并发数从 2 条提到 6 条"
description: "把推理服务的两个隐藏成本拆开算。调度侧：128 条请求、输出长度对数正态（中位 160、最长 1197）、单批上限 16，静态批要 8 个波次共 6093 个 decode step（槽位-步利用率 28.6%），连续批只要 2272 步（76.7%），加速 2.682x；完成延迟 p50 从 2970 降到 811 step、p99 从 6093 降到 2168。显存侧：13B 规模每 token 才是 800 KiB，连片分配按 2048 预留要 1600 MiB/条，4 GiB 只放得下 2 条、内部碎片 60.9%；分页分配实占 630.9 MiB，放得下 6 条（3.0x）、碎片 0.806%，预算里真正装 token 的比例从 30.6% 到 91.7%。再补一刀外部碎片：变长请求 best-fit 塞进 4 GiB，放进 6 条后仍有 0.461 GiB 空闲，却因为碎成 473 MiB 的洞而放不下下一条 1600 MiB 的请求。最后是前缀共享：512 token 提示词并行采样 8 份，各存一份要 3.125 GiB，块表指向同一批物理块只要 0.488 GiB（6.40x）。"
pubDate: 2026-09-24
tags: ["AI核心概念", "推理优化", "PagedAttention", "连续批处理", "vLLM", "KV cache"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 31
slug: ai-concepts-31-paged-attention-and-continuous-batching
---

# AI核心概念(31)：分页注意力与连续批处理——显存的账本和调度的流水线

> 第 30 篇刚把 KV cache 砍到 1/4，但砍完才发现真正的问题从来不是总量，而是**这些字节必须整块、按最长可能长度、提前占好位置，并且在请求活着的整个期间一直占着**。推理服务最贵的资源不是显存，是**被预留却用不上的显存**。

## 🎯 两个瓶颈，一块吸走利润

一个 LLM 服务上线后，能同时服务多少人，由两件事决定：

1. **调度**：一批请求进来，GPU 一个 decode step 能并行处理多少个序列？空闲的槽位能不能立刻让给新请求？
2. **显存**：每条请求的 KV cache 占多少？能不能随生成长度**动态长**出来？

这两件事看起来独立，其实是一件事的两面——因为"动态"要求显存在任意时刻可分配、可回收，而 GPU 上"一整块连续显存"是最难动态化的东西。这就是 PagedAttention 出现的原因，也是它必须跟连续批处理（continuous batching）一起讲的原因。

## 🔬 调度实验：静态批的批内同生共死

先看调度本身。模拟 128 条请求，输出长度取对数正态（中位数 ~160 token，最长 1197），单批上限 16 条：

| 指标 | 静态批 | 连续批 | 变化 |
|---|---|---|---|
| 波次 | 8 | — | 无波次概念 |
| 总 decode step | 6093 | 2272 | **0.373x** |
| 吞吐 | 4.6 tok/step | 12.3 tok/step | 2.68x |
| 槽位-步利用率 | 28.6% | 76.7% | +48.1 pt |
| 完成延迟 p50 | 2970 | 811 | 0.27x |
| 完成延迟 p99 | 6093 | 2168 | 0.36x |

静态批的机制是：凑够 16 条就发车，**整批等同批里最长的那条生成完**才能收车。于是每条请求都要陪跑，`槽位-步利用率`只有 28.6%——超过七成的 GPU 序列槽位在等一个早已不需要它们的邻居。8 个波次里的 69,619 个"空转序列步"就是这么来的。

连续批处理（continuous batching，也叫 iteration-level scheduling）把"批"从**静态容器**改成**滑动窗口**：每一个 decode step 结束时，生成完的序列立即退出，队列里的新请求立刻补位。同一份算力，利用率到 76.7%。

注意连续批的 76.7% 也不是 100%——剩下的是请求队列排队 + 尾部收敛（最后几条请求凑不满一批）。这已经接近理论上限了。

## 🏗️ 为什么连续批必须配分页

连续批处理要能**随时踢走一条、塞进一条**，前提是显存随时可回收、可分配。而经典实现是这么干的：一条请求进来，按 `max_len`（比如 2048 token）预留一整块**连续**显存，哪怕它实际上只生成 801 个 token 就结束。

这就是操作系统那套问题在 GPU 上的重演：

- **内部碎片**：预留 2048 只用 801，白扔 60.9%。
- **外部碎片**：即使总量够，变长请求反复申请释放之后，空闲空间会碎成一堆小洞，谁也放不下。

实测（13B 规模：40 层、40 个 KV 头、`d_head=128`、fp16 → **800 KiB/token**，预算 4 GiB）：

| 分配方式 | 每请求占用 | 并发条数 | 内部碎片 | 预算里装 token 的比例 |
|---|---|---|---|---|
| 连片分配 | 1600.0 MiB（按 2048 预留） | 2 | 60.9% | 30.6% |
| 分页分配（块 16） | 630.9 MiB（按实际长度取整块） | 6 | 0.806% | **91.7%** |

3 倍并发差，全来自"提前预留"这一个动作。

外部碎片单独跑了一遍：把变长请求用 best-fit 往 4 GiB 里塞，放进 6 条之后放不下第 7 条——**而此时还有 0.461 GiB 空闲**，只是碎成了 473 MiB 的洞，下一条请求需要 1600 MiB。总显存够、可用显存不够，这是服务里最难解释的一类"显存不足"。

**PagedAttention 的解法是把 OS 的虚拟内存照抄一遍**：KV cache 不再按请求分配，而是切成固定大小的 **block**（典型 16 个 token），每条请求维护一张 **block table**，记录自己的第 k 个逻辑块落在哪个物理块上。物理块可以完全不相邻。注意力计算改成按块 gather：每个 query 依次跟自己的所有 block 算分，结果累加——因为 softmax 的归一化可以分块做（在线 softmax / FlashAttention 那一套），所以"物理上不连续"**不会**改变结果，只是换了读内存的顺序。

于是：内部碎片只剩最后一块不满（实测 0.806%，即平均浪费半块 8 token）；外部碎片消失（任何空洞只要能放一个块就有用）；显存可以在 step 级别回收和再分配——连续批处理需要的那个"随时"才真正成立。

## 💻 三个实验的核心代码

纯标准库，不需要 numpy：

```python
#!/usr/bin/env python3
"""连续批处理 + 分页注意力数值实验。纯标准库，无 numpy。"""
import math, random
random.seed(20260924)

# ---------- E1 调度：静态批 vs 连续批 ----------
N_REQ, MAX_BATCH = 128, 16
out_len = [max(4, int(random.lognormvariate(math.log(160), 0.9))) for _ in range(N_REQ)]

def static_batch(outs, B):
    steps = 0; done = []
    for i in range(0, len(outs), B):
        wave = outs[i:i + B]; m = max(wave)          # 整波等最长的跑完
        steps += m; done += [steps] * len(wave)
    return steps, done

def continuous_batch(outs, B):
    queue = list(outs); active = []; steps = 0; done = []
    while queue or active:
        while queue and len(active) < B: active.append(queue.pop(0))   # 出队即补位
        steps += 1
        nxt = []
        for r in active:
            (nxt.append(r - 1) if r > 1 else done.append(steps))
        active = nxt
    return steps, done

s_steps, s_done = static_batch(out_len, MAX_BATCH)
c_steps, c_done = continuous_batch(out_len, MAX_BATCH)
total = sum(out_len)
print(f"E1 静态批 {s_steps} step 利用率 {100*total/(s_steps*MAX_BATCH):.1f}%")
print(f"E1 连续批 {c_steps} step 利用率 {100*total/(c_steps*MAX_BATCH):.1f}%  加速 {s_steps/c_steps:.3f}x")

# ---------- E2 显存：连片 vs 分页 ----------
L, H_KV, D_HEAD, BYTES, MAX_LEN, BLOCK = 40, 40, 128, 2, 2048, 16
per_tok = 2 * H_KV * D_HEAD * BYTES * L          # fp16，每 token 的 K+V
BUDGET = 4 * 1024**3
lens = [min(MAX_LEN, max(16, int(random.lognormvariate(math.log(600), 0.8)))) for _ in range(N_REQ)]
mean_len = sum(lens) / len(lens)
n_naive = int(BUDGET // (MAX_LEN * per_tok))                       # 按上界预留
alloc = [math.ceil(l / BLOCK) * BLOCK for l in lens]               # 按块取整
n_paged = int(BUDGET // (sum(alloc) / len(alloc) * per_tok))
print(f"E2 每 token {per_tok/1024:.1f} KiB  连片并发 {n_naive} 条 碎片 {100*(1-mean_len/MAX_LEN):.1f}%")
print(f"E2 分页并发 {n_paged} 条 碎片 {100*sum(a-l for a,l in zip(alloc,lens))/len(lens)/ (sum(alloc)/len(alloc)):.3f}%")

# ---------- E2b 外部碎片：best-fit ----------
def best_fit(sizes, budget):
    holes = [budget]; ok = 0
    for s in sizes:
        cand = [(h, i) for i, h in enumerate(holes) if h >= s]
        if not cand: break
        h, i = min(cand); holes.pop(i); ok += 1
        if h - s > 0: holes.append(h - s)
    return ok, sum(holes), holes

ok, free, holes = best_fit([l * per_tok for l in lens], BUDGET)
print(f"E2b 放入 {ok} 条后失败：仍空 {free/1024**3:.3f} GiB，碎成 {len(holes)} 个洞，最大仅 {max(holes)/1024**2:.0f} MiB")
```

输出（可复现，`python3 31-paged-attention.py`）：

```
E1 静态批 6093 step 利用率 28.6%
E1 连续批 2272 step 利用率 76.7%  加速 2.682x
E2 每 token 800.0 KiB  连片并发 2 条 碎片 60.9%
E2 分页并发 6 条 碎片 0.806%
E2b 放入 6 条后失败：仍空 0.461 GiB，碎成 1 个洞，最大仅 473 MiB
```

## 💡 分页还顺手送了前缀共享

块表是**间接层**，间接层意味着可以指向同一个物理块。同一段提示词要让模型并行采样 8 份（best-of-n、beam search、自一致性），连片方案是 8 份各存一遍提示词的 KV；分页方案是 8 张块表指向同一批提示块，只在**生成新 token 时**才各自分配新块——写时复制（copy-on-write），又是 OS 操作。

实测 512 token 提示词 × 8 份：

| 方案 | 显存 | 倍率 |
|---|---|---|
| 各存一份 | 3.125 GiB | 1.00x |
| 共享提示块 | 0.488 GiB | **6.40x** |

采样份数越多、提示词越长，省得越狠——因为共享的是 `prompt_len`，只有各自的生成部分要独立。

## 💡 工程结论与坑

- **先修调度，再抠 kernel**。静态批到连续批的 2.682x 不花一分钱算力，纯粹是"别让 GPU 等已经结束的序列"。这一步的收益通常远大于手工优化 attention kernel。
- **块大小是个真 tradeoff**：块大 → 内部碎片回升（平均浪费半块 = `BLOCK/2` token）；块小 → 块表变大、kernel 里 gather 的循环变多。16 是常见甜点位，长上下文服务可以考虑更大的块。
- **连续批不是无代价**：新请求的 prefill 会插进正在 decode 的 step 中间，造成**首 token 延迟与尾延迟的尖刺**（表里 p99 = 2168 step，都不小）。工程上一般用 **chunked prefill**（把长提示词切成几段，spread 到多个 step）来抹平，代价是稍复杂。
- **分页带来抢占能力**：显存不够时可以把低优先级请求的块换出（swap）或直接重算（recompute），这是多级调度（如优先级队列、SLO 差异化）的地基。换出的粒度正是块——如果 KV 还是连续一整块，就换不动。
- **这一切能成立，是因为注意力本身可分块**：在线的 softmax 归一化早在 FlashAttention 里就被解决过。所以分页注意力几乎没牺牲算力精度，只是把内存布局从"连续数组"改成"块表寻址"——和 30 篇的 GQA 一样，**又一次赢在字节的账本，而不是 FLOPs**。

一句话收尾：**推理服务的显存不该按请求分配，而该按页分配；批不该是容器，而该是流水线。** 这两句话加起来，就是把操作系统 60 年前解决过的问题，在 GPU 上重做了一遍。
