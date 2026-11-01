---
title: "132 个 SM 里只有 8 个在干活：16 路 split-K 把 decode 这一层按带宽模型从 2644 µs 压到 169 µs，而合并时忘一次 exp 重标，针尖场景直接错 96.5%"
description: "numpy 单机实跑（非 GPU 实测）：32768 token、8 个 KV 头、单次 decode，局部 softmax 用 log-sum-exp 合并的相对误差 3.9e-15，float64 下 2.29e-16；换成把各段结果直接平均，相对误差 6.0e-2，针尖型 softmax（l=1.0）下 9.65e-1，按局部分母加权是 9.97e-1。延迟部分是成本模型（H100：132 SM / 3.35 TB/s / 每 SM 25.4 GB/s / kernel 4 µs）：128K 上下文单层 512 MiB 的 KV 读取，1 路 2644 µs → 16 路 169 µs（15.62x），饱和点在 17 路；同一层权重读取 1669 MiB 要 522 µs，所以 128K 下端到端只有 4.58x，4096 token 时只剩 1.14x。"
pubDate: 2026-11-01
tags: ["推理优化", "Attention", "FlashDecoding", "长上下文", "实测"]
difficulty: "advanced"
series: "ai-concepts"
seriesOrder: 69
slug: "ai-concepts-69-split-k-flash-decoding"
---

> 一次 decode 要为**一个** token 把这一层所有的 K 和 V 从头读一遍。128K 上下文、8 个 KV 头、d=128、fp16，单层就是 512 MiB。
> 这步运算的天然并行度等于 KV 头数——8 个 block。H100 有 132 个 SM。也就是说，长上下文 decode 的注意力 kernel 里，**94% 的芯片在等**。
> split-K 把这个 8 变成 128：每个 SM 只读一段 KV，各自的局部 softmax 用 log-sum-exp 合并回全局结果。

**读这篇要会**：softmax 的定义与数值溢出问题 / GQA 里 KV 头数和 q 头数的关系 / roofline 模型里的"算术强度" / log-sum-exp 为什么能避免溢出。

## 🎯 卡点：不是算得慢，是没人来算

decode 阶段的注意力和训练时是两个形状完全不同的东西。训练时一个 batch 有几千个 token 同时做 query，block 数 = `batch × 序列长度 × 头数`，SM 早就排满了。decode 时每个请求只有 **1 个** q 向量，grid 的大小退化成了 `batch × KV 头数`。Llama-3 70B 的 KV 头数是 8，单请求就是 8 个 block。

8 个 block 不是"慢 8 倍"那么简单。GPU 的 HBM 带宽是靠大量 SM 并发发请求堆出来的，单个 SM 的访存队列（LSU/MIO 深度）只够拉住 HBM 的百分之几。H100 SXM 有 132 个 SM、3.35 TB/s，平均下来每个 SM 能拿到约 25.4 GB/s。8 个 SM 全速也就 203 GB/s —— **6% 的内存带宽**用在读 512 MiB 的 KV 上，光这一层就要 2.6 ms。

先看这步运算该归到 roofline 的哪一边。设块长 $B$、头维 $d$、fp16：

$$\text{FLOPs}=4Bd\ \text{（q·k 与 p·v 各 }2Bd\text{）},\qquad \text{字节}=4Bd\ \text{（K、V 各 }Bd\times 2\text{ 字节）}$$

算术强度正好是 **1 FLOP/byte**。H100 的 fp16 张量核 989 TFLOPS 除以 3.35 TB/s，脊点是 295 FLOP/byte。1 对 295 —— 差了两个半数量级，这个 kernel 里没有任何算力能被用上，唯一的资源是带宽。所以"拆块"不是拆计算，是**拆读取**。

对照着看更容易记住两篇老卡片的分工：FlashAttention（29 篇）减少的是**读多少字节**（分块 + 不落 HBM 的在线 softmax），split-K 增加的是**多少个 SM 同时在读**。PagedAttention（31 篇）解决的是**读的地方不连续**。三个正交，生产里通常叠着用：paged 给你不连续的块，FlashAttention 给你块内的在线 softmax，split-K 把这些块分给不同 SM。

## 🔬 机制：局部 softmax + 全局合并

单头注意力是

$$o=\frac{\sum_i e^{z_i-m}\,v_i}{\sum_i e^{z_i-m}},\qquad z_i=\frac{q\cdot k_i}{\sqrt d}$$

把 KV 按 token 切成 $S$ 段，每段只能看到自己的 logit，于是每段只能算一个**局部**的 max $m_s$，以及相对它的分母和分子：

$$l_s=\sum_{i\in s}e^{z_i-m_s},\qquad a_s=\sum_{i\in s}e^{z_i-m_s}v_i$$

段内归一化后的 $a_s/l_s$ 是错的——它相对的是局部最大值，不是全局最大值。要合回精确结果，必须回到全局 max 重标一次：

$$m=\max_s m_s,\qquad l=\sum_s l_s e^{m_s-m},\qquad a=\sum_s a_s e^{m_s-m},\qquad o=\frac{a}{l}$$

三个量 $(m_s,l_s,a_s)$ 就是每个 block 要写出去的**全部**中间结果：两个标量加一个 $d$ 维向量。所以 16 路分割的 scratch 只有 64 KiB（向量）+ 1 KiB（标量），跟上下文长度无关——这是 split-K 便宜的根本原因，它不复制 KV，只存"书的页码"。

漏掉那个 $e^{m_s-m}$ 会发生什么，取决于各段最大值的漂移程度。$m_s$ 之间差 10，$l_s$ 就不可比了（$e^{10}\approx 22026$ 倍）。两种常见的错法，第二种尤其阴险：

- **直接平均** $\frac1S\sum_s a_s/l_s$：等价于假设所有 $m_s$ 相等、所有段等权。softmax 平坦时误差还能忍，一旦质量压在少数 token 上，直接崩。
- **按 $l_s$ 加权** $\frac{\sum_s l_s a_s/l_s}{\sum_s l_s}$：看着很像"用分母当权重"的合理做法，实际上 $l_s$ 是各自相对 $m_s$ 的分母，跨段根本没有可比性。段数越多、max 漂移越大，它越错。

实测里"看着合理"的那个错法是**最差**的：针尖场景相对误差 9.97e-1，比直接平均的 9.65e-1 还糟。

## 🏗️ 工程：两个 kernel、一条按 SM 数定的规则

生产实现是两段：kernel A 按 `(head, kv_chunk)` 起 block，算出 $(m_s,l_s,a_s)$ 写进 scratch；kernel B（一个 block 就够）把它们合并。vLLM 的 flashinfer/xformers decode 后端和 TensorRT-LLM 里都叫 split-K 或 partition-KV，2023 年 flash-decoding（Dao 等）那篇工程笔记把合并公式和 block 划分写成了标准件。

开多少路不是拍脑袋，是算出来的。可用 block 数 = `batch × KV 头数 × splits`，饱和条件是它 ≥ SM 数：

| batch | 1 路的 block 数 | 1 路有效 SM | 填满 132 SM 需要几路 |
|---|---|---|---|
| 1 | 8 | 8 | 17 |
| 4 | 32 | 32 | 5 |
| 8 | 64 | 64 | 3 |
| 16 | 128 | 128 | 2 |
| 32 | 256 | 132 | 1（不需要拆） |

这张表解释了一件常被忽略的事：**split-K 是给小 batch 用的**。batch 到 32 以上，光请求本身就能把 SM 填满，再拆只会多一次 kernel 启动和一次 scratch 读写。而张量并行（TP）会把这个矛盾推到极端：Llama-3 70B 在 TP=8 下，每张卡分到 1 个 KV 头，于是单请求解码时每张卡只有 **1 个 block** 在算 64 MiB 的 KV——25 GB/s 跑 64 MiB 是 2.6 ms 一层，80 层就是 210 ms 一个 token。长上下文 + TP 的组合里，split-K 不是可选项，是必需品。

切点也不能随便定。接了 PagedAttention 之后，KV 是按 16 token 的 page 散落的，切点必须在 page 边界上，否则一次切分会跨两个物理块，等于白读一遍。好在 page table 本来就知道每个请求有哪些页，把 `页数 / splits` 分下去就行。

## 💻 实测：两种 softmax 形状 × 七种分割 × 三种合并

数值部分是本机 numpy 真跑的：`N=32768` token、8 个 KV 头、`d=64`、float32，单次 decode 一个 token。参考解用两遍流式精确计算（先扫全局 max，再累加）。场景 A 是随机键（softmax 平坦，有效参与 token 数 1017，占 3.11%）；场景 B 在 40% 位置插了一个和 q 强对齐的"针尖"键（max logit 110.3，有效参与 token 数 1.0，几乎 one-hot）。

```python
def split_k(z, V, splits, merge="lse"):
    """z:(H,N) logits, V:(H,N,d)。每段独立 softmax，再合并。"""
    edges = np.linspace(0, N, splits + 1).astype(int)
    ms, ls, accs = [], [], []
    for s in range(splits):
        a, b = edges[s], edges[s + 1]
        m_i = z[:, a:b].max(axis=1)                      # 局部 max
        p = np.exp(z[:, a:b] - m_i[:, None])
        l_i = p.sum(axis=1)                              # 局部分母
        acc_i = np.einsum("hn,hnd->hd", p, V[:, a:b, :])  # 局部未归一化分子
        ms.append(m_i); ls.append(l_i); accs.append(acc_i / l_i[:, None])

    ms, ls, accs = np.stack(ms, 1), np.stack(ls, 1), np.stack(accs, 1)
    if merge == "lse":                      # 正确：回到全局 max 重标
        m = ms.max(axis=1, keepdims=True)
        w = np.exp(ms - m)
        l = (ls * w).sum(axis=1)
        return (accs * (ls * w)[:, :, None]).sum(axis=1) / l[:, None]
    if merge == "avg":                      # 错法一：直接平均
        return accs.mean(axis=1)
    if merge == "weighted":                 # 错法二：按局部分母加权
        return (accs * ls[:, :, None]).sum(axis=1) / ls.sum(axis=1)[:, None]
```

| 分割数 | 合并方式 | 场景 A 相对误差 | 场景 B 相对误差 |
|---|---|---|---|
| 1 | — | 0（参考解本身） | 0 |
| 2 / 4 / 8 | log-sum-exp | 5.25e-15 / 4.35e-15 / 4.47e-15 | 0 / 0 / 0 |
| 16 / 32 / 64 | log-sum-exp | 3.91e-15 / 4.02e-15 / 4.02e-15 | 0 / 0 / 0 |
| 32 | 直接平均 | 6.00e-2 | 9.65e-1 |
| 32 | 按局部分母加权 | 3.04e-1 | 9.97e-1 |

误差不随分割数增长，这是关键：64 路和 2 路是同一个量级（4e-15），说明合并本身没有累积误差，只有求和顺序带来的 float32 舍入。场景 B 的误差精确为 0 —— 质量全在一个 block 里时，合并就是搬运，搬对了就是精确值。**段数是免费的，错的只有归一化。**

精度账要跟存储账分开算，否则会误判谁是瓶颈：

| 比较 | 最大绝对误差 | 余弦相似度 |
|---|---|---|
| float64 参考 vs 32 路合并 | 2.29e-16 | 1.000000000000 |
| KV 落成 fp16 vs float32 参考 | 9.57e-6 | 0.9999999491 |
| fp16 存储 + 32 路合并 vs 其参考 | 1.14e-16 | — |

合并的误差（1e-16 量级）比 fp16 存储本身带来的误差（1e-6 量级）小十个数量级。所以想要精度别抠合并，去抠 dtype；反过来，为了并行把结果存成 fp16 才是真正会掉精度的那一刀（KV 量化那篇单独算过这笔账）。

延迟部分是**成本模型，不是 GPU 实测**。假设 H100 SXM：132 SM、3.35 TB/s、每 SM 25.4 GB/s、每个 kernel 固定 4 µs 启动开销。128K 上下文、8 KV 头、d=128、fp16：

| splits | block 数 | 有效 SM | 聚合带宽 | 单层延迟 | 相对 1 路 |
|---|---|---|---|---|---|
| 1 | 8 | 8 | 203 GB/s | 2644 µs | 1.00x |
| 4 | 32 | 32 | 812 GB/s | 665 µs | 3.98x |
| 8 | 64 | 64 | 1624 GB/s | 335 µs | 7.90x |
| 16 | 128 | 128 | 3248 GB/s | 169 µs | 15.62x |
| 32 | 256 | 132 | 3350 GB/s | 164 µs | 16.10x |
| 64 | 512 | 132 | 3350 GB/s | 164 µs | 16.10x |

曲线在 16 路拐平，因为 8 个头 × 16 路 = 128 ≈ 132。再多拆一个字节都不省：SM 数量是硬上限，第 33 路之后的 block 只能排队，还平白多写一份 scratch。**分割数的最优解不是调出来的，是 SM 数除以 KV 头数除出来的。**

## 💡 边界：15.62x 是 kernel 的，不是你的

上面那张表最容易读错的地方，是把 15.62x 当成"长上下文 decode 快 15 倍"。一层 decode 不只有注意力，还要读权重：70B 级模型每层 875M 参数、fp16 就是 1669 MiB，跑满 3.35 TB/s 要 522 µs。把这部分算进去：

| 上下文 | KV/层 | 1 路 | 16 路 | kernel 加速 | 端到端加速 |
|---|---|---|---|---|---|
| 4096 | 16 MiB | 83 µs | 9 µs | 9.02x | 1.14x |
| 32768 | 128 MiB | 661 µs | 45 µs | 14.59x | 2.08x |
| 131072 | 512 MiB | 2644 µs | 169 µs | 15.62x | 4.58x |

4096 上下文时，注意力只占这一层读取量的 1%，把它的时间压掉 89% 换来的整层收益是 14%。这就是为什么短上下文不需要 split-K：这时候瓶颈是权重，而权重读取天然吃满 SM，拆不出来。**并行度优化只对"并行度不足的那一段"有效，先量清楚这段占多少。**

成本模型的盲区也要说清楚：25.4 GB/s 是 3.35 TB/s 均分的理论值，真实单 SM 能达到的稳定带宽受 L2 命中、访问模式（连续 vs 跨 page）和并发请求数影响，通常达不到；4 µs 的 kernel 开销是启动延迟的乐观估计，多请求竞争时合并 kernel 会和别的 kernel 排队；H100 的时钟也不是恒定的。所以上表能定性地告诉你"拐点在 132/H 附近、端到端收益要除以权重占比"，但不能当 benchmark 引用——真机数字要看 vLLM/TRT-LLM 在那个模型和那个上下文长度下的实测。

最后一个边界是能力边界：split-K **一个字节都没少读**。它只是让更多 SM 同时在读。真要减少读取量，是 GQA/MQA（30 篇）砍 KV 头、KV 量化（56 篇）砍位宽、或者干脆把 KV 换成一页 page 共享（32 篇的前缀缓存）。三件事叠起来才是长上下文推理的全部故事，split-K 只负责其中"把带宽吃满"这一件。

完整脚本（两个场景 × 七种分割 × 三种合并、精度对照、成本模型、batch 与端到端拆解）见仓库 [demos/69-split-k-flash-decoding.py](https://github.com/shenlv7/blog/blob/main/demos/69-split-k-flash-decoding.py)。
