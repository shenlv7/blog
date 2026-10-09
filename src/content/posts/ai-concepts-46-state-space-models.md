---
title: "46 | State Space Models（状态空间模型）：推理状态是常数，代价是你的记忆有上限"
description: "Mamba 那一系被反复讲的卖点——推理期状态 O(1)、不用 KV cache——这篇不重复结论，只量它的账。本机纯 CPU numpy 三段实测：(1) 选择性 SSM 的顺序递推与并行前缀扫描最大绝对误差 3.553e-14，逐元素一致，说明 O(L log L) 的并行训练是真的；但朴素 numpy 实现下，L=512 时扫描 73.2 ms、比 attention 的 17.0 ms 还慢，交叉点在 L≈4096（此处 scan 675 ms vs attn 883 ms），L=8192 时 attention 已是 scan 的 2.46 倍——常数项和实现方式决定谁快，论文里的 5x 吞吐不是免费午餐。(2) 推理状态做账：一个 GQA 配置的 KV cache 是 4 KB/token，128K 上下文 512 MiB，而 Mamba-2 量级的状态 0.75 MiB 恒定，比值 682.7x，1M 上下文是 5461x。(3) 代价在这里：用固定维线性记忆代理测关联召回，状态维 16 时 4 对 key-value 命中率 74.5%、8 对掉到 28.0%；状态维 128 时 16 对 92.5%、32 对 52.0%。容量随状态维线性涨、与序列长度无关，而 attention 在所有档位都是 100%。"
pubDate: 2026-10-09
tags: ["AI核心概念", "State Space Models", "状态空间模型", "Mamba", "线性注意力", "KV Cache", "长上下文", "架构"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 46
slug: ai-concepts-46-state-space-models
---

# AI核心概念(46)：State Space Models（状态空间模型）——推理状态是常数，代价是你的记忆有上限

> 「Mamba 推理时状态恒定、不存 KV cache、序列长度线性扩展，5 倍吞吐」——这句话是对的，但它漏了后半句：**状态恒定意味着你的记忆容量也恒定**。省下来的显存，是从「能存多少东西」里扣出来的。
>
> 这篇不重复卖点，只量账。三段实验全部本机纯 CPU numpy 跑出，不调 API、不用 GPU：第一段验证选择性 SSM 的顺序递推与并行前缀扫描确实等价（最大绝对误差 `3.553e-14`），并给出同一份 numpy 下扫描 vs attention 的墙钟时间——**朴素实现下 L=512 时扫描反而更慢，交叉点在 L≈4096**；第二段算推理期状态大小，KV cache 从 4 KB/token 一路涨到 1M 上下文 4096 MiB，而 SSM 状态恒定 0.75 MiB；第三段是这篇的重点：用固定维记忆做关联召回，量出「状态维 → 能存多少对 key-value」的真实天花板。
>
> 一句话预告：**状态维 16 时，4 对 key-value 还能命中 74.5%，8 对就掉到 28.0%；状态维 128 时，16 对 92.5%、32 对 52.0%——而 attention 在所有档位都是 100%。**

## 🎯 术语与出处对齐

- **SSM（state space model，状态空间模型）**：把序列建模写成一个线性时不变系统的离散递推：`h_t = A h_{t-1} + B x_t`，`y_t = C h_t`。`h` 是**固定维**的状态向量，与序列长度 `L` **无关**——这是它推理便宜的全部来源。
- **S4**（Gu、Goel、Ré，*Efficiently Modeling Long Sequences with Structured State Spaces*，arXiv:2111.00396，ICLR 2022）：把 SSM 的参数矩阵 `A` 参数化成结构化形式（HiPPO / 对角化），让它可以**并行卷积训练**，第一次在长程依赖基准（Long Range Arena）上打平甚至超过 Transformer。它的问题是**时不变**——参数不随输入变，表达力被卡住。
- **Mamba / S6**（Gu、Dao，*Mamba: Linear-Time Sequence Modeling with Selective State Spaces*，arXiv:2312.00752，2023）：核心改动是**选择性（selective）**——让 `Δ`、`B`、`C` 都变成输入 `x_t` 的函数，于是模型可以「决定记住什么、忘掉什么」。代价是：参数随时间变，卷积形式不再成立，只能走**并行前缀扫描（parallel scan）**。
- **Mamba-2 / SSD**（Dao、Gu，*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*，arXiv:2405.21060，ICML 2024）：论文标题就是它的主张——**半可分矩阵（semiseparable matrix）** 同时是 SSM 递推和 masked attention 的两种写法，于是把 SSM 和注意力放进同一个数学框架，并推出比 Mamba 快 2–8 倍的 SSD 算法（论文报告值）。
- **线性注意力 / 保留网络这一族**：RWKV（arXiv:2305.13048）、RetNet（arXiv:2307.08621）跟 Mamba 在这一点上是同构的——都是把「到处看」压成「一个固定大小的汇总状态」，区别在汇总算子怎么写。所以后面第三段的容量结论，对它们**同样适用**。
- **并行前缀扫描（associative scan）**：把递推看成可结合算子 `(a₁,b₁) ∘ (a₂,b₂) = (a₂a₁, a₂b₁ + b₂)`，就能用 Hillis-Steele / Blelloch 那种并行前缀和算法把 `L` 步串行压成 `O(log L)` 层。Martin & Cundy（*Parallelizing Linear Recurrent Neural Networks over the Sequence Dimension*，2020）给了这条线的经典形式。
- **MQAR（multi-query associative recall，多查询关联召回）**：Zoology 那篇（Arora、Eyuboglu、Timalsina、Johnson、Poli、Zou、Rudra、Ré，arXiv:2312.04927，2023）提出的诊断任务，专门测「从上下文里按 key 取回 value」。它给的对照很扎心：**70M 参数的 attention 模型在关联召回上胜过 1.4B 参数的卷积门控模型**。我这篇第三段是自己写的无训练代理版，判据、盲区下面会说清。

## 🔬 实验 1：顺序递推和并行扫描，真的等价吗

Mamba 敢用「线性时间」讲性能，前提是并行扫描给出的结果和逐 token 递推**逐元素相同**——否则训练和推理就不是同一个模型了。这不是想当然，得验。

设置：`L=512`、`d_model=32`、`d_state=16`，`Δ`、`B`、`C` 都由 `x` 线性投影得到（`Δ` 过 softplus 保证为正），`A` 取负实对角。离散化用 `A_bar = exp(Δ·A)`、`B_bar = Δ·B`（零阶保持的常用简化）。然后同一组参数跑两遍：

- **顺序递推**：Python 循环，`h = A_bar[t] * h + B_bar[t] * x[t][:,None]`，`y_t = Σ_N C_t ⊙ h`；
- **并行扫描**：把 `(a,b)` 展平成 `L × (D·N)` 一维可结合算子，Hillis-Steele 双层循环翻倍归并。

结果：

```
顺序递推 vs 并行扫描：最大绝对误差 = 3.553e-14   （相对量级 1.621e-16）
是否逐元素一致（1e-12 内）= True
```

float64 下 `3.6e-14` 的绝对误差就是「同一个数被加了两遍的浮点顺序不同」的量级，相对量级 `1.6e-16` 贴着机器精度。**结论：等价，可以放心并行。**

但等价不等于快。同一份 numpy、单核、同一组参数，扫一遍不同长度：

| L | SSM 并行扫描 (ms) | 朴素 attention (ms) | 比值 attn/scan |
|---|---|---|---|
| 512 | 73.21 | 16.97 | 0.23x |
| 1024 | 169.71 | 59.70 | 0.35x |
| 2048 | 291.89 | 223.58 | 0.77x |
| 4096 | 674.98 | 882.56 | 1.31x |
| 8192 | 1379.98 | 3388.51 | 2.46x |

这张表比任何一句「线性时间」都诚实：

1. **L≤2048 时，扫描比朴素 attention 还慢。** 原因不神秘：Hillis-Steele 要跑 `log₂L` 层，每层都要在全 `L×(D·N)` 张量上做两次乘加再复制一次，`log` 层数带来的**常数因子**在短序列上根本收不回来；而 attention 那 `L×L` 打分矩阵在 L 小时只是一次干净的小 matmul。
2. **交叉点在 L≈4096**，到 8192 时 attention 已经是扫描的 2.46 倍，`O(L²)` 开始压过 `O(L log L)`。这就是为什么长上下文里 SSM 的故事成立、短序列里它反而吃亏。
3. **这个数字不能外推到 GPU。** 本机没有 GPU，这里用的是朴素 numpy；真实 Mamba 在 GPU 上用的是**硬件感知的 selective scan kernel**（把状态留在 SRAM、只让 `Δ/B/C` 过 HBM），论文报告的是 5 倍于 Transformer 的推理吞吐——那是 kernel 工程的结果，不是这两个复杂度式子的直接推论。**复杂度只决定趋势，常数项和实现决定你实际拿到什么。**

## 🏗️ 机制：为什么能并行，以及为什么推理省

**能并行的原因只有一个——递推对状态是线性的。** `h_t = a_t ⊙ h_{t-1} + b_t` 这个形式里，`h` 是线性出现的，所以「先算 `h_2` 再算 `h_3`」和「把 `(a₂,b₂)` 和 `(a₃,b₃)` 先合成一个算子再作用一次」结果一样。可结合 → 前缀扫描 → 训练时不用串行跑 `L` 步。这也是它跟 RNN 的根本区别：RNN 的 `h_t = tanh(W h_{t-1} + U x_t)` 因为 `tanh` 非线性而**不可结合**，只能老老实实串行。

**推理省的原因也只有一个——`h` 是固定维的。** attention 要生成第 `t` 个 token，必须回头查前面所有 token 的 K、V，所以每来一个 token 就往显存里追加一份，`L` 越长涨得越多。SSM 只需要维护一个 `d_model × d_state` 的矩阵 `h`，来一个 token 更新一次，**大小从头到尾不变**。拿两个真实量级的配置做账：

- **attention（Llama-3-8B 量级的 GQA）**：32 个 query 头、8 个 KV 头、head_dim=128、fp16 → 每 token 存 K 和 V 两组 = `2 × 8 × 128 × 2 = 4096` B = **4.00 KiB/token**；
- **SSM（Mamba-2 量级）**：`d_model=1536`、`d_state=128`、fp32 → 状态 = `1536 × 128 × 4 = 786432` B = **0.75 MiB 常数**。

| 上下文 L | KV cache | SSM 状态 | KV / SSM |
|---|---|---|---|
| 4 096 | 16.0 MiB | 0.75 MiB | 21.3x |
| 32 768 | 128.0 MiB | 0.75 MiB | 170.7x |
| 131 072 | 512.0 MiB | 0.75 MiB | **682.7x** |
| 1 048 576 | 4096.0 MiB | 0.75 MiB | **5461.3x** |

这就是所有「Mamba 长上下文便宜」论证的全部基础，数字上确实成立。**但请注意第二列的「常数」两个字——它既是优点，也是上限。**

## 🧪 实验 2：固定状态的账单，在「召回」这一栏

设一个最朴素的关联记忆任务：上下文里有 `P` 对随机 `(k_i, v_i)`，末尾给一个查询 `k_q`，要求读回对应的 `v_q`。记忆用**无训练的线性记忆代理**：状态 `S = Σ φ(k_i) ⊗ v_i`（形状 `m × d_val`），读出 `Sᵀ φ(k_q)`；命中判据 `cos(读出, v_q) ≥ 0.9`。`m` 就是这个记忆的状态维，对应 SSM 的 `d_state`。每格 200 次试验。

| 状态维 m | 4 对 | 8 对 | 16 对 | 32 对 | 64 对 | 128 对 |
|---|---|---|---|---|---|---|
| 16 | 74.5% | 28.0% | 7.0% | 0.0% | 0.0% | 0.0% |
| 32 | 90.0% | 65.0% | 16.5% | 1.5% | 0.0% | 0.0% |
| 64 | 96.5% | 91.0% | 59.5% | 19.0% | 1.0% | 0.0% |
| 128 | 100.0% | 98.5% | 92.5% | 52.0% | 13.0% | 4.0% |

对照组是 softmax attention，同一批 key/value，查询取「匹配 key 的放大版」：**4 对到 128 对，全是 100.0%。**

三个读法：

1. **容量随状态维线性涨，涨得还很贵。** 想稳定存住 16 对，`m=16/32/64/128` 的命中率是 7.0% → 16.5% → 59.5% → 92.5%。也就是说**大致要 `m ≳ 4×对数` 才够用**（这个系数是我的实验条件推出来的经验值，不是定理），想存 128 对，`m=128` 只有 4.0%。
2. **容量和序列长度无关——这正是关键。** 上下文从 128 token 涨到 128K token，只要里面塞了同样多的 key-value 对，SSM 那 `m` 维状态的负担**一模一样**。attention 不是这样：它的「记忆」就是原始 KV，随上下文线性增长（代价见上表第二列）。**所以这里没有免费的午餐，只有一张兑换单：省下的显存，是从召回容量里扣的。**
3. **这就是 Zoology 和 Repeat After Me 那两篇在讲的事。** Zoology（arXiv:2312.04927）用 MQAR 量出「70M 的 attention 打赢 1.4B 的卷积门控模型」；Jelassi 等（*Repeat After Me*，arXiv:2402.01032，ICML 2024）证明了**两层 Transformer 可以拷贝指数长的字符串，而固定大小潜状态的模型在原理上做不到**，并且在预训练模型上验证：从上下文里复制/检索这件事，Transformer 明显更强。

## 💻 可运行代码（本机跑通版，纯 numpy）

下面是从实验脚本里摘出的核心三段，`python3` 直接可跑（numpy 2.x，无需 GPU/API）：

```python
import numpy as np, time
rng = np.random.default_rng(0)
L, D, N = 512, 32, 16                     # 序列长度 / d_model / d_state
x = rng.normal(size=(L, D))
W_delta = rng.normal(size=(D, D)) * 0.1
W_B = rng.normal(size=(D, D, N)) * 0.3
W_C = rng.normal(size=(D, D, N)) * 0.3
delta = np.log1p(np.exp(x @ W_delta + 0.5))          # softplus -> 正步长
B = np.einsum("ld,den->len", x, W_B)                 # (L,D,N) 输入相关的 B
C = np.einsum("ld,den->len", x, W_C)
A = -np.exp(rng.normal(size=(D, N)) * 0.2)           # 负实对角
A_bar = np.exp(delta[:, :, None] * A[None])          # 离散化
B_bar = delta[:, :, None] * B

def ssm_sequential(x, A_bar, B_bar, C):              # O(L) 步串行，Python 循环
    L, D, N = B_bar.shape
    h, ys = np.zeros((D, N)), np.empty((L, D))
    for t in range(L):
        h = A_bar[t] * h + B_bar[t] * x[t][:, None]
        ys[t] = np.sum(C[t] * h, axis=1)
    return ys

def ssm_scan(x, A_bar, B_bar, C):                    # 并行前缀扫描，log L 层
    L, D, N = B_bar.shape
    a = A_bar.reshape(L, D * N).copy()
    b = (B_bar * x[:, :, None]).reshape(L, D * N).copy()
    d = 1
    while d < L:                                     # 结合律: (a1,b1)o(a2,b2)=(a2a1, a2b1+b2)
        a_new, b_new = a.copy(), b.copy()
        a_new[d:] = a[d:] * a[:-d]
        b_new[d:] = b[d:] + a[d:] * b[:-d]           # 注意用旧的 a
        a, b = a_new, b_new
        d *= 2
    return np.sum(C * b.reshape(L, D, N), axis=2)    # 初值 h_{-1}=0，故 b 即 h_t

print("最大绝对误差:", np.abs(ssm_sequential(x, A_bar, B_bar, C)
                            - ssm_scan(x, A_bar, B_bar, C)).max())   # -> 3.55e-14
```

召回实验的核心只有四行（状态 = 外积累加，读出 = 内积）：

```python
kf = rng.normal(size=(n_pairs, m)); kf /= np.linalg.norm(kf, axis=1, keepdims=True)
v  = rng.normal(size=(n_pairs, 16))
S  = kf.T @ v                                        # m × 16 的固定状态，大小与 n_pairs 无关
read = S.T @ kf[j]                                   # 读出第 j 个 value
```

## 💡 小结与三条陷阱

- **陷阱一：把「状态恒定」当成纯优点。** 它同时是容量上限。论文里的「线性时间、常数显存」和「关联召回弱于 attention」是同一件事的两种说法——Zoology、Repeat After Me 已经把它写成定理和经验证据了。选型时问自己一句：**我的任务需要从上下文里精确取回多少条不同的信息？** 少于状态的容量，SSM 划算；远超，就得混 attention。
- **陷阱二：相信「线性时间」就是更快。** 本机朴素 numpy 的实测是：**L=512 时扫描 73.2 ms，比 attention 的 17.0 ms 慢 4 倍**；交叉点在 L≈4096；到 8192 才反过来（2.46x）。短序列上 SSM 的 `log L` 层和逐元素开销是实打实的负担。论文里的 5x 吞吐是 GPU kernel 工程的成果，不能拿复杂度式子推。
- **陷阱三：以为容量问题是纯「架构宿命」。** 有反例。*Mimetic Initialization Helps State Space Models Learn to Recall*（arXiv:2410.11135，2024-10）发现，Mamba 在拷贝/召回上的糟糕表现**有相当一部分是训练难、初始化不对**，用一个刻意模仿 attention 的结构化初始化，就能让 Mamba 明显学会拷贝和关联召回。所以「固定状态弱」的边界，还取决于你**怎么训**——这不是童话，但足以说明：把架构缺陷当成不可逾越的天花板，也会误判。
- **工程上真实发生的事是混血，不是二选一。** Jamba（AI21，arXiv:2403.19887）把 Transformer 层和 Mamba 层交错堆叠，官方报告 256K 上下文；Samba、Zamba 走的是同一路线。**attention 负责精确取回，SSM 负责廉价长程汇总**——把两者的账单各取一半，是目前最耐打的答案。

最后声明本次实验的边界：全都跑在本机 CPU 的 numpy 上，**不是 GPU 实测，也不是训练过的 Mamba**。第一段是数值等价性验证（可信）；第二段是纯算术（可信，但配置换了数字就换）；第三段用无训练的线性记忆代理测**固定维状态的容量量级与趋势**，命中率曲线会随特征分布和判据阈值变，别把我的具体百分比当成某个模型的性能。**复杂度只决定趋势，常数项和实现决定你实际拿到什么。**
