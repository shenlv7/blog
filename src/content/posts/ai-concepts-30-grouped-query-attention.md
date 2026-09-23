---
title: "30 | GQA 分组查询注意力：KV cache 从 64 GiB 砍到 16 GiB，把 K/V 头复制给整组 q 头逐元素误差精确为 0；均值池化转换相对误差 94.1%"
description: "把 MHA / GQA / MQA 放在同一根轴上实测。复制语义：2 个 KV 头按 repeat 广播给 8 个 q 头，与「本来就共享同一份 K/V」的 MHA 逐元素 max|Δ| = 0.000e+00——GQA 不是近似，是同一个函数。显存账：32 层 / d_head=128 / 32 个 q 头 / fp16，MHA 每 token 每层 16384 B，128k 上下文单条 64.00 GiB，GQA-8 16.00 GiB（0.250x），GQA-4 8.00 GiB，MQA 2.00 GiB（0.0312x）；batch=32 时 MHA 要 2048 GiB。按 HBM 3.35 TB/s 算单步解码要读完整份 cache：MHA 20.51 ms/步（理论 48.7 token/s），GQA-8 5.13 ms（195.0 token/s），MQA 0.64 ms（1560 token/s）。但注意力分数 FLOPs 三档完全一样，都是 1.074 GFLOP/token——GQA 省的是显存和带宽，不是矩阵乘。反过来做了转换实验：均值池化 K/V 后输出相对误差 94.1171%、余弦相似度只剩 0.3562、注意力熵从 1.5395 拉到 1.6933 nats（均匀分布是 1.7918），K 的平均范数缩了 2.80 倍；直接取组内第 1 个头更糟，相对误差 130.2510%、余弦 0.1180。最后用 logits 秩量了表达上限：d_head=4、n=32、8 个 q 头共享 1 份 K 时共享侧秩只有 4，独立 K 是 8——共享 K 让整组能同时表达的独立注意力方向被压在 min(d_head, n)。"
pubDate: 2026-09-23
tags: ["AI核心概念", "注意力机制", "GQA", "推理优化", "KV cache", "LLM"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 30
slug: ai-concepts-30-grouped-query-attention
---

# AI核心概念(30)：GQA 分组查询注意力——KV cache 砍到 1/4，逐元素误差精确为 0

> 第 22 篇算 KV cache 的时候留了个尾巴：这份缓存的大小正比于 **KV 头数**，而注意力要算几次正比于 **q 头数**。这两个数在标准 Transformer 里被死死绑在一起。GQA 做的事情就是把它们拆开——q 头还是 32 个，KV 头只留 8 个，然后把这 8 份 K/V 复制给 4 个头一组共用。复制这个动作本身不损失任何精度：实测 max|Δ| = 0.000e+00，逐位相等。

## 🎯 前提：KV cache 是解码期的显存税

自回归生成时，每多一个 token，每一层就要多存一份 K 和一份 V——历史不能重算，否则整段上下文要重跑一遍前向。这份缓存的大小是：

```
KV cache 字节数 = 2 (K和V) × h_kv × d_head × 2 Byte (fp16) × 层数 × token数 × batch
```

标准多头注意力（MHA）里 `h_kv = h_q`。拿一个很常见的配置算账——32 层、`d_head=128`、32 个 q 头、fp16：

| 配置 | h_kv | 字节/token/层 | 128k 上下文单条 | 128k × batch 32 |
|---|---|---|---|---|
| MHA | 32 | 16,384 | 64.00 GiB | 2048.0 GiB |
| GQA-8 | 8 | 4,096 | 16.00 GiB | 512.0 GiB |
| GQA-4 | 4 | 2,048 | 8.00 GiB | 256.0 GiB |
| MQA | 1 | 512 | 2.00 GiB | 64.0 GiB |

64 GiB 装不下任何一张消费级卡。更麻烦的是，解码阶段每生成 **一个** token，都要把整份 cache 读一遍——这是纯显存带宽开销。按 HBM 3.35 TB/s 算：

| 配置 | 单步读取量 | 单步耗时 | 理论吞吐上限 |
|---|---|---|---|
| MHA | 64.00 GiB | 20.51 ms | 48.7 token/s |
| GQA-8 | 16.00 GiB | 5.13 ms | 195.0 token/s |
| MQA | 2.00 GiB | 0.64 ms | 1560.0 token/s |

注意这几行里 **算力完全没变**。同样的实验里，QKᵀ 部分的 FLOPs 三档都是 1.074 GFLOP/token——因为分数矩阵的规模由 q 头数决定，而 q 头数我们一个都没动。GQA 砍掉的只有「搬数据的量」。

## 🔬 三档配置是同一根轴上的三个刻度

三个名字其实只差一个参数——KV 头数 `h_kv`：

- **MHA**（Multi-Head Attention）：`h_kv = h_q`，每个 q 头自己一份 K/V。
- **MQA**（Multi-Query Attention）：`h_kv = 1`，所有 q 头共用同一份 K/V。
- **GQA**（Grouped-Query Attention）：`1 < h_kv < h_q`，把 q 头分成 `h_kv` 组，组内共用一份 K/V。`h_q / h_kv` 叫分组大小 G。

实现上 GQA 就是在算完那 `h_kv` 份 K/V 之后，把每个 KV 头按 `repeat_interleave` 广播 G 份，再照常和 q 头配对。它和 MHA 的关系值得较真一下：如果 MHA 的第 0~3 号头「本来就共用同一份 K/V」，那它算出来的东西和 GQA 的组 0 应该**完全一样**。

实测（h_q=8、h_kv=2、G=4、n=6、d=16，纯 fp64 手写矩阵乘）：

```
max|Δ| = 0.000e+00
```

不是 1e-16，是 0。因为两边执行的是同一串浮点运算，逐位相同。**GQA 不是对 MHA 的近似，它是 MHA 的一个子集**——这个细节很关键：它意味着推理时把一个 GQA 模型当 MHA 跑是等价的，也意味着训练 GQA 时梯度是精确回传到共享的那份 K/V 上的，不存在近似误差累积。

## 🏗️ 省下来的到底是什么

把成本拆成三项看更清楚：

| 项 | 正比于 | GQA 是否降低 |
|---|---|---|
| KV cache 显存 | h_kv × 层数 × token 数 | ✅ 降为 h_kv/h_q |
| 每步解码的显存读取 | 同上 | ✅ 同比例 |
| QKᵀ、PV 的 FLOPs | h_q | ❌ 不变 |
| 参数量（K/V 投影矩阵） | h_kv | ✅ 略降 |

所以 GQA 的收益在大 batch、长上下文、显存带宽吃紧的场景最大——这正是今天推理服务的常态。4k 短上下文时 MHA 只要 2048 MiB、GQA-8 是 512 MiB，差距还在但没那么要命；上下文拉到 128k，差距就是 48 GiB 的绝对值，决定能不能跑。

## 💻 转换实验：均值池化到底丢了多少

如果手上已经有一个 MHA checkpoint（比如 8 个 KV 头），想转成 GQA-2，论文给的路线是把组内的 K/V 头**均值池化**，再用远小于原预训练量的 uptraining 补回质量。池化这一步丢了多少？我把两件事都跑了一遍：直接取组内第 1 个头（丢弃其余），和组内平均。

```
h_q=8 → h_kv=2 (G=4)，K/V 形状 n=6 × d=32

均值池化:      max|Δ|=1.5437e+00   平均相对误差=94.1171%   余弦相似度=0.3562
取组内第1个头:  max|Δ|=1.8642e+00   平均相对误差=130.2510%  余弦相似度=0.1180
注意力分布 KL(原‖池化) = 0.1571 nats
注意力熵: 原 1.5395 → 池化 1.6933 nats   (均匀分布 = 1.7918 nats)
```

相对误差 94% 看着很吓人，但要看它来自哪里：组内平均把 K 的范数缩小了约 √G，这个实验里实测平均范数比是 2.80 倍。分数整体缩水 → softmax 被拉平 → 注意力熵从 1.5395 升到 1.6933（越接近 1.7918 越均匀）。**方向没有全乱（余弦还有 0.356），强度先塌了。** 这就是为什么转换要配 uptraining：模型需要重新学会在更小的分数尺度上做选择。

## 🔬 表达上限：共享一份 K 之后，一组头还剩几个方向

共享 K/V 有没有硬性损失？我把问题简化成「固定共享的 K，让组内 q 头随机，看 logits 矩阵 (h_g × n) 的有效秩」——秩就是这一组头能**同时**表达多少种互相独立的 key 选择模式。共享时所有 logits 都落在同一份 K 的列空间里，所以上限是 `min(h_g, d_head, n)`：

| h_g | n | d_head | KV 头 | 共享 K 秩 | 独立 K 秩 | 共享侧上限 |
|---|---|---|---|---|---|---|
| 8 | 32 | 4 | 1 | **4** | 8 | 4 |
| 8 | 32 | 64 | 1 | 8 | 8 | 8 |
| 16 | 8 | 64 | 4 | 8 | 8 | 8 |
| 16 | 32 | 64 | 4 | 16 | 16 | 16 |
| 16 | 32 | 64 | 16 | 16 | 16 | 16 |

第一行是唯一露出破绽的地方：`d_head=4` 小于头数时，共享侧只有 4 个独立方向，独立 K 有 8 个——**瓶颈是 d_head**。第二行把 d_head 提到 64，共享侧立刻追平。这解释了为什么 h_kv=1 的 MQA 会明显掉质量，而 GQA-8 在 `d_head=128` 的模型上几乎不掉：128 维的列空间足够容纳 4 个 q 头的表达需求。表里后几行共享与独立完全相等，也说明**在常规配置下 GQA 的表达能力根本没有被压缩到紧要处**——真正稀缺的是显存，不是这套 logits 的秩。

## 💡 工程结论

- **h_kv 是解码性能最便宜的一个旋钮**。从 32 降到 8，128k 上下文的缓存从 64 GiB 变 16 GiB，单步解码从 20.51 ms 变 5.13 ms，而算力一分不少——因为省的是 HBM 流量。
- **复制语义是精确的**（max|Δ|=0），所以「GQA 模型 = 分组共享的 MHA」这句话可以放心用；但**转换语义不是**（池化后相对误差 94.1%），换 checkpoint 要配 uptraining。
- **别贪到 MQA**。瓶颈公式是 `min(h_g, d_head, n)`：当 d_head 相对分组头数不够宽时（第 1 行，4 维放 8 个头），共享就真的开始压缩表达了。GQA-8 在 `d_head=128` 的模型上是常见的甜点位。
- 复现实验：下面是核心片段（完整脚本共 5 组实验，含池化转换与秩统计），**纯 Python 标准库**，不需要 numpy，`python3 30-gqa.py` 直接跑。

```python
#!/usr/bin/env python3
"""GQA 分组查询注意力数值实验。纯标准库。"""
import math, random
random.seed(20260923)

def rand_tensor(h, n, d, scale=1.0):
    """h 个头，每个头 n 个 d 维向量"""
    return [[[random.gauss(0, scale) for _ in range(d)] for _ in range(n)] for _ in range(h)]

def softmax(v):
    m = max(v); e = [math.exp(x - m) for x in v]; s = math.fsum(e)
    return [x / s for x in e]

def attn_head(q, K, V, scale):
    """单个 q 头对 (K, V) 做注意力。q:(d,) K:(n,d) V:(n,d) -> (d,)"""
    s = [math.fsum(a * b for a, b in zip(q, k)) * scale for k in K]
    p = softmax(s)
    out = [0.0] * len(V[0])
    for w, v in zip(p, V):
        for i in range(len(v)):
            out[i] += w * v[i]
    return out, p

def rank(rows):
    """高斯消元求行空间有效维数"""
    M = [list(r) for r in rows]; ncol = len(M[0]); r = 0
    for c in range(ncol):
        piv = next((i for i in range(r, len(M)) if abs(M[i][c]) > 1e-12), None)
        if piv is None: continue
        M[r], M[piv] = M[piv], M[r]
        pv = M[r][c]
        for i in range(len(M)):
            if i != r and abs(M[i][c]) > 0:
                f = M[i][c] / pv
                for j in range(c, ncol): M[i][j] -= f * M[r][j]
        r += 1
        if r == len(M): break
    return sum(1 for row in M if math.sqrt(math.fsum(x * x for x in row)) > 1e-9)

# E1 复制语义：GQA 与"本来就共享 K/V"的 MHA 逐元素相等
H_Q, H_KV, N, D = 8, 2, 6, 16
G = H_Q // H_KV; scale = 1.0 / math.sqrt(D)
Q = [[random.gauss(0, 1) for _ in range(D)] for _ in range(H_Q)]
K_kv, V_kv = rand_tensor(H_KV, N, D), rand_tensor(H_KV, N, D)
# 参考(MHA)：8 个头各持一份，组内 4 份内容是从共享张量复制来的
K_mha = [[K_kv[j // G][i] for i in range(N)] for j in range(H_Q)]
V_mha = [[V_kv[j // G][i] for i in range(N)] for j in range(H_Q)]
ref = [attn_head(Q[j], K_mha[j], V_mha[j], scale)[0] for j in range(H_Q)]
# GQA：物理上只存 2 份，8 个 q 头现场按 j // G 取（等价于 repeat_interleave）
gqa = [attn_head(Q[j], K_kv[j // G], V_kv[j // G], scale)[0] for j in range(H_Q)]
print("E1 max|Δ| =", max(max(abs(a - b) for a, b in zip(x, y)) for x, y in zip(gqa, ref)))

# E4 显存账
LAYERS, D_HEAD, BYTES = 32, 128, 2
kv = lambda h_kv, tok, bs=1: 2 * h_kv * D_HEAD * BYTES * LAYERS * tok * bs / 1024**3
for name, h_kv in [("MHA", 32), ("GQA-8", 8), ("GQA-4", 4), ("MQA", 1)]:
    print(f"E4 {name:<6} h_kv={h_kv:>2}  B/token/层={2*h_kv*D_HEAD*BYTES:>6,}  "
          f"128k={kv(h_kv, 131072):>6.2f} GiB  "
          f"单步 {kv(h_kv, 131072)*1024**3/(3.35e12)*1e3:>5.2f} ms")
```

一句话收尾：**GQA 的思路不是「少算」，而是「少存少搬」。** 当瓶颈从算力转到带宽，能省的复杂度就从 FLOPs 挪到了字节数上——而字节数这个账里，KV 头数是最容易拧动的那一颗螺丝。
