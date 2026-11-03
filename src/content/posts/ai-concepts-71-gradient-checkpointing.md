---
title: "激活重算：24 MiB 压到 6 MiB 只多 33% 算力，可 dropout mask 一重抽，梯度就偏了 6.7%"
description: "numpy CPU 实跑（非 GPU 实测）：32 层 4×128×128 残差链，全存激活 24.00 MiB、每层检查点 9.00 MiB、√L 检查点 6.00 MiB；后两者都多算 33.4% FLOPs（12.88→17.18 TFLOP），梯度与标准反向逐位相同（最大绝对误差 0.0）。成本模型按 B·T=8192、D=4096、80 层估：30.0 GiB / 10.12 GiB / 2.38 GiB。重算时不复位 RNG，重抽一次 dropout mask，梯度相对误差 6.7e-2；复位后 0。"
pubDate: 2026-11-03
tags: ["训练显存", "激活重算", "Gradient Checkpointing", "反向传播", "实测"]
difficulty: "advanced"
series: "ai-concepts"
seriesOrder: 71
slug: "ai-concepts-71-gradient-checkpointing"
---

> 32 层的残差链，把每块的中间激活全留着是 **24 MiB**；全扔掉、反向时重算，压到 **6 MiB**，代价只是多算 33% 的前向。
> 这笔交易的价格是定死的：显存按 $1/L$ 或 $1/\sqrt{L}$ 降，算力按 $4/3$ 涨，无论怎么分段都绕不开那 33%。
> 真正会把它变成 bug 的不是数学，是重算时又抽了一次 dropout mask —— 梯度相对误差 6.7e-2，而参考实现是 0。

**读这篇要会**：反向传播里每个中间量为什么会被用到 / matmul 的 FLOPs 怎么数（$2M N K$ 个 FLOP）/ 训练显存 = 权重 + 优化器状态 + 激活 这三块的分工。

## 🎯 卡点：激活随 token 数涨，权重不涨

8B 模型全参数训练，bf16 权重 16 GB；Adam 还要 fp32 master 权重和两个动量，每参数 12 字节，合计 96 GB。这部分和 batch 无关，靠 ZeRO 切分就能摊薄，而且是常数。

激活完全另一回事：它正比于 $B\cdot T\cdot d\cdot L$。一个 transformer 块至少要留残差输入、QKV 投影、attention 输出、MLP 的四倍中间维、激活函数输出，按 5 张 $B\cdot T\cdot d$ 张量估，bf16 下 $8192\times4096\times2\times5$ ≈ 320 MiB / 块，32 层就是 10 GiB。序列从 8K 上到 32K，这一项直接乘 4 —— 先 OOM 的是激活，不是权重。

FlashAttention 把 $S^2$ 的打分矩阵从显存里拿掉了，但它只是不落 HBM，块里其余中间量照存。梯度检查点（gradient checkpointing，也叫激活重算）动的正是这一层：**前向只留块边界，反向需要时把块重算一遍**。它不改变模型、不改变梯度，只改变「拿算力换显存」的汇率。

## 🔬 机制：1/3 的前向，换掉 2/3 的缓存

设单块缓存 $c$、单张边界张量 $s$（本例一个块存 3 张，$c=3s$），全存是 $M=Lc$。两种重算策略：

第一种，每层一个检查点。只留每块的输入共 $L+1$ 张，反向进到第 $i$ 块时从它的输入重算一遍前向：

$$M=(L+1)s+c,\qquad \text{FLOPs}=3F+F=4F$$

前向 $F$、反向约 $2F$，重算补上那一个 $F$，总额从 $3F$ 涨到 $4F$，正好 33.3%。

第二种，每 $k$ 层一个检查点。留 $L/k$ 张边界，反向时重算一段（$L/k$ 块）并把这段的缓存摊开：

$$M(k)=\frac{L}{k}s+\frac{L}{k}c,\qquad k^\*=\sqrt{\frac{Lc}{s}}=\sqrt{3L},\qquad M^\*\approx 2\sqrt{3L}\,s$$

同样的 $4F$，显存从 $O(L)$ 掉到 $O(\sqrt{L})$。**这是这个机制唯一的自由度：$k$ 只决定显存怎么分布，不决定你多花多少算力。** 想再省算力，只能少存点别的东西，或者接受重算的二次代价 —— 有些实现为了压到 $2\sqrt{L}$ 以内，会在一段里反复重放前缀，那时多花的就不是一个 $F$ 而是一串。

## 🏗️ 三种策略的真实账

实测配置：$B=4$、$T=128$、$d=128$、扩展 4 倍、32 层，float32，numpy 手写前向反向，单块缓存 0.75 MiB。峰值按张量形状算术计（不是 profiler），时间取 3 次中最快的一次。

| 策略 | 保留的激活峰值 | 总 FLOPs | 墙钟 | 梯度最大绝对误差 |
|---|---|---|---|---|
| A 全存（标准反向） | 24.00 MiB | 12.88 TFLOP | 462 ms | — |
| B 每层检查点 | 9.00 MiB（2.67×） | 17.18 TFLOP | 611 ms | 0.0 |
| C 每 $\sqrt{L}$ 层检查点 | 6.00 MiB（4.0×） | 17.18 TFLOP | 608 ms | 0.0 |

B 和 C 的 FLOPs 都是 $4/3$，墙钟涨 32%，和 33.4% 的账对得上 —— 说明这份实现里重算确实只是多算了一遍，没有额外搬运。

换成真实尺寸：$B\cdot T=8192$、$d=4096$，单张边界 128 MiB、单块缓存 384 MiB，80 层。全存 30.0 GiB，每层检查点 10.12 GiB，$\sqrt{L}$ 检查点 2.38 GiB。**10 GiB 的卡装不下 30 GiB 的激活，但装得下 2.4 GiB** —— 这就是为什么长上下文训练几乎默认开着它。梯度误差为 0.0 也不是巧合：重算走的是同一串指令、同一份权重，浮点结果逐位可复现。

## 💻 实跑：重算必须复现随机性

上面那套「误差 0.0」有一个前提 —— 块是确定性的。给块里塞一个 dropout，重算时就会再抽一次 mask，而前向用的是第一次抽的那份：

```python
def blk_fwd_drop(x, i, r, rate=0.1):
    h = x @ W1[i]
    a = np.maximum(h, 0.0)
    keep = ((r.random(a.shape) >= rate).astype(np.float32) / (1.0 - rate))
    return x + (a * keep) @ W2[i], (x, h, a, keep)

# 前向：每块进入前记下 RNG 状态
states.append(r.bit_generator.state)
x, _ = blk_fwd_drop(x, i, r)

# 反向：重算前必须复位到该块进入时的状态
r.bit_generator.state = states[i]
_, c = blk_fwd_drop(xs[i], i, r)
```

同一份种子下，重抽 mask 的梯度相对误差 **6.694e-2**，复位 RNG 状态后 **0**。所以 `torch.utils.checkpoint` 默认 `preserve_rng_state=True`，这不是保守，是正确性的一部分 —— 关掉它省下的那点状态拷贝，换来的是梯度和前向对不上，loss 曲线会缓慢地漂。

```python
"""核心循环：每层检查点，反向时重算该块。完整脚本见文末链接。"""
xs, x = [x0], x0
for i in range(L):
    x, _ = fwd_block(x, i)
    xs.append(x)                       # 只留块边界

loss, dx = loss_grad(x)
for i in reversed(range(L)):
    _, cache = fwd_block(xs[i], i)     # 重算，算完立刻丢掉
    dx, _, _ = bwd_block(dx, cache, i)

def fwd_block(x, i):                   # x -> x + relu(xW1)W2
    h = x @ W1[i]
    a = np.maximum(h, 0.0)
    return x + a @ W2[i], (x, h, a)

def bwd_block(dy, cache, i):
    x, h, a = cache
    dW2 = a.reshape(-1, H).T @ dy.reshape(-1, D)
    dh = (dy @ W2[i].T) * (h > 0)
    dW1 = x.reshape(-1, D).T @ dh.reshape(-1, H)
    return dy + dh @ W1[i].T, dW1, dW2   # 残差项别丢
```

最后一行是手写重算最容易漏的地方：重算块的前向时拿到了缓存，反向却忘了把恒等路径的 $dy$ 加回去，梯度会按块的衰减因子连乘 32 次 —— 这份草稿上第一次跑出来的输入梯度是 $10^{-42}$ 量级，loss 正常、梯度看着「很小」，非常像梯度消失。

## 💡 边界：什么时候值得，什么时候白搭

**先看激活占比。** 权重和优化器状态是常数，激活随 $B\cdot T$ 走。小 batch、短序列、单卡放得下的时候，重算纯亏 33% 算力；长上下文、大 batch、要开大模型时才回本。判据很简单：$M_{act}/M_{total}$ 不过 30%，别开。

**别对 attention 块做无脑检查点。** FlashAttention 的反向本来就要重算打分，attention 内部的中间量已经不在显存里了；再给整个 attention 子层套一层检查点，重算的是已经被重算过的东西，收益接近零，只多付一次前向。要检查点就挑缓存最大、重算最便宜的部分 —— 通常是 MLP 的四倍中间维。

**它要求确定性。** dropout、随机深度、随机 padding，都必须在重算时复现；非确定性的 reduce kernel 也会让「误差 0.0」变成 $10^{-7}$ 级的噪声。分布式里还要注意重算发生在哪个 rank 上，别让重算的通信模式和前向不一致。

这篇的盲区写在数字里：这是 numpy CPU 实跑，不是 GPU 实测；峰值是形状算术，不是 profiler，真实框架还有 allocator 碎片、临时 buffer、attention kernel 内部的峰值；FLOPs 也是计数器算的，没有考虑重算在 HBM 带宽上的实际代价 —— 在带宽受限的卡上，重算的 33% 会贵于 33%。能确定的只有两条：梯度等价，和那 33% 是机制本身的价格，不是实现不好。

完整脚本（含 $\sqrt{L}$ 分段与 dropout 对照）：[demos/71-gradient-checkpointing.py](https://github.com/shenlv7/blog/blob/main/demos/71-gradient-checkpointing.py)。
