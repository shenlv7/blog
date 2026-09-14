---
title: "21 | 量化 Quantization：把 16 位压成 4 位，模型凭什么还能用"
description: "训练是往权重里写信息，量化是往同样的权重里塞信息——bits 就是预算。本文用 200 行 numpy 实现 per-tensor / per-channel / 分组三档量化，实测出一个反直觉的现场：per-tensor INT8 会把 89.74% 的普通权重直接压成 0，只为了伺候那 0.8% 的离群通道；再真训练一个 MLP 做端到端对照，INT8 交叉熵几乎零损伤（Δ+0.0005），INT4 看粒度差 18 倍，2bit 直接崩掉 10.75 个百分点。附 GPTQ / AWQ / NF4 / Q4_K_M / FP8 的真实参数账。"
pubDate: 2026-09-14
tags: ["AI核心概念", "量化", "Quantization", "INT4", "推理优化", "离群值"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 21
slug: ai-concepts-21-quantization
---

# AI核心概念(21)：量化 Quantization——把 16 位压成 4 位，模型凭什么还能用

> "训练是把信息写进权重，量化是把信息塞进权重。`bits` 是一张固定额度的预算，误差是你为它花的钱——所以真正的问题从来不是「能不能压」，而是「这笔预算怎么分配才不亏」。"

## 🎯 量化是什么：给权重换一把更粗的尺子

**量化（Quantization）**：把用 FP16/FP32 存的权重，换成位数更少的整数（INT8/INT4），只额外存一个**缩放系数（scale）**。取回真值时再乘回去，这个过程叫**反量化（dequantization）**。

对称量化（symmetric，零点固定为 0）就三行：

```python
s = max(|W|) / (2**(bits-1) - 1)      # 尺子的最小刻度：最大值 / 最大整数
q = clip(round(W / s), -2**(bits-1), 2**(bits-1)-1)   # 存这个整数
W_hat = q * s                          # 用的时候乘回来
```

先分清两个经常被搞混的量：

| | 参数量 | 每个参数的字节 | 显存/带宽 |
|---|---|---|---|
| 剪枝、蒸馏 | **变少** | 不变 | 变少 |
| **量化** | **不变** | 变少（16 → 8 / 4 / 2 bit） | 变少 |

也就是说，量化**不会让模型变小型的**，它只是让同一个模型"更轻"。为什么值得？因为大模型的推理是**内存带宽受限（memory-bound）**的：每生成一个 token，都得把整个权重矩阵从显存里读一遍，算的量反而不大。既然瓶颈在读字节，那把 16 bit 压到 4 bit，理论上读取量直接降到 1/4。

账很好算（按 1024 进制，纯权重，不含 KV cache 和激活）：

| 模型 | FP16 | INT8 | INT4 | 2bit |
|---|---|---|---|---|
| 7B | 14 GB | 7 GB | 3.5 GB | 1.75 GB |
| 13B | 26 GB | 13 GB | 6.5 GB | 3.25 GB |
| 70B | 140 GB | 70 GB | 35 GB | 17.5 GB |

一个 70B 模型从"要两张大卡"变成"一张 A100 80G 装得下还有余量"，这就是量化的全部商业价值。**但压缩有损**——接下来是我本机跑出来的损失现场。

## 🔬 误差从哪来：粒度、离群值、位数

量化误差只有一个来源：**同一个 scale 下所有数字共用一套刻度**。所以误差不取决于你要压多少，取决于**一组里最大和最小的数差多少倍**。三个坑：

1. **粒度（granularity）**：整张矩阵共用一个 scale（per-tensor）→ 一行一个（per-channel）→ 每 32 个元素一个（group-wise）。粒度越细，残差越小，代价是 scale 本身也要占空间。
2. **离群值（outlier）**：大模型权重是重尾分布，总有极少数通道的数值比邻居大几十倍。如果它们和普通人共用一把尺子，**尺子是按大数造的，普通人的数值就会被挤进同一格、甚至挤成 0**。
3. **位数**：刻度数量随位数指数缩小，位数每减一半，理论误差大致翻倍到翻几倍。

## 🧪 本机实测 A：模拟权重上，per-tensor INT8 把 89.74% 的权重压成了 0

我造了一个贴近真实形态的 512×512 权重：每行尺度差几倍（lognormal），再选 4 个通道放大 40 倍当作离群通道。结果是三个数：

| 方案 | 整体误差 | **普通通道误差** | 离群通道误差 |
|---|---|---|---|
| per-tensor INT8 | 16.255% | **56.725%** | 2.098% |
| per-channel INT8 | 3.846% | 13.486% | 0.311% |
| group=32 INT8 | 1.224% | **4.305%** | 0.004% |
| per-tensor INT4 | 43.857% | **100.000%** | 34.828% |
| per-channel INT4 | 28.500% | 98.546% | 5.426% |
| group=32 INT4 | 11.392% | 40.063% | 0.058% |
| group=32 INT3 | 13.932% | 48.992% | 0.152% |
| group=32 INT2 | 21.373% | 75.151% | 0.419% |

再加一行最扎心的诊断：

```text
  per-tensor INT8 的伤全在普通人身上？看被挤掉的部分：
    普通通道被量化到 0 的元素占比：89.74%（per-channel 是 17.65%）

  离群值自己被压准了吗？（幅度最大的 8 个元素）
    per-tensor INT8 大数误差 0.2986%
    group=32 INT8   大数误差 0.0000%
    group=32 INT4   大数误差 0.0000%
```

四条结论，全部来自真实输出：

1. **per-tensor INT8 等于没用。** 尺度被 40 倍的大数占满，普通通道 89.74% 的权重直接归零——参数量一点没少，有效信息已经掉光。这就是为什么**没有任何一个正经 LLM 部署方案用 per-tensor 做权重 INT8**。
2. **粒度换来的收益是十几倍量级。** 同样 8 bit，per-tensor 普通通道误差 56.7%，per-channel 13.5%，group=32 只要 4.3%。**一分钱的 scale 开销，买回 13 倍的精度。**
3. **per-tensor 的"准"是用普通人换的。** 它对离群值极准（0.30%），因为尺子就是照它们做的——但这是它唯一的优点。分组量化是两边都要：大数误差 0.0000%，普通人 4.3%。
4. **4 bit 开始，粒度就是生死线。** 同一个矩阵，group=32 INT4 普通通道 40% 误差，per-tensor INT4 直接 100%（全灭）。2bit 连分组都救不回来（75%）——**真实模型里 2bit 之所以还能说话，靠的是 calibration、混合精度和重要性保护，不是"硬压"**。

## 💻 动手 Demo：200 行 numpy，量化算子 + 端到端精度对照

第二个实验才是关键：**光看权重误差不算数，要看模型的准确率掉不掉。** 所以我真训练了一个两层 MLP（64→128→10，6000 训练 / 2000 测试），再把它的权重按不同粒度量化回推理。完整脚本如下（`python3 03-quantization.py`，本机 5 秒跑完）：

```python
"""03-quantization.py — 权重量化：误差从哪来？精度掉多少？
纯 numpy，CPU 约 5 秒。含：三种量化粒度 + 真实训练的 MLP 端到端测精度。
"""
import numpy as np

# ============================================================
# 1. 量化算子：对称量化（symmetric, zero-point = 0）
# ============================================================


def _quant(w, bits, scale):
    qmax = 2 ** (bits - 1) - 1
    q = np.round(w / scale)
    q = np.clip(q, -qmax - 1, qmax)
    return q * scale


def per_tensor(W, bits=8):
    """整张权重共用一个 scale：最省，但被离群值带偏。"""
    s = np.abs(W).max() / (2 ** (bits - 1) - 1)
    return _quant(W, bits, s or 1e-12)


def per_channel(W, bits=8):
    """每个输出通道（每一行）一个 scale。"""
    s = (np.abs(W).max(axis=1, keepdims=True) / (2 ** (bits - 1) - 1))
    return _quant(W, bits, np.maximum(s, 1e-12))


def group_wise(W, bits=4, g=32):
    """按 g 个输入元素一组，每组一个 scale（Q4_K_M/NF4 这一路）。"""
    rows, cols = W.shape
    pad = (-cols) % g
    Wp = np.pad(W, ((0, 0), (0, pad))) if pad else W
    blocks = Wp.reshape(rows, -1, g)
    s = np.abs(blocks).max(axis=2, keepdims=True) / (2 ** (bits - 1) - 1)
    deq = _quant(blocks, bits, np.maximum(s, 1e-12)).reshape(rows, -1)
    return deq[:, :cols]


def rel_err(W, Wh, cols=None):
    """相对误差 ‖W-Ŵ‖/‖W‖；cols 给定则只在指定列上算（用来把离群通道隔离出来）。"""
    if cols is not None:
        W, Wh = W[:, cols], Wh[:, cols]
    return float(np.linalg.norm(W - Wh) / np.linalg.norm(W))


def rel_err_head(W, Wh, head=8):
    """只看幅度最大的 head 个元素（离群通道就在这里）：量化对「大数」准不准。"""
    idx = np.argsort(-np.abs(W).ravel())[:head]
    return float(np.linalg.norm(W.ravel()[idx] - Wh.ravel()[idx]) / np.linalg.norm(W.ravel()[idx]))


# ============================================================
# 2. 实验 A：LLM 权重长什么样 —— 离群值如何毁掉 per-tensor
# ============================================================


def make_llm_like_weight(seed=0, shape=(512, 512), outlier_cols=4, outlier_gain=40.0):
    """模拟大模型权重：行尺度不一 + 极少数通道是离群值（outlier channel）。"""
    rng = np.random.default_rng(seed)
    W = rng.normal(scale=1.0, size=shape)
    W *= rng.lognormal(mean=0.0, sigma=0.8, size=(shape[0], 1))   # 每行尺度差几倍
    cols = rng.choice(shape[1], outlier_cols, replace=False) if outlier_cols else np.array([], dtype=int)
    if outlier_cols:
        W[:, cols] *= outlier_gain
    return W, cols


def experiment_a():
    W, outliers = make_llm_like_weight()
    normal = np.setdiff1d(np.arange(W.shape[1]), outliers)
    print("=== 实验 A：模拟权重矩阵 512x512（4 个通道被放大 40 倍 = 离群通道）===")
    print(f"  {'方案':<26}{'整体误差':>12}{'普通通道误差':>14}{'离群通道误差':>14}")
    for name, Wh in [
        ("per-tensor INT8 (8bit)", per_tensor(W, 8)),
        ("per-channel INT8 (8bit)", per_channel(W, 8)),
        ("group=32 INT8 (8bit)", group_wise(W, 8, 32)),
        ("per-tensor INT4 (4bit)", per_tensor(W, 4)),
        ("per-channel INT4 (4bit)", per_channel(W, 4)),
        ("group=32 INT4 (4bit)", group_wise(W, 4, 32)),
        ("group=32 INT3 (3bit)", group_wise(W, 3, 32)),
        ("group=32 INT2 (2bit)", group_wise(W, 2, 32)),
    ]:
        print(f"  {name:<26}{rel_err(W, Wh):>11.3%}{rel_err(W, Wh, normal):>14.3%}"
              f"{rel_err(W, Wh, outliers):>14.3%}")

    # 离群值到底多伤 per-tensor：把离群通道的“代价”单独算
    print("\n  per-tensor INT8 的伤全在普通人身上？看被挤掉的部分：")
    Wh = per_tensor(W, 8)
    print(f"    普通通道被量化到 0 的元素占比："
          f"{(np.abs(Wh[:, normal]) == 0).mean():.2%}"
          f"（per-channel 是 {(np.abs(per_channel(W,8)[:, normal]) == 0).mean():.2%}）")

    print("\n  离群值自己被压准了吗？（幅度最大的 8 个元素）")
    for name, Wh in [("per-tensor INT8", per_tensor(W, 8)),
                     ("group=32 INT8", group_wise(W, 8, 32)),
                     ("group=32 INT4", group_wise(W, 4, 32))]:
        print(f"    {name:<16}大数误差 {rel_err_head(W, Wh):.4%}")

    print("\n  位数 vs 误差（group=32 分组量化，普通通道）")
    for bits in (2, 3, 4, 8):
        Wh = group_wise(W, bits, 32)
        print(f"    {bits}bit  整体 {rel_err(W, Wh):.3%} | 普通通道 {rel_err(W, Wh, normal):.3%}")


# ============================================================
# 3. 实验 B：端到端 —— 训练一个 MLP，再把它量化
# ============================================================


def make_task(n=8000, dim=64, hid=8, classes=10, seed=3):
    """非线性分类任务：z~N(0,I) 经随机映射+tanh 变非线性，标签是另一个随机映射的 argmax。"""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, hid))
    A = rng.normal(size=(hid, dim)) / np.sqrt(hid)
    B = rng.normal(size=(dim, dim)) / np.sqrt(dim)
    C = rng.normal(size=(hid, classes))
    X = np.tanh(z @ A) @ B + rng.normal(scale=0.3, size=(n, dim))
    y = np.argmax(z @ C, axis=1)
    X = X.astype(np.float64)
    return X, y


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    """两层 MLP：64 -> 128 (ReLU) -> 10。手写反向传播 + Adam。"""

    def __init__(self, din, hid=128, dout=10, seed=0):
        g = np.random.default_rng(seed)
        self.W1 = g.normal(scale=np.sqrt(2 / din), size=(din, hid))
        self.b1 = np.zeros(hid)
        self.W2 = g.normal(scale=np.sqrt(2 / hid), size=(hid, dout))
        self.b2 = np.zeros(dout)
        self.st = {k: [np.zeros_like(self.params()[k]), np.zeros_like(self.params()[k])] for k in self.params()}
        self.t = 0

    def params(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, X):
        h = np.maximum(X @ self.W1 + self.b1, 0)
        return softmax(h @ self.W2 + self.b2), h

    def loss_and_grads(self, X, y):
        p, h = self.forward(X)
        n = len(X)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        dp = p.copy()
        dp[np.arange(n), y] -= 1
        dp /= n
        gW2, gb2 = h.T @ dp, dp.sum(axis=0)
        dh = (dp @ self.W2.T) * (h > 0)
        gW1, gb1 = X.T @ dh, dh.sum(axis=0)
        return float(loss), {"W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    def step(self, X, y, lr=3e-3):
        _, g = self.loss_and_grads(X, y)
        self.t += 1
        for k, p in self.params().items():
            m, v = self.st[k]
            m *= 0.9; m += 0.1 * g[k]
            v *= 0.999; v += 0.001 * g[k] ** 2
            p -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)


def accuracy(model, X, y):
    return float((model.forward(X)[0].argmax(axis=1) == y).mean())


def ce_loss(model, X, y):
    """交叉熵：比准确率更平滑的精度指标，量化损伤会先在这里显形。"""
    p = model.forward(X)[0]
    return float(-np.log(p[np.arange(len(X)), y] + 1e-12).mean())


def size_mb(arrays_bytes):
    return arrays_bytes / 1024 / 1024


def experiment_b():
    X, y = make_task()
    ntr = 6000
    Xtr, ytr, Xte, yte = X[:ntr], y[:ntr], X[ntr:], y[ntr:]
    m = MLP(din=X.shape[1], seed=1)
    rng = np.random.default_rng(0)
    for i in range(1200):
        idx = rng.integers(0, ntr, 256)
        m.step(Xtr[idx], ytr[idx], lr=2e-3)
    acc0, ce0 = accuracy(m, Xte, yte), ce_loss(m, Xte, yte)
    print("\n=== 实验 B：真训练一个 MLP（64→128→10，6000 训练 / 2000 测试）===")
    print(f"  FP32 基线：测试准确率 {acc0:.2%}，交叉熵 {ce0:.4f}")

    nparam = sum(v.size for v in m.params().values())
    print(f"  参数量 {nparam:,}（W1 {m.W1.size} + W2 {m.W2.size} + bias）"
          f" | FP32 权重 {size_mb(nparam*4):.3f} MB")

    def eval_quant(wq_fn, label, bytes_per_w, extra_per_group=0.0, layers=("W1", "W2")):
        m2 = MLP(din=X.shape[1], seed=1)
        m2.W1 = wq_fn(m.W1) if "W1" in layers else m.W1.copy()
        m2.W2 = wq_fn(m.W2) if "W2" in layers else m.W2.copy()
        m2.b1, m2.b2 = m.b1.copy(), m.b2.copy()
        a, c = accuracy(m2, Xte, yte), ce_loss(m2, Xte, yte)
        nw = sum(getattr(m, L).size for L in layers)
        mb = size_mb(nw * bytes_per_w + (nw / 32) * extra_per_group)
        print(f"  {label:<30}准确率 {a:>6.2%} (Δ {a-acc0:+.2%})  交叉熵 {c:.4f} "
              f"(Δ {c-ce0:+.4f})  量化部分体积 {mb:.3f} MB")
        return a

    print("\n  同一模型，只把权重矩阵量化（bias 保持 FP32，和 llama.cpp 的做法一致）：")
    eval_quant(lambda W: per_tensor(W, 8), "INT8 per-tensor", 1)
    eval_quant(lambda W: per_channel(W, 8), "INT8 per-channel", 1)
    eval_quant(lambda W: group_wise(W, 8, 32), "INT8 group=32", 1, 2 * 4 / 32)
    eval_quant(lambda W: group_wise(W, 4, 32), "INT4 group=32", 0.5, 2 * 4 / 32)
    eval_quant(lambda W: per_channel(W, 4), "INT4 per-channel", 0.5)
    eval_quant(lambda W: per_tensor(W, 4), "INT4 per-tensor", 0.5)
    eval_quant(lambda W: group_wise(W, 2, 32), "INT2 group=32", 0.25, 2 * 4 / 32)

    print("\n  哪一层更怕量化？（INT4 group=32）")
    eval_quant(lambda W: group_wise(W, 4, 32), "只量化 W1（第一层）", 0.5, 2 * 4 / 32, ("W1",))
    eval_quant(lambda W: group_wise(W, 4, 32), "只量化 W2（输出层）", 0.5, 2 * 4 / 32, ("W2",))

    # 量化误差和模型误差的关系：同样误差，哪层代价更大
    print("\n  两层权重本身的量化误差（相对误差）")
    for bits, g in ((8, 32), (4, 32), (2, 32)):
        print(f"    {bits}bit group=32: W1 {rel_err(m.W1, group_wise(m.W1, bits, g)):.3%}"
              f" | W2 {rel_err(m.W2, group_wise(m.W2, bits, g)):.3%}"
              f"（W1 是 64→128，W2 是 128→10）")


if __name__ == "__main__":
    experiment_a()
    experiment_b()
```

实测输出（本机原样贴）：

```text
=== 实验 B：真训练一个 MLP（64→128→10，6000 训练 / 2000 测试）===
  FP32 基线：测试准确率 80.05%，交叉熵 0.5885
  参数量 9,610（W1 8192 + W2 1280 + bias） | FP32 权重 0.037 MB

  同一模型，只把权重矩阵量化（bias 保持 FP32，和 llama.cpp 的做法一致）：
  INT8 per-tensor               准确率 80.15% (Δ +0.10%)  交叉熵 0.5890 (Δ +0.0005)  量化部分体积 0.009 MB
  INT8 per-channel              准确率 80.10% (Δ +0.05%)  交叉熵 0.5885 (Δ +0.0000)  量化部分体积 0.009 MB
  INT8 group=32                 准确率 80.10% (Δ +0.05%)  交叉熵 0.5888 (Δ +0.0003)  量化部分体积 0.009 MB
  INT4 group=32                 准确率 80.50% (Δ +0.45%)  交叉熵 0.5914 (Δ +0.0029)  量化部分体积 0.005 MB
  INT4 per-channel              准确率 79.65% (Δ -0.40%)  交叉熵 0.5974 (Δ +0.0089)  量化部分体积 0.005 MB
  INT4 per-tensor               准确率 79.15% (Δ -0.90%)  交叉熵 0.6408 (Δ +0.0523)  量化部分体积 0.005 MB
  INT2 group=32                 准确率 69.30% (Δ -10.75%)  交叉熵 1.0506 (Δ +0.4621)  量化部分体积 0.002 MB

  哪一层更怕量化？（INT4 group=32）
  只量化 W1（第一层）                   准确率 80.15% (Δ +0.10%)  交叉熵 0.5886 (Δ +0.0001)  量化部分体积 0.004 MB
  只量化 W2（输出层）                   准确率 80.10% (Δ +0.05%)  交叉熵 0.5934 (Δ +0.0049)  量化部分体积 0.001 MB

  两层权重本身的量化误差（相对误差）
    8bit group=32: W1 0.544% | W2 0.432%（W1 是 64→128，W2 是 128→10）
    4bit group=32: W1 9.889% | W2 7.546%（W1 是 64→128，W2 是 128→10）
    2bit group=32: W1 66.798% | W2 52.976%（W1 是 64→128，W2 是 128→10）
```

四条读数：

1. **INT8 在这个模型上等于无损。** 三种粒度交叉熵都在 +0.0005 以内，`per-channel` 的差异甚至打印成 0.0000。这也解释了一个行业默契：**8 bit 是几乎免费的午餐，真正需要研究的是 4 bit 及以下。**
2. **4 bit 开始，粒度价差 18 倍。** 同样 INT4：group=32 交叉熵 +0.0029，per-tensor +0.0523（准确率 -0.90%）。前面实验 A 里"4bit 普通通道 40% vs 100%"的误差差，到这里变成了真实的模型损伤——**这就是 llama.cpp 里 Q4_0 和 Q4_K_M 差的那 0.3 GB 到底买到了什么。**
3. **2 bit 会崩，而且崩得不对称。** 权重相对误差 W1 66.8%、W2 53.0%，看着"还有一半准"；但模型交叉熵从 0.589 涨到 1.051（近乎翻倍），准确率掉 10.75 个百分点。**误差不是线性传导的**——过了一个阈值，后面的层再也纠正不回来。
4. **敏感度 ≠ 误差大小。** W2 自己的量化误差更小（7.5% vs W1 的 9.9%），但只量化它造成的交叉熵损伤是 W1 的 49 倍（+0.0049 vs +0.0001）。输出层直接对着 logits，误差没有缓冲。所以工程上常见的做法是：**第一层和最后一层、以及 attention 的 V/O 投影，给的比特数比别处多。**

## 🏗️ 工程上真的怎么做

把上面的结论搬到大模型，就有了整条技术路线：

| 方法 | 做法 | 关键取舍 |
|---|---|---|
| **RTN + per-channel** | 直接四舍五入，一行一个 scale | 8bit 够用，4bit 就浪费精度；最省事 |
| **分组量化（Q4_K/Q4_K_M/NF4）** | 每 32~64 个权重一组一个 scale，block size 常取 64 | 最好的"每 bit 精度"；代价是 scale 本身的开销——**llama.cpp 里 Llama-2-7B 的 Q4_K_M 实际是 4.84 bit/参数，不是 4.0** |
| **GPTQ**（arXiv:2210.17323） | 用一小撮校准数据算 Hessian，逐列量化，把这一列的误差补偿到还没量化的权重上 | 免训练，3~4 bit 常用；要校准集，量化过程本身要跑几分钟到几十分钟 |
| **AWQ**（arXiv:2306.00978） | 用**激活值**的幅度判断哪些权重重要，只重点保护约 1% 的显著权重（按通道缩放） | 官方结论：保护 1% 就能大幅降误差；比 GPTQ 更容易工程化 |
| **NF4 + 双重量化**（QLoRA，arXiv:2305.14314） | 4bit 分位数量化（对正态分布信息论最优），再把 scale 本身也量化一遍 | 官方账：65B 微调显存从 >780 GB 降到 <48 GB；双重量化再省 0.37 bit/参数 |
| **FP8 / 低精度训练** | 权重、激活、甚至梯度都用 FP8 算（H100 起有原生支持），DeepSeek-V3 就是 FP8 混合精度训练的 | 省的是**算力**不只是显存，但需要硬件支持 |
| **QAT（量化感知训练）** | 训练时就插入伪量化，让模型自己学会适应误差 | 效果最好、成本最高，边缘设备/极致小模型才用 |

几个真实数字（都是官方公开数据）：

- llama.cpp 的 Llama-2-7B 文件大小：FP16 13 GB → **Q8_0 7.16 GB** → Q4_K_M 4.08 GB → q2_K 2.87 GB。注意 Q8_0 的困惑度（5.9070）和 FP16（5.9066）几乎一模一样，而 Q4_0 涨到 6.1565。
- 文献里比较一致的边界：**3 bit 以下开始出现肉眼可见的精度下降**（Dettmers & Zettlemoyer 2023 的结论），4 bit 是当前"能用于生产"的下限。
- 一个容易被忽略的前提：以上全是**仅权重量化（weight-only）**——激活值仍然按 FP16 算，只是每次要先把 4bit 反量化回 FP16 再做矩阵乘。所以 **省显存和带宽是确定的，纯算力收益要看实现**；想要算力也省，得上激活量化或 FP8。

最后一句工程建议：**别只看困惑度选量化方案。** 我的实验里 2bit 的权重"看起来只损失了一半精度"，模型的准确率却掉了 10 个点——**量化损伤是在你的任务上测出来的，不是在别人的 benchmark 上。**

## 💡 小结

- 量化不动参数量，只动**每个参数占几个字节**；省的是显存和内存带宽，而大模型推理恰好卡在带宽上，所以它是当下性价比最高的优化手段
- 三个变量决定精度损失：**粒度（per-tensor → per-channel → group）、离群值、位数**
- 本机实测最扎心的一条：模拟权重含 40× 离群通道时，**per-tensor INT8 把普通通道 89.74% 的权重压成 0**；换成 group=32，同一份 8bit 误差从 56.7% 降到 4.3%，大数误差还从 0.30% 降到 0.0000%
- 端到端对照：INT8 三种粒度几乎无损（Δ交叉熵 ≤0.0005）；**INT4 粒度差 18 倍**（group=32 +0.0029 vs per-tensor +0.0523）；**2bit 直接崩**，准确率 -10.75%、交叉熵翻倍
- 敏感度不看误差大小：输出层（W2）自己的误差更小，损伤却是第一层的 49 倍——所以真实方案给首层、末层和 V/O 投影留更多比特
- 心法：**量化的本质是预算分配题**——bits 有限，把刻度花在哪儿（粒度）比花多少（位数）更值钱 ⚖️

下次看到「4bit 无损」的宣传，先问两句：**粒度是 per-tensor 还是分组？量化前跑过哪些校准数据？** 4.84 bit 的 Q4_K_M 和 4.0 bit 的 Q4_0 之间那 0.84 bit，就是这些问题的答案。 🧮

---

_概念卡片持续更新中，下一个概念见~_ 🔫
