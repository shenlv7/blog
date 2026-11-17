---
title: "同一个 int3 格点：训练后量化 91.6%，训练时量化 99.9%——而 int2 只把崩到 50.0% 的模型捞回 89.3%，且 QAT 的权重误差比 PTQ 更大却更准"
description: "4352 个权重压到 4 个格点上，93.7% 变成精确的 0，测试准确率 50.0%——模型只会猜一个类。round() 的真实导数在 12 次有限差分里 12 次都是 0，能用的只有 STE 的「假装它是 1」。同一套 numpy MLP、同一条双月数据：int3 per-tensor 训练后量化 91.6% → 训练时量化 99.9%；int2 per-channel 82.7% → 99.9%；int2 per-tensor 只能从 50.0% 捞回 89.3%，剩下 10.6 个点补不回来。反直觉的对照：QAT 相对权重误差 0.624 大于 PTQ 的 0.504，准确率却高 10.6 个百分点。另测 STE 的盲区：每 50 步只有 2.01% 的权重真的翻过一个格点（第 50 步 7.44%，1250 步降到 0.97%），其余更新全部堆在死区里。"
pubDate: 2026-11-17
tags: ["AI核心概念", "量化", "QAT", "STE", "直通估计器", "低比特训练"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 85
slug: "ai-concepts-85-qat-ste"
---

> 4352 个权重、4 个格点、一个 scale：int2 per-tensor 之后 **93.7% 的权重变成精确的 0**，测试准确率 **50.0%**——模型只会猜一个类。
> 把同一个量化器放回训练里，其余条件不变（同样的 1500 步、同一份数据）：int3 per-tensor 从 **91.6%**（训练后量化）回到 **99.9%**；int2 per-channel 从 **82.7%** 回到 **99.9%**；int2 per-tensor 只从 50.0% 捞到 **89.3%**，剩 **10.6** 个点补不回来。
> 让这件事成立的是一个数学上站不住的导数：`round()` 在 12 次有限差分里 **12 次都是 0**，真正在用的梯度来自直通估计器（STE）——**假装它的导数是 1**。
> 还有一条反直觉的：QAT 训出来的模型**相对权重误差 0.624**，比 PTQ 的 **0.504** 更大，准确率却高 **10.6 个百分点**。量化伤不伤模型，不看误差有多大，看误差落在哪个方向。

读这篇要会：均匀量化里的 scale 与 per-tensor / per-channel 的区别 / 反向传播的链式法则 / 有限差分求导的用法与误差量级。

## 🎯 4 个格点吃掉 4352 个权重

均匀对称量化只有一句话：把权重除以一个步长 `s`、四舍五入到整数、再乘回去。

```
w_q = s · clip( round(w / s), −q_max − 1, q_max )        # q_max = 2^(b−1) − 1
```

`b` 是位宽，`s = max|w| / q_max` 是步长。per-tensor 指整张矩阵共用一个 `s`，per-channel 指每一列（输出神经元）各算一个。

int2 的 `q_max = 1`，可用的整数只有 `-2, -1, 0, 1`，也就是说**整张矩阵只剩 4 个可取值**。我这张 2-64-64-2 的 MLP 有 `2×64 + 64×64 + 64×2 = 4352` 个权重，最大绝对值由每层那几个离群权重决定，步长随之变得很粗——`s` 在三个层里分别是 3.01 / 1.74 / 1.16。结果：**93.7% 的权重落到 0 上**，剩下 6.3% 平分到 ±1、±2 三个非零格点。测试准确率 50.0%，三个种子一个不差——网络把全部输入判成同一类。

这就是训练后量化（PTQ，post-training quantization）在低位宽下的失败形状：函数是被"投影"到一个粗格点上的，投影之后没人管它还像不像原函数。

## 🔬 round() 的真实导数是 0，STE 只做一件事

要让模型自己适应格点，就得让梯度穿过 `round()`。问题是它穿不过去。

`round()` 是阶梯函数：输入在 `[k−0.5, k+0.5)` 内取值恒定，段内导数处处为 0，段与段的交界处不可导。我做了 12 次有限差分：随机挑一个权重，挪 `1e-6`，看量化后的损失动多少——**12 次全是 0**（中位数 `0.00e+00`，没有一次超过 `1e-12`）。用这个梯度训练，除了刚好压在边界上的那几个权重，其余全部收到 0 信号。数学上量化网络是不可训练的。

直通估计器（STE，straight-through estimator）的做法是**改写反向**：前向老老实实走 `round()`，反向声明 `d round(x)/dx = 1`，再用 clip 的范围做个掩码，范围外梯度截断。

```
前向：  w_q = s · clip(round(w/s), −q_max−1, q_max)
反向：  ∂w_q/∂w  ≔  1[ |w| ≤ (q_max + 0.5)·s ]          ← 唯一被改写的一行
        ∂L/∂w   =  ∂L/∂w_q · 1[ |w| ≤ (q_max + 0.5)·s ]
```

前向和反向对同一个函数给出两套不同的导数，这在数值上明显"错"，但它带来了一个正确的定性行为：**损失下降的方向，也是让量化权重往下降方向挪的方向**。这不是近似的梯度，而是一个方向约定——STE 保证下降方向一致，不保证步长对不对，所以 QAT 通常要把学习率压到全精度训练的十分之一左右。

## 🏗️ 能"训回来"的原因不是格点变细，是误差落在哪

先看四组数（双月数据，3000 训练 / 3000 测试，Adam，1500 步；准确率为 3 个种子的均值，误差与零值比例取 seed 1；全部是权重-only 量化、激活保持 float）：

| 格式 | 训练方式 | 测试准确率 | 相对权重误差 | 量化后恰为 0 的权重 |
|---|---|---|---|---|
| FP32 | — | 0.9991 | 0 | 0% |
| int4 per-tensor | PTQ | 0.9979 | 0.219 | 28.4% |
| int3 per-tensor | PTQ | 0.9158 | 0.504 | 54.9% |
| int2 per-channel | PTQ | 0.8270 | 0.640 | 65.7% |
| int2 per-tensor | PTQ | 0.5000 | 0.902 | 93.7% |
| int3 per-tensor | QAT-STE | **0.9993** | 0.624 | 61.6% |
| int2 per-channel | QAT-STE | **0.9991** | 0.653 | 63.4% |
| int2 per-tensor | QAT-STE | **0.8932** | 0.872 | 94.7% |

int3 那一行是全篇最干净的证据：**同一个 2^3 = 8 个格点**，PTQ 掉 8.4 个点，QAT 一点不掉（0.9993，三个种子分别是 0.9997 / 0.9993 / 0.9990，比 FP32 还高 0.02 个点，属于噪声）。

但请注意 int3 的"相对权重误差"：PTQ **0.504**，QAT **0.624**——QAT 的权重离原值更远，准确率反而高 8.4 个点。这说明**权重误差不是准确率的代理指标**。原因在于误差的方向：PTQ 的误差由 `max|w|` 一个数决定，均匀撒在所有通道上，落点完全不受控；QAT 的误差是训练自己挑的——网络会把该被扰动的地方（冗余方向、后层的小权重）扰动得厉害，把不该动的地方钉死在格点上。误差大，但落在函数不在乎的坐标轴上。

第二层证据是我顺手测的：int2 per-tensor 的两个模型里，量化后**恰为 0 的权重是 93.7% 和 94.7%**——QAT 版本非零权重更少，反而多出 39 个点的准确率。整个网络其实是靠 5% 的权重在支撑，QAT 做的事是把这 5% 挑对、并给它们配上合适的符号，而不是保住更多权重。

同一份数据里还有一个"格点没变细"的证据。QAT 训练时 `s` 是从**潜权重**（latent weight，参与梯度更新但从不真正使用的浮点权重）的最大值算的，而潜权重会膨胀：第一层的 `max|w|` 从 3.01 涨到 4.49（1.49 倍），三个层平均的有效格距从 1.97 涨到 2.48，**粗了 1.26 倍**。格点更粗、准确率更高 39 个点——所以 QAT 不是靠"更细的格点"取胜的，是靠权重挪到了格点有分辨率的地方。

## 💻 让梯度穿过 round() 的那二十行

核心就三处：量化器、前向用量化权重、反向不除 `ds/dw`。

```python
import numpy as np

def qmax(bits):                      # int2→1, int3→3, int4→7
    return 2 ** (bits - 1) - 1

def quantize(w, bits, per_channel=False):
    qm = qmax(bits)
    s = (np.abs(w).max(axis=0, keepdims=True) if per_channel
         else np.full(1, np.abs(w).max())) / qm
    s = np.maximum(s, 1e-12)
    return np.clip(np.round(w / s), -qm - 1, qm) * s, s

def forward(model, X, bits=None, per_channel=True):
    h, cache = X, []
    for i, (W, b) in enumerate(zip(model.W, model.b)):
        Wuse = W if bits is None else quantize(W, bits, per_channel)[0]
        z = h @ Wuse + b
        cache.append((h, z, Wuse))
        h = np.maximum(z, 0) if i < len(model.W) - 1 else z
    return h, cache

def loss_grad(model, X, y, bits=None, per_channel=True):
    logits, cache = forward(model, X, bits, per_channel)
    logits -= logits.max(1, keepdims=True)
    p = np.exp(logits); p /= p.sum(1, keepdims=True)
    n = len(y)
    loss = -np.log(p[np.arange(n), y]).mean()
    dz = p.copy(); dz[np.arange(n), y] -= 1; dz /= n
    gW = [None] * len(model.W); gb = [None] * len(model.b)
    for i in reversed(range(len(model.W))):
        h, z, Wuse = cache[i]
        gW[i] = h.T @ dz          # ← 这里乘的是 h，不是 h·(dWuse/dW)
        gb[i] = dz.sum(0)         #   dWuse/dW 被当成 1，这一行就是 STE
        if i:
            dz = (dz @ Wuse.T) * (cache[i - 1][1] > 0)
    return loss, gW, gb
```

丢进那个 2-64-64-2 的 MLP、双月数据、Adam 1500 步，输出：

```
fp     0.9991 / 0.0002   [0.9993, 0.999, 0.999]
ptq4   0.9979 / 0.0014   [0.9993, 0.9983, 0.996]
ptq3   0.9158 / 0.0165   [0.8933, 0.9323, 0.9217]
ptq2pc 0.8270 / 0.0678   [0.9143, 0.8177, 0.749]
ptq2   0.5000 / 0.0000   [0.5, 0.5, 0.5]
qat3   0.9993 / 0.0003   [0.9997, 0.9993, 0.999]
qat2pc 0.9991 / 0.0006   [0.9993, 0.9997, 0.9983]
qat2   0.8932 / 0.0148   [0.9023, 0.905, 0.8723]

STE 盲区：每 50 步里量化权重真正翻格的比例
  step    50  0.0744
  step   350  0.0319
  step   650  0.0110
  step   950  0.0110
  step  1250  0.0097
  全程平均 0.0201
```

完整的脚本（含有限差分验证、有效格距统计、逐层诊断）：
[demos/ai-concepts-85-qat-ste.py](https://github.com/shenlv7/blog/blob/main/demos/ai-concepts-85-qat-ste.py)

最后那段数字值得单说。STE 给出的梯度**不知道一步更新有没有把权重挪过格点**：如果更新量小于到最近边界距离，这次更新对前向完全无效，但潜权重还是被改了。实测每 50 步只有 **2.01%** 的权重真的换了一个量化等级——训练初期 7.44%（权重还没落到格点上），到 1250 步降到 **0.97%**。也就是说 98% 的更新在当下不改变模型，只是把潜权重往边界方向攒。这就是潜权重膨胀的来源：它本质上是个**累加器**，攒够了才翻一格。这也解释了为什么 QAT 常配权重衰减、也解释了为什么有人把 `s` 本身做成可学习参数（LSQ 的做法）——`s` 不动，累加器只能无限往外撑。

## 💡 判据：什么时候值得付 QAT 的训练成本

三条按性价比排序：

**一、先换格点粒度，再考虑重训。** 上面 int2 per-channel（0.8270）比 int2 per-tensor（0.5000）高 32.7 个点，代价是每列多存一个 float scale——对一个 `64×64` 的矩阵，这是 64 个额外数，相对 4096 个 int2（1KB）多出 256 字节，仍然划算。per-channel 是免费的午餐，QAT 不是。

**二、QAT 的收益随位宽下降而放大。** int4 上 PTQ 只掉 0.12 个点，QAT 的空间本来就没有；int3 掉 8.4 个点，QAT 全补回；int2 per-channel 掉 17.2 个点，QAT 也全补回。所以"要不要 QAT"先看 PTQ 掉了多少：**掉不到 1 个点的别训，掉 5 个点以上的训练比换方案便宜**。

**三、2-bit per-tensor 这类格式训练救不回来。** 4 个格点管一张矩阵，QAT 只到 89.3%——剩下 10.6 个点不是训练不足，是容量不够：非零权重只剩 5.3%，信息已经被格点本身丢掉了。真实的 LLM 上到 2-bit 也一样靠混合精度（关键层留 4-bit）或分组量化硬撑，不是靠多训几步。

这条线的出处：STE 的系统性讨论是 Bengio、Léonard、Courville 2013 年的《Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation》，把它用到二值网络上的是 Courbariaux 等的 BinaryConnect（2015）；训练时量化的工程化版本是 Jacob 等 2018 年 CVPR 的《Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference》，把步长做成可学习参数的是 Esser 等的 LSQ（ICLR 2020）。LLM 一侧，QAT 路线的代表是 Liu 等的 LLM-QAT（2023），PTQ 路线的代表是 GPTQ（Frantar 等，ICLR 2023）与 AWQ（Lin 等，MLSys 2024）——后者不需要反向传播，所以能在几小时内处理 70B 级模型，这也是工业界先上 PTQ 的原因。

**这几个数字的边界**：以上是 4352 参数的 MLP、二维合成数据、只量化权重（激活保持 float32）、没有 BN 需要折叠、`s` 用 min-max 且不可学习。真实 LLM 上，激活量化会引入额外的离群通道问题（per-tensor 激活几乎必然崩，需要 per-token 或 per-channel 激活粒度），低位宽下还需要 mixed-precision 兜底。可迁移的是三条定性结论：round() 的真实导数为 0、STE 的下降方向一致但步长不可信、以及误差方向和误差大小是两件事。
