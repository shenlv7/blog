---
title: "改一个样本，同批另外 7 个的输出变了 53%：BatchNorm 把 batch 焊成一根绳，LayerNorm 精确为 0"
description: "同一个 batch 里扰动第 0 个样本 +1.0，其余 7 个样本的输出最大变化 0.5420（输出自身标准差 1.0185，比值 0.532）；换成 LayerNorm 是 0.0000。batch=1 时 BatchNorm 的 xhat 恒为 0、‖dx‖=0.000e+00、‖dγ‖=0.000e+00，训练 600 步验证准确率 0.209（随机猜 0.25），LayerNorm 同配置 1.000。差别不在公式漂亮不漂亮，而在统计量取自 B 个样本还是 D 个特征。"
pubDate: 2026-11-21
tags: ["AI核心概念", "归一化", "BatchNorm", "LayerNorm", "训练稳定性", "小 batch"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 89
slug: "ai-concepts-89-batchnorm-vs-layernorm"
---

> 一个 8 样本的 batch，把第 0 个样本的每个特征都加 1.0，另外 7 个样本的输出**跟着变了 0.5420**——是它们自身输出标准差的 53.2%。同样的一次扰动，LayerNorm 下另外 7 个样本的输出变化**精确为 0.0000**。
> batch 缩到 1：BatchNorm 对两个完全不同的输入给出**完全相同**的输出（幅度都是 0），反向传回来的梯度 ‖dx‖ = 0.000e+00、‖dγ‖ = 0.000e+00。600 步训练后验证准确率 0.209，比随机猜的 0.25 还低；LayerNorm 在同样配置下是 1.000。
> 这不是实现 bug，是两种归一化对"用多少样本来估 μ 和 σ"的回答不同：一个用 batch 里的 B 个样本，一个用样本自己的 D 个特征。

读这篇要会：矩阵的 batch 维与特征维 / 反向传播的链式法则 / 归一化里的 running mean 与 running var 是什么。

## 🎯 改动一个样本，另外七个为什么跟着动

归一化的动作本身没什么可争的：减均值、除标准差、再乘 γ 加 β。分歧全在**均值和方差从哪一堆数里算**。

- BatchNorm：对每个特征通道，横跨 batch 的 B 个样本算 μ、σ²。第 j 个特征的统计量是 $(\mu_j, \sigma_j^2) = (\frac{1}{B}\sum_{i=1}^{B} x_{ij},\ \frac{1}{B}\sum_{i=1}^{B}(x_{ij}-\mu_j)^2)$。
- LayerNorm：对每个样本，横跨自己的 D 个特征算 μ、σ²。第 i 个样本的统计量是 $(\mu_i, \sigma_i^2) = (\frac{1}{D}\sum_{j=1}^{D} x_{ij},\ \frac{1}{D}\sum_{j=1}^{D}(x_{ij}-\mu_i)^2)$。

一个是"纵向"取数，一个是"横向"取数。横向取数不跨样本，所以样本之间没有共享变量；纵向取数跨样本，于是任何一个样本动了，同一特征通道的 μ_j、σ_j² 就动，同批所有样本的输出都跟着动。这是耦合的定义，不是副作用。

本机实测（B=8、D=16、随机正态输入、γ=1、β=0）：

| 归一化 | 扰动样本 0 后其余 7 个样本输出的最大变化 | 输出自身标准差 | 比值 |
|---|---|---|---|
| BatchNorm | 0.5420 | 1.0185 | 0.532 |
| LayerNorm | 0.0000 | 1.0000 | 0.000 |

53.2% 这个量级不是巧合。BN 的输出里带着一个用 8 个样本估出来的方差：$\hat\sigma^2$ 作为随机变量的相对标准差约 $\sqrt{2/(B-1)}$，B=8 时是 0.535，B=64 时降到 0.178。batch 越小，这个"自己估出来的噪声"越大，同批样本被绑得越紧。

## 🔬 batch=1：xhat 恒等于 0，梯度也恒等于 0

把 B=1 代进去，公式自己会说话：

$$x_{ij}\equiv\mu_j\ \Rightarrow\ \hat x_{ij}=\frac{x_{ij}-\mu_j}{\sqrt{\sigma_j^2+\epsilon}}=0,\qquad y_{ij}=\gamma_j\cdot 0+\beta_j=\beta_j$$

输出和输入无关，只剩 β。实测两个不同的随机输入，BatchNorm 给出的输出差异 0.0000、幅度 0.0000；LayerNorm 同样两个输入，输出差异 1.7124，各自幅度 2.4044——它保留了样本内部的相对结构，只是把它缩放到单位方差。

反向更致命。上游梯度 ‖dy‖ = 6.293 时：

| 归一化（B=1） | ‖dx‖ | ‖dγ‖ |
|---|---|---|
| BatchNorm | 0.000e+00 | 0.000e+00 |
| LayerNorm | 5.724e+00 | 5.821e+00 |

推导一行就够：$\partial L/\partial\gamma_j=\sum_i \hat x_{ij}\,\partial L/\partial y_{ij}$，而 $\hat x_{ij}\equiv 0$，所以 dγ 是零向量；$\partial L/\partial x$ 里剩下的是 $\hat x$ 与自身的内积项和两个均值项，代入 $\hat x\equiv 0$ 后三项相消，也精确为 0（不是"很小"，是浮点意义上 0）。

训练实验把这件事落地：4 类、16 维、800 训练样本、两层 tanh MLP、Adam lr=5e-3、600 步、5 个种子：

| 配置 | bs=1 | bs=8 | bs=64 |
|---|---|---|---|
| 无归一化 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| BatchNorm | **0.209 ± 0.169** | 1.000 ± 0.000 | 1.000 ± 0.000 |
| LayerNorm | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |

BN 那一格的 0.209 不是"学得慢"，是**一步没学**：W1 的梯度恒为 0，模型停在随机初始化上，验证准确率就是随机模型的水平（略低于 0.25 是因为随机 logits 的 argmax 有偏）。

这里必须把两件事分开，混起来就会得出错误的结论：

- **推理时 batch=1 用 BatchNorm 没问题**。推理走的是训练期累积的 running mean / running var，与当前 batch 里有几个样本无关。
- **训练时 batch=1 用 BatchNorm 会坏死**。训练走的是当前 batch 的统计量，B=1 直接让 $\hat x\equiv 0$。
- 由此还有一个不显眼但真实的推论：**把 batch=8 拆成 8 次 bs=1 的梯度累积，对 BatchNorm 不等于 batch=8**。按上面的推导，每一次微批的梯度都是零，累积 8 次仍然是零。梯度累积想复现大 batch 的行为，前提是归一化不跨样本——这条前提在 BN 上不成立。

## 💻 复现：三十行核心代码

完整脚本（含耦合测量、梯度探针、9 组训练）在仓库：

`https://github.com/shenlv7/blog/blob/main/demos/89-batchnorm-vs-layernorm.py`

```python
import numpy as np
EPS = 1e-5

def bn_fwd(x, g, b):                 # x: (B, D)
    mu, var = x.mean(0), x.var(0)
    std = np.sqrt(var + EPS)
    xhat = (x - mu) / std
    return g * xhat + b, (xhat, std)

def bn_bwd(dy, cache, g):
    xhat, std = cache
    m = dy.shape[0]
    dg = (dy * xhat).sum(0)
    db = dy.sum(0)
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(0) - xhat * (dxhat * xhat).mean(0)) / std
    return dx, dg, db

def ln_fwd(x, g, b):
    mu = x.mean(1, keepdims=True)
    var = x.var(1, keepdims=True)
    std = np.sqrt(var + EPS)
    xhat = (x - mu) / std
    return g * xhat + b, (xhat, std)

def ln_bwd(dy, cache, g):
    xhat, std = cache
    dg = (dy * xhat).sum(0)
    db = dy.sum(0)
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(1, keepdims=True)
          - xhat * (dxhat * xhat).mean(1, keepdims=True)) / std
    return dx, dg, db
```

两个反向公式差在**在哪个轴上求均值**：`bn_bwd` 的 `.mean(0)` 跨样本，`ln_bwd` 的 `.mean(1, keepdims=True)` 跨特征。把这两行写反，前向照样跑、损失照样降，只是梯度悄悄错了——BN 的 $\partial L/\partial x$ 少了跨样本的校正项时，训练在 bs=64 上仍然能收敛，所以这个错误通常不会被损失曲线抓到，只会让"改一个样本、别的样本跟着抖"这个性质变得更脏。

真实输出（本机 CPU、numpy，全脚本 23 秒）：

```
== 1. batch 内耦合 ==
BN: 改动样本0 后，其余 7 个样本输出的最大变化 0.5420（输出自身标准差 1.0185，比 0.532）
LN: 改动样本0 后，其余 7 个样本输出的最大变化 0.0000（输出自身标准差 1.0000，比 0.000）
BN: batch=1、两个不同输入的输出差异 0.0000（各自输出幅度 0.0000）
LN: batch=1、两个不同输入的输出差异 1.7124（各自输出幅度 2.4044）

== 2. batch=1 时谁还能传梯度 ==
BN: batch=1、上游梯度 ||dy||=6.293 时，||dx||=0.000e+00，||dγ||=0.000e+00
LN: batch=1、上游梯度 ||dy||=6.293 时，||dx||=5.724e+00，||dγ||=5.821e+00

== 3. 训练：验证准确率（5 种子均值 ± 标准差，随机猜 0.25）==
none bs= 1: 1.000 ± 0.000   bn bs= 1: 0.209 ± 0.169   ln bs= 1: 1.000 ± 0.000
```

## 💡 所以 Transformer 里一个 BatchNorm 都没有

BatchNorm（Ioffe & Szegedy, 2015）为卷积网络设计，那里 batch 里是**同尺寸**的图像、每个通道在整张 feature map 上统计，batch 大、尺寸固定，样本间共享统计量是合理的归纳偏置。LayerNorm（Ba, Kiros & Hinton, 2016）被 Transformer（Vaswani et al., 2017）采用，理由是四条具体的、不是审美上的：

1. **序列长度不一**。文本样本按 token 拼进一个 batch，pad 到同一长度后，BN 的 μ_j 会被 padding 位置污染；要么掩掉、要么按位置分别统计（那 BN 就退化成一堆小统计量）。LN 只看单条序列自己的 D 维隐状态，pad 多长都不影响非 pad 位置。
2. **每卡 batch 太小**。长上下文训练时单卡往往只有个位数样本；由 $\sqrt{2/(B-1)}$ 可见，B=4 时方差估计的相对噪声约 0.816，BN 的输出本身就在抖。
3. **训练/推理统计量不一致**。BN 依赖 running stats，而推理时的输入分布（用户 prompt、领域漂移）和训练集不同时，这份统计量是过期的；LN 没有可过期的状态——同一个 token 前向一次，训练和推理算的是一模一样的东西。
4. **分布式**。多卡各自算 BN 统计量会让每张卡的归一化不同，要一致就得上 SyncBN，多一次 all-reduce。LN 不需要任何跨卡通信。

代价也真实：LN 丢掉的是"跨样本的尺度信息"。BN 的白化让同一通道在不同样本间对齐，LN 只保证单个 token 的隐状态均值 0、方差 1——所以 LN 后面通常紧跟一个可学的 γ/β 把尺度还回去，而深网络的残差流尺度问题（Pre-LN 与 Post-LN 的差别）正是由此而来，那是另一篇的事：`/blog/posts/ai-concepts-82-prenorm-postnorm/`。

判断规则可以收成一句：**统计量必须跨样本共享，用 BN（并要求 batch 足够大、样本同构）；每个样本必须独立处理，用 LN**。前半句里"batch 足够大"不是经验之谈，是 $\sqrt{2/(B-1)}$ 这个数写在公式里；后半句里"独立处理"意味着变长序列、梯度累积、逐 token 推理这三件事都不会被归一化偷偷改账。
