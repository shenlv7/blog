---
title: "q/k 的范数漂到 3 倍，注意力就从看 80 个键塌成看 2 个：QK-norm 修的是什么"
description: "logits 的尺度等于 q、k 每行 RMS 的平方，而架构里没有任何一项约束这个量。模拟里 r 从 1.0 涨到 3.0，softmax 熵从 6.31 bit 掉到 0.90 bit，有效参与键数从 79.5 掉到 2.2，回传梯度范数从 0.942 掉到 0.271；r=2.86 是熵跌破 1 bit 的临界点。bf16 的 8 位尾数再补一刀：max|logit|=250 时净是舍入颗粒就让注意力权重最大偏移 19.6%。QK-norm 把尺度从两个投影矩阵的范数换成一个显式增益 γ，实测 logit std 恒等于 γ²，权重怎么漂都不动。ViT-22B 报告不做归一化时 logits 涨到 5 万、权重几乎 one-hot——按这里的比例正好对应 r≈113。"
pubDate: 2026-11-09
tags: ["AI核心概念", "Attention", "QK-norm", "训练稳定性", "数值精度", "bf16"]
difficulty: "advanced"
series: "ai-concepts"
seriesOrder: 77
slug: "ai-concepts-77-qk-norm-attention-entropy"
---

> 同一个机制、同一个 softmax，只把 q、k 的每行 RMS 从 1.0 提到 3.0：熵 6.31 → 0.90 bit，
> 有效键 79.5 → 2.2（共 128 个），回传梯度 0.942 → 0.271。提到 8.0 时熵剩 0.105 bit，梯度 0.039。
> QK-norm 之后，logit 标准差只等于增益的平方，跟投影矩阵的范数再没关系。

**读这篇要会**：attention 的 q/k/v 投影与缩放点积／softmax 的雅可比是 $p_i(\delta_{ij}-p_j)$／RMSNorm 的公式／bfloat16 有 8 位显式尾数。不需要跑过多卡训练。

## 🎯 一个不写在 config 里的数

做推理的人看 attention 只看两件事：KV cache 多大、能不能用 FlashAttention。做训练的人还会看第三件：**logits 的尺度**。它不在任何 config 里，因为它是两个学出来的矩阵范数的乘积。

attention 打分是

$$s_{ij}=\frac{q_i\cdot k_j}{\sqrt{d_h}},\qquad q_i=h_iW_Q,\quad k_j=h_jW_K$$

$h$ 在 pre-LN 里已经被归一化过（每行 RMS ≈ 1），所以 $q$ 的尺度完全由 $W_Q$ 决定。而 $W_Q$、$W_K$ 的范数没有任何一项损失函数或架构约束 —— 训练想让注意力更锐利，最省事的做法就是把这两个矩阵放大，梯度也乐意配合。ViT-22B 的论文（Dehghani et al., *Scaling Vision Transformers to 22 Billion Parameters*, ICML 2023, arXiv:2302.05442）把这个现象写得很直白：不做归一化时，attention logits 在训练早期就涨到 5 万量级，softmax 之后权重几乎 one-hot，训练 loss 和梯度跟着不稳。

我用纯 numpy 把这条链路单独拎出来测了一遍：固定其它一切，只扫 q、k 的每行 RMS（记为 $r$，初始化的自然值是 1），128 个键、64 维头、256 条查询。下表是本机跑出来的（`python3 77-qk-norm-attention-entropy.py`，2 秒内跑完）。

| r | logit std | max\|logit\| | 熵（bit） | 有效键 | 峰值权重 | 回传梯度范数 |
|---|---|---|---|---|---|---|
| 0.5 | 0.3 | 1.0 | 6.955 | 124.1 | 0.014 | 0.990 |
| 1.0 | 1.0 | 3.9 | 6.306 | 79.5 | 0.066 | 0.942 |
| 1.5 | 2.3 | 8.8 | 4.269 | 21.7 | 0.272 | 0.754 |
| 2.0 | 4.0 | 15.7 | 2.350 | 6.3 | 0.524 | 0.524 |
| 3.0 | 9.0 | 35.3 | 0.897 | 2.2 | 0.779 | 0.271 |
| 4.0 | 16.1 | 62.7 | 0.472 | 1.5 | 0.874 | 0.163 |
| 8.0 | 64.3 | 250.7 | 0.105 | 1.1 | 0.972 | 0.039 |
| 16.0 | 257.1 | 1003.0 | 0.019 | 1.0 | 0.995 | 0.007 |

「有效键」是 $2^{H}$，即熵换算回键数：128 个键全部均匀参与时是 128。r=1 时它只有 79.5，因为随机方向的 cos 相似度本身有方差，天花板就到不了满值 —— 这也说明熵的绝对值要跟同 d_head 的基线比，别拿 7 bit 当归一化标准。

真正的断崖在 r 从 1 到 3 这一段。二分法找到临界点：**r ≈ 2.86（logit std ≈ 8.23）时平均熵跌破 1 bit**，此时有效键 2.3/128，梯度范数 0.294。而 r 从 1 漂到 3，只需要 $W_Q$ 的范数整体涨 3 倍 —— 结构里没有任何一项在阻止它这么长。

## 🔬 机制：$r^2$、梯度熄灭、bf16 颗粒

logit 标准差对 $r$ 是**平方关系**，不是线性。$q$、$k$ 的分量近似独立、方差 $r^2$，点积的方差是 $d_h\cdot r^4$，再除以 $d_h$：

$$\operatorname{std}(s)\approx\frac{\sqrt{d_h\,r^4}}{\sqrt{d_h}}=r^2$$

表里 $r=1,2,3,4,8,16$ 对应的标准差是 1.0、4.0、9.0、16.1、64.3、257.1，与 $r^2$ 的误差都在舍入范围内。所以「矩阵范数涨 3 倍」在 logits 上是涨 9 倍：一个温和的 3 倍漂移，直接把 softmax 从「看 80 个键」推到「看 2 个键」。

熵塌了为什么就学不动？softmax 的雅可比是

$$J=\frac{\partial p}{\partial s}=\operatorname{diag}(p)-pp^{\top},\qquad \|J\|_2\le\max_i p_i$$

把 $p$ 推到 one-hot，$\max_i p_i\to 1$ 而 $p$ 的其它分量 → 0，梯度向量的范数反而趋于 0 —— 看着像「很确定」，实际是**这一层收不到有效学习信号**。表里梯度从 0.99 掉到 0.007，就是这个效应的直接读数。此时损失曲面对这一层的导数几乎是零，任何学习率都救不回来。要恢复只能靠别的层的梯度把 $W_Q$ 拽回去，而那个路径通常更长、更弱。

第二刀来自精度。bfloat16 只有 8 位显式尾数，任何一个数在它自己的量级上只有约 $2^{-8}$ 的相对分辨率：$|x|=250$ 时相邻可表示数之间差约 1，$|x|=1000$ 时差约 4。而 softmax 是**指数**函数，logit 差 1 就是权重差 $e\approx2.72$ 倍。同一个 logits 向量存成 bf16 再算 softmax，实测权重最大相对偏差：

| r | max\|logit\| | bf16 ulp | softmax 权重最大偏差 |
|---|---|---|---|
| 1.0 | 3.9 | 0.02 | 0.4% |
| 4.0 | 62.7 | 0.24 | 5.2% |
| 8.0 | 250.7 | 0.98 | 19.6% |
| 16.0 | 1003.0 | 3.92 | 46.5% |

也就是说在 r=8 这个已经塌缩的区间，注意力权重里将近五分之一的数值是舍入噪声说了算。fp16 的尾数多两位（10 位），在这个量级稍好，但上限只有 65504，logits 再涨就直接溢出。**熵塌缩和低精度的坏处是乘起来的，不是相加的**：bf16 训练里范数漂移不是精度问题的一个小扰动，它决定了精度是否还生效。

## 🏗️ QK-norm：把尺度从两个矩阵换成一个标量

做法短得不像个「架构改动」：$q$、$k$ 出投影后各做一次 RMSNorm，再乘一个可学习的逐维增益，然后照常点积。

$$\hat q=\gamma_q\odot\frac{q}{\operatorname{RMS}(q)},\qquad \hat k=\gamma_k\odot\frac{k}{\operatorname{RMS}(k)},\qquad s_{ij}=\frac{\hat q_i\cdot\hat k_j}{\sqrt{d_h}}$$

因为归一化把 RMS 钉在 1，$\hat q$、$\hat k$ 的每个分量只由 $\gamma$ 决定，于是

$$\operatorname{std}(s)\approx\frac{\|\gamma_q\|\,\|\gamma_k\|}{\sqrt{d_h}}\quad(\text{两增益同尺度时}\approx\gamma^2)$$

实测扫增益：$\gamma=0.5,1.0,1.4,2.0,2.8,4.0$ 对应 logit std $0.3,1.0,2.0,4.0,7.9,16.1$，熵 $6.955,6.306,4.745,2.350,1.053,0.472$。这不是「QK-norm 让熵变高」，而是**它让熵只依赖一个参数**。模型仍然可以要锐利的注意力，代价是显式把 $\gamma$ 学大 —— 这个动作在 checkpoint 里看得见、可监控、可以加权重衰减管；而「$W_Q$ 悄悄长了 3 倍」在几万个张量里是查不出来的。

这就是为什么它被写成「稳定训练」的改动而不是「提升效果」的改动。OLMo 2 的技术报告（*2 OLMo 2 Furious*, arXiv:2501.00656）把它和「把 norm 挪到 attention/MLP 输出之后」一起用，报告的梯度 spike score 从 0.108 降到 0.069；报告里对它的描述是「避免 logits 过大导致 loss 发散」。更早的版本是 Henry、Dachapally、Pawar、Chen 的 *Query-Key Normalization for Transformers*（Findings of EMNLP 2020，arXiv:2010.04245）：对 q、k 沿 head 维做 ℓ2 归一化，再用一个可学习标量替代 $1/\sqrt{d_h}$，论文里的理由就是「让 softmax 不容易任意饱和」。当时它只在小资源翻译上验证了约 0.93 BLEU 的提升；把它变成大模型必要件的，是 ViT-22B 那次把 220 亿参数堆起来的尝试。

这里有个容易混的点：**attention logits 和 output logits 是两处独立的尺度问题**。前者进 softmax 决定「看谁」，后者进词表 softmax 决定「输出什么」。OLMo 2 对前者用 QK-norm（乘性、给范数装上限），对后者用 z-loss（加 $10^{-4}\log^2 Z$，把配分函数往 1 拉）。两处都要管，因为失效方式不同：attention 熵塌了是这一层学不动，output logits 涨了是输出分布变尖、校准崩掉，而 loss 曲线可能都没动静。

实现上另有一个顺序问题：QK-norm 放在 RoPE 之前还是之后。旋转是正交变换，不改 ℓ2 范数，所以标量增益两种顺序等价；但逐维增益放在旋转之前会被旋转混掉 —— 旋转把相邻两维做线性组合，等于把「按维缩放」洗成「按旋转后的方向缩放」。要么把增益挪到旋转之后，要么干脆用一个标量。

## 💻 跑一遍：临界点在哪

下面这四十来行就是上面所有数字的来源，依赖只有 numpy。完整脚本在 [demos/77-qk-norm-attention-entropy.py](https://github.com/shenlv7/blog/blob/main/demos/77-qk-norm-attention-entropy.py)。

```python
import numpy as np

D_HEAD, N_KEYS, N_Q = 64, 128, 256

def unit_rows(n, d, g):
    X = g.normal(size=(n, d))
    return X / np.sqrt((X ** 2).mean(axis=1, keepdims=True))

def logits_for(r, gain=None, seed=0):
    g = np.random.default_rng(seed)          # 固定种子，复现正文表格
    q, k = unit_rows(N_Q, D_HEAD, g), unit_rows(N_KEYS, D_HEAD, g)
    if gain is not None:                     # QK-norm：RMS 钉成 1，尺度只留给增益
        q = q / np.sqrt((q ** 2).mean(1, keepdims=True)) * gain
        k = k / np.sqrt((k ** 2).mean(1, keepdims=True)) * gain
        r = 1.0
    return (q * r) @ (k * r).T / np.sqrt(D_HEAD)

def stats(L):
    z = L - L.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    ent = -(p * np.log2(np.maximum(p, 1e-30))).sum(1)
    y = np.zeros_like(p); y[np.arange(len(p)), L.argmax(1)] = 1.0
    return L.std(), ent.mean(), (2 ** ent).mean(), np.linalg.norm(p - y, axis=1).mean()

def to_bf16(x):
    u = np.asarray(x, np.float32).view(np.uint32).copy()
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u & 0xFFFF0000).view(np.float32).astype(np.float64)

for r in (1.0, 2.0, 3.0, 8.0):
    L = logits_for(r)
    sd, ent, eff, g = stats(L)
    zz = to_bf16(L); zz -= zz.max(1, keepdims=True)
    pb = np.exp(zz) / np.exp(zz).sum(1, keepdims=True)
    z = L - L.max(1, keepdims=True)
    p = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    dev = np.abs(p - pb).max() / p.max()
    print(f"r={r:4.1f} std={sd:7.1f} 熵={ent:6.3f} 有效键={eff:5.1f} "
          f"梯度={g:.4f} bf16偏差={dev:6.1%}")
```

输出（本机真实结果，不是 GPU 实测，是同一套 numpy 计算）：

```
r= 1.0 std=    1.0 熵= 6.306 有效键= 79.5 梯度=0.9420 bf16偏差= 0.4%
r= 2.0 std=    4.0 熵= 2.350 有效键=  6.3 梯度=0.5236 bf16偏差= 1.3%
r= 3.0 std=    9.0 熵= 0.897 有效键=  2.2 梯度=0.2709 bf16偏差= 2.7%
r= 8.0 std=   64.3 熵= 0.105 有效键=  1.1 梯度=0.0393 bf16偏差=19.6%
```

想让脚本替你找临界点，把 `for` 循环换成就地二分：`ent > 1.0` 就抬下界。上面报的 r ≈ 2.862 是这么来的，40 次迭代不到一秒。

## 💡 三个可用的判断

**一、QK-norm 是保险，不是提分项。** 它的收益随「范数漂移」出现，短训练、小模型、r 一直待在 1 附近的话，测不出差别 —— 这也是它早期被忽略的原因。别期待加了它 eval loss 变好，期待的是曲线别炸。判断标准应该是「同样的 lr，加与不加，梯度范数曲线长什么样」，而不是「最终 loss 谁低 0.002」。

**二、它会让注意力变钝，而且这个变钝是可调的。** 归一化后峰值权重被 $\gamma$ 封顶，需要精确复制（例如 induction head 那种硬检索）的任务，模型必须先把 $\gamma$ 学大，前期收敛会慢一些。这也是它通常和「把 norm 挪到注意力输出之后」一起出现的原因：后者让残差流本身别再膨胀，前者只管打分尺度，两件事分开管。

**三、已经在用 QK-norm 的模型，$\gamma$ 是免费的健康指标。** 训练日志里把每层的 $\|\gamma_q\|$、$\|\gamma_k\|$ 画出来，就是一张「模型有多想要锐利注意力」的图。持续单调上涨到 $\gamma^2$ 对应熵 < 1 bit，说明它在用注意力记死表 —— 这是过拟合和长尾泛化变差的早期信号，比看 loss 早得多。

对照 ViT-22B 报的 5 万量级再算一次：本模拟里 $\max|s|\approx3.9r^2$，5 万对应 $r\approx113$，$\gamma$ 得学到 10 以上。所以不是「大模型才需要 QK-norm」，而是**范数漂移是深度的函数**：层数越多、训练越久，$W_Q$ 越有机会长胖，而熵塌缩一旦发生不可逆 —— 梯度已经熄了，没有机制把它拉回来。
