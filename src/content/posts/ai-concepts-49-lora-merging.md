---
title: "49 | LoRA 合并：低秩的 delta 没有让合并变划算，只让它变便宜"
description: "承接 48 篇（全参数密集微调的合并）在 LoRA 上重测同一套实验。本机纯 CPU numpy，2112 维 delta、rank r=8、单 seed、14.2 秒，反向传播中心差分自检 1.732e-07。LoRA 专家的自身任务准确率 0.9580/0.9587，交叉只剩 0.6697/0.6740（灾难性遗忘照样发生）。两任务 delta 的余弦从全参数微调的 −0.5638 变成 −0.4282——低秩确实降低了冲突，但没把它变成正数，合并依然全部不高于未微调的 θ0（0.8348）：linear avg 0.8323、task arith λ=1 0.8303、λ=1.5 0.8272。三个关键量化：(1) 低秩 ≠ 稀疏——元素级近零坐标占比只有 0.4129，而全参数 delta 是 0.6851，TIES/DARE 的前提在 LoRA 上更不成立（实测 TIES −2.26pp、DARE-TIES −4.28pp，但仍低于朴素求和）。(2) 秩拼接（PEFT 的 cat）是恒等式而非魔法：max|Δ|=5.551e-17，它和 task arith 的权重求和数学上是同一个东西，唯一区别是秩 2r 可以再截回 r，实测截断保留 94.97% 能量、准确率一位不差。(3) element-wise 组合（LoraHub / PEFT linear 的做法，(ΣwA)(ΣwB)）与真·权重相加的差是 0.5019，占目标模长的 51.55%——交叉项 U_A V_B 不是零。全部为 toy 模型结果，不是 GPU/LLM 实测。"
pubDate: 2026-10-12
tags: ["AI核心概念", "LoRA", "模型合并", "PEFT", "秩拼接", "TIES", "DARE", "低秩适配", "微调"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 49
slug: ai-concepts-49-lora-merging
---

# AI核心概念(49)：LoRA 合并——低秩的 delta 没有让合并变划算，只让它变便宜

> 上一篇（48 Model Merging）的结尾留了一句话：TIES / DARE 报告的收益出现在**同源检查点 + 稀疏/低秩的 delta** 场景，我的全参数密集微调没有这种稀疏性，所以要在 LoRA 上重测一遍。这篇就是那次重测。
>
> 期待是很具体的：LoRA 的 delta 天生是低秩的（`ΔW = U V`，秩 ≤ r），参数少、结构规整，听起来正是「合并手艺的主场」——如果合并真有个甜蜜区，应该在这里。
>
> 实测结果是**一半兑现、一半落空**。兑现的一半：低秩确实降低了两个任务更新之间的冲突，余弦从全参数微调的 **−0.5638** 变成 **−0.4282**。落空的一半：它没把冲突变成正数，所以合并的结局一点没变——**在 LoRA delta 上做合并，依然没有一个方法能超过「什么都不做」的预训练模型 θ0**（合并最好 0.8325，θ0 0.8348）。
>
> 顺带量到了三件在 LoRA 生态里被反复引用、但很少被摆到实验台上说清的事：**秩拼接不是魔法**（它恒等于权重相加，`max|Δ|=5.551e-17`）；**低秩不等于稀疏**（元素级近零占比 0.4129，比全参数 delta 的 0.6851 还低，DARE 的 drop 前提更站不住）；以及 **element-wise 组合会引入占模长 51.55% 的交叉项**——而它的准确率居然几乎不变，这本身就是个该被记住的教训。

## 🎯 术语与出处对齐

- **LoRA**：Hu 等，*LoRA: Low-Rank Adaptation of Large Language Models*（ICLR 2022，arXiv:2106.09685）。冻结基座 `W0`，只训一对低秩矩阵 `ΔW = B A`（`B∈R^{d×r}`、`A∈R^{r×d}`），且 `A` 初始化为 0 使起点严格等于 `W0`。本文与 PEFT 的记法一致（`h = W0 x + B A x`），代码里我写成 `ΔW = U V`，只是换了字母。
- **秩拼接（`cat`）**：HuggingFace PEFT 官方博客 *PEFT welcomes new merging methods* 给出的定义：`A_merged = concat(w₁s₁A₁, w₂s₂A₂, dim=0)`、`B_merged = concat(B₁, B₂, dim=1)`，于是 `B_merged A_merged = w₁s₁B₁A₁ + w₂s₂B₂A₂`——**精确的加权合并**，且允许参与合并的适配器秩不同。同一篇博客的观察 1 是：「多数场景 `cat` 效果很好，直接从它开始；但合并很多适配器时产物会因拼接而过大导致 OOM」。这句话就是本文实验四的动机。
- **SVG/`svd` 与秩截断**：先算 `Σ w_i ΔW_i`，再对结果做 SVD 截断回指定秩，「默认取参与者中最大的秩」——即把 2r 压回 r 的那一步。
- **element-wise 组合（`linear`）**：LoraHub（Huang 等，arXiv:2307.13269，COLM 2024）的 Compose 阶段定义是 `m̂ = (w₁A₁+w₂A₂+…)·(w₁B₁+w₂B₂+…)`——**先各自加权求和 A、B，再相乘**，并要求各适配器秩相同。注意这个式子里藏着交叉项 `A₁B₂ + A₂B₁`，本文实验五专门量它。
- **TIES**：Yadav 等，arXiv:2306.01708（trim → elect sign → disjoint mean）；**DARE**：Yu 等，arXiv:2311.03099（随机 drop 比例 p 的 delta 再按 `1/(1−p)` 放大）。两者都隐含「大部分坐标上的更新是小噪声」这一稀疏前提——本文实验三直接测这个前提。
- **一个来自工程现场的旁证**：PEFT 博客的观察 4 原话是 *dare_ties wasn't giving good results*。我这次在 toy 上的实测与它一致（DARE-TIES 是六个方法里最差的一档），所以下面那格数字不是我调参调坏了。

## 🔬 实验设置：把 48 篇的模型原样换成 LoRA

沿用 48 篇的任务族：两个任务共享方向 `shared`、各带一个正交的 `spec_t`，`w_t = (shared + β·spec_t)/‖·‖`，β=1.0。区别只有一处——**微调方式**：

| | 48 篇（全参数密集微调） | 49 篇（LoRA） |
|---|---|---|
| 可训练参数 | 2112（W1 2048 + W2 64） | 1288（U1 256 + V1 512 + U2 512 + V2 8） |
| delta 的秩 | 满秩（W1 块秩 32） | ≤ 8 |
| 训练步数 / 学习率 | 200 / 0.06 | 400 / 0.08 |
| 反向传播自检 | 3.175e-07 | **1.732e-07**（中心差分） |
| 单次运行耗时 | ~10 秒 | 14.2 秒 |

θ0 完全相同：两任务混合数据上全参数预训练 400 步，之后**永久冻结**，作为 LoRA 的基座。评测集各 3000 样本。**这是 toy 模型，不是 GPU/LLM 实测**——2 层 MLP、`32→64→1`、单 seed。

先看 LoRA 自己有没有学到东西：

| 模型 | acc_A | acc_B |
|---|---|---|
| θ0 预训练（未微调） | 0.8363 | 0.8333 |
| LoRA 适配器 A（只训 A） | **0.9580** | 0.6697 |
| LoRA 适配器 B（只训 B） | 0.6740 | **0.9587** |
| 全参数微调 A（48 篇对照） | 0.9777 | 0.6900 |
| 全参数微调 B（48 篇对照） | 0.6863 | 0.9757 |

两件事同时成立：**LoRA 是能专精的**（自身任务 +12.2pp），**灾难性遗忘也照样发生**（跑到对方任务上 0.6697，比 θ0 低 16.7pp）。但 LoRA 的专精幅度明显弱于全参数微调（0.9580 vs 0.9777，差 2.0pp）——这一点后面有用，因为它意味着「合并相对专家」的落差天然变小了。**比较合并方法时，分母是 θ0 还是专家，结论会很不相同。**

## 🔬 实验一：低秩确实降低了冲突——但没降到零

| 诊断量 | LoRA (r=8) | 全参数微调 |
|---|---|---|
| cos(τA, τB) | **−0.4282** | −0.5638 |
| 元素级近零占比（\|τ\|<5%×max） | **0.4129** | 0.6851 |
| 最大奇异值占奇异值总和 | 0.8982 | 0.5182 |
| 达到 99.9% 奇异值能量所需分量数 | 8（= 秩上限） | 32 |
| ‖τ‖ | 1.8205 | 1.2655 |
| W1 块前 5 个奇异值 | 1.6358 / 0.0518 / 0.0404 / 0.0297 / 0.0238 | 1.0173 / 0.1486 / 0.0976 / 0.0743 / 0.0546 |

两处值得停下来看：

1. **冲突确实变小了**：余弦从 −0.56 抬到 −0.43，实测的「一致度」提高了约 0.14。这句是 48 篇留下的假设里唯一被证实的部分：低秩约束把两个更新挤进更少的方向，重叠的概率变大。
2. **但低秩 delta 的形状和我预想的完全不同**。我以为低秩意味着「结构化 = 好合并」，实测更精确的结论是：**低秩 delta 极度集中在单一方向上**——最大奇异值占了奇异值总和的 **0.8982**，也就是说 `ΔW1` 实质上是接近秩 1 的东西。同时它的**元素级稀疏度比全参数 delta 更低**（近零占比 0.41 vs 0.69）。两条一起读：低秩 ≠ 稀疏，低秩 = 「少量方向、每个方向都很密」。TIES 和 DARE 赌的是「少量坐标携带信号」，在 LoRA 上这个赌注下得更糟。

## 🔬 实验二：合并的结局一个字都没变——还是回到 θ0

| 方法 | acc_A | acc_B | 平均 |
|---|---|---|---|
| **θ0 预训练（λ=0，不合并）** | **0.8363** | **0.8333** | **0.8348** |
| linear avg (t=0.5) | 0.8323 | 0.8323 | 0.8323 |
| task arith (λ=1) | 0.8280 | 0.8327 | 0.8303 |
| task arith (λ=1.5) | 0.8243 | 0.8300 | 0.8272 |
| sum → 秩截断 r=8 | 0.8280 | 0.8327 | 0.8303 |
| sum → 秩截断 r=16 | 0.8280 | 0.8327 | 0.8303 |
| TIES (trim 20%) | 0.8193 | 0.7960 | 0.8077 |
| DARE-TIES (p=0.9) | 0.7910 | 0.7840 | 0.7875 |
| element-wise (ΣwA)(ΣwB), w=0.5 | 0.8263 | 0.8387 | **0.8325** |
| element-wise (ΣwA)(ΣwB), w=1.0 | 0.7983 | 0.8257 | 0.8120 |

**所有方法都低于 λ=0。** 最好的一个是 element-wise w=0.5（0.8325），依然比 θ0 低 0.23pp；唯一的「亮点」是它在 acc_B 上比 θ0 高 0.54pp，代价是 acc_A 低 1.00pp——这是典型的零和交换，不是能力回收。

把它和 48 篇的全参数微调对照摆在一起，才是这篇真正的信息量：

| 方法 | LoRA (r=8) 平均 | 全参数微调平均 | 相对 θ0（0.8348） |
|---|---|---|---|
| linear avg (t=0.5) | 0.8323 | 0.8358 | −0.25pp / +0.10pp |
| task arith (λ=1) | 0.8303 | 0.8347 | −0.45pp / −0.01pp |
| TIES (trim 20%) | 0.8077 | 0.8220 | −2.71pp / −1.28pp |
| DARE-TIES (p=0.9) | 0.7875 | 0.7493 | −4.73pp / −8.55pp |
| sum → 秩截断 r=8 | 0.8303 | 0.8347 | −0.45pp / −0.01pp |

**结论：换到低秩 delta，合并没有变好，反而更差一点**（task arith 从 −0.01pp 掉到 −0.45pp）。TIES / DARE 这两条「聪明路线」在 LoRA 上依然垫底（各 −2.26pp / −4.28pp，相对 task arith λ=1）。稀疏前提更不成立（近零占比 0.41 vs 0.69）这件事，并没有让它们变得更容易合并。

一个必须说清的细节：**DARE 的惩罚在 LoRA 上缩小了一半**（相对朴素求和：全参数 −8.54pp → LoRA −4.28pp）。为什么？因为 DARE 的伤害机制是「随机丢 90% 的坐标再乘 10 倍」，而在一个由单一方向主导（最大奇异值占比 0.8982）、坐标分布连续的 delta 上，乘 10 之后整体被同向放大，损失比在全参数 delta 上略小——**这是两份伤害的相对大小变了，不是低秩救了 DARE**。它依然是这张表里最差的一档。

## 🔬 实验三：秩拼接是恒等式，不是「保住两个专家」

LoRA 圈里有一条流传很广的说法：把两个适配器拼起来（而不是相加），就能同时保住两个能力。这在数学上对，但**它的含义和我一开始以为的不一样**：

```python
# 拼接：U = [U_A, U_B] (d×2r)，V = [V_A; V_B] (2r×d)
U_cat = np.concatenate([UA, UB], axis=1)
V_cat = np.concatenate([VA, VB], axis=0)
U_cat @ V_cat   # == UA@VA + UB@VB
```

实测恒等误差 `max|Δ| = 5.551e-17`（浮点零）。**所以「拼接」和「把两个 delta 相加」是同一个函数**——PEFT 博客自己写得很清楚（`B_merged A_merged = w₁s₁B₁A₁ + w₂s₂B₂A₂`）。它比相加多出来的东西不是能力，而是**秩**：产物是 2r，于是你获得了「重新截断回 r」的自由度。

那截断有没有损失？把 `τA + τB`（秩 ≤ 16）压回 r=8：

| 操作 | 保留的能量 | acc_A | acc_B | 平均 |
|---|---|---|---|---|
| 秩截断 r=16（= 不截断） | 1.000000 | 0.8280 | 0.8327 | 0.8303 |
| 秩截断 r=8 | **0.949742** | 0.8280 | 0.8327 | **0.8303** |
| 秩截断 r=4 | 0.885273 | — | — | （未测） |

`τA+τB` 的 W1 块奇异值是 `[1.6051, 0.2195, 0.0568, 0.0506, 0.0442, 0.0389, 0.0305, 0.0271, 0.0231, 0.0229, …]`——一个主方向吃掉 76% 的能量，第二、三个方向再吃掉 13%。所以**丢掉 5.03% 的能量，准确率一位不差**。这就是「拼接 + 截断」这个组合真正在做的事：**它把 2r 的存储压回 r，代价近似为零**。它不能把两个专家还给你，因为相加本身就不能（实验二的表已经证明了）。

## 🔬 实验四：element-wise 组合引入了 51% 的交叉项——而准确率没动

LoraHub / PEFT 的 `linear` 走的是另一条路：不先算各自的 `ΔW` 再相加，而是**先各自加权求和 U、V，再相乘**：

```
(Σw·U)(Σw·V) = Σw²·U·V + Σw·U_A·V_B + Σw·U_B·V_A     ← 后两项是交叉项
```

实测（w=0.5，即对 U、V 各取平均）：

- `(ΣwU)(ΣwV)` 与真·权重相加 `0.5(τA+τB)` 的差：**0.5019**，占目标模长的 **51.55%**。
- 准确率：element-wise w=0.5 → 0.8325；真·权重相加 → 0.8323。差 0.02pp。
- element-wise w=1.0 → 0.8120（权重没归一化，等价于把 delta 放大 4 倍，掉 2.28pp）。

**两条独立的结论叠在一起**：交叉项在函数空间里是**真实且巨大**的扰动（超过目标模长的一半），在准确率这把尺子上却是**不可见的**。这提醒我：toy 实验里只报准确率是很钝的镜头——两个参数完全不同的函数可以有同样的 3000 样本准确率。想真正区分，得看 logit 分布、置信度、以及分布外样本上的行为，而这篇没测（后面盲区里会再提）。

而 PEFT 把 `cat` 列为「多数场景首选」、把 element-wise `linear` 只作为备选的工程建议，在这次实验里得到了一个具体解释：`cat` 精确等于权重相加（误差 5.551e-17），`linear` 会额外引入一个不受控的 51% 扰动。

## 💻 可运行 Demo

下面这段是自包含的验证版（本机 14 秒跑完，随机流与主脚本一致，因此数字与上表完全相同）。完整脚本 `49-lora-merging.py` 还含中心差分梯度自检、秩能量谱、全参数微调对照组。

```python
import numpy as np

D, H, N, BETA, R = 32, 64, 1500, 1.0, 8
rng = np.random.default_rng(1000)


def unit(r):
    v = r.normal(size=D)
    return v / np.linalg.norm(v)


def orth(r, b):
    v = unit(r)
    v = v - (v @ b) * b
    return v / np.linalg.norm(v)


shared = unit(rng)
wA = shared + BETA * orth(rng, shared)
wB = shared + BETA * orth(rng, shared)
wA, wB = wA / np.linalg.norm(wA), wB / np.linalg.norm(wB)


def gen(r, w, n):
    X = r.normal(size=(n, D))
    return X, np.where(X @ w + 0.3 > 0, 1.0, -1.0)


XA, yA = gen(rng, wA, N)
XB, yB = gen(rng, wB, N)
XAe, yAe = gen(rng, wA, 3000)
XBe, yBe = gen(rng, wB, 3000)

# ---- θ0：全参数预训练，之后永久冻结（LoRA 的 base）----
th0 = {"W1": rng.normal(size=(D, H)) * 0.6 / np.sqrt(D), "b1": np.zeros(H),
       "W2": rng.normal(size=H) * 0.6 / np.sqrt(H), "b2": np.zeros(1)}


def fwd(p, X):
    Hh = np.tanh(X @ p["W1"] + p["b1"])
    return Hh, Hh @ p["W2"] + p["b2"][0]


def sigm(z):
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1 / (1 + e), e / (1 + e))


def gbase(p, X, y):
    Hh, lo = fwd(p, X)
    d = (-y) * sigm(-y * lo) / len(y)
    dZ = np.outer(d, p["W2"]) * (1 - Hh ** 2)
    return {"W1": X.T @ dZ, "b1": dZ.sum(0), "W2": Hh.T @ d, "b2": np.array([d.sum()])}


for _ in range(400):                       # 两任务混合预训练
    g = gbase(th0, np.concatenate([XA, XB]), np.concatenate([yA, yB]))
    th0 = {k: th0[k] - 0.15 * g[k] for k in th0}


# ---- LoRA：ΔW1 = U1@V1 (32×8 @ 8×64)，ΔW2 = U2@V2；V 初始化为 0 使 ΔW=0 ----
def lo_init(r):
    return {"U1": r.normal(size=(D, R)) * 0.6 / np.sqrt(D), "V1": np.zeros((R, H)),
            "U2": r.normal(size=(H, R)) * 0.6 / np.sqrt(H), "V2": np.zeros(R)}


def lof(l, X):
    W1 = th0["W1"] + l["U1"] @ l["V1"]
    W2 = th0["W2"] + l["U2"] @ l["V2"]
    Hh = np.tanh(X @ W1 + th0["b1"])
    return Hh, Hh @ W2 + th0["b2"][0], W2


def log(l, X, y):
    Hh, lo, W2 = lof(l, X)
    d = (-y) * sigm(-y * lo) / len(y)
    dW2 = Hh.T @ d
    dW1 = X.T @ (np.outer(d, W2) * (1 - Hh ** 2))
    return {"U1": dW1 @ l["V1"].T, "V1": l["U1"].T @ dW1,
            "U2": np.outer(dW2, l["V2"]), "V2": l["U2"].T @ dW2}


def train_lora(seed, X, y):
    l = lo_init(np.random.default_rng(seed))
    for _ in range(400):
        g = log(l, X, y)
        l = {k: l[k] - 0.08 * g[k] for k in l}
    return l


def lacc(l, X, y):
    return float(np.mean(np.sign(lof(l, X)[1]) == y))


lA, lB = train_lora(2001, XA, yA), train_lora(2002, XB, yB)
tA = np.concatenate([(lA["U1"] @ lA["V1"]).ravel(), (lA["U2"] @ lA["V2"]).ravel()])
tB = np.concatenate([(lB["U1"] @ lB["V1"]).ravel(), (lB["U2"] @ lB["V2"]).ravel()])


def acc(p, X, y):
    return float(np.mean(np.sign(fwd(p, X)[1]) == y))


def ev(t):
    p = {"W1": th0["W1"] + t[:D * H].reshape(D, H), "b1": th0["b1"],
         "W2": th0["W2"] + t[D * H:], "b2": th0["b2"]}
    a, b = acc(p, XAe, yAe), acc(p, XBe, yBe)
    return a, b, (a + b) / 2


def ties(vs, trim=0.2, drop=None, r=None):
    out = []
    for v in vs:
        if drop:
            v = v * (r.random(v.shape) > drop) / (1 - drop)
        if trim:
            m = np.zeros_like(v)
            m[np.argsort(-np.abs(v))[:int(round((1 - trim) * v.size))]] = 1.0
            v = v * m
        out.append(v)
    V = np.stack(out)
    el = np.sign(V.sum(0))
    Vm = np.where((np.sign(V) == el[None, :]) & (V != 0), V, 0.0)
    c = (Vm != 0).sum(0)
    return np.where(c > 0, Vm.sum(0) / np.maximum(c, 1), 0.0)


def trunc(t, k):
    U, S, Vt = np.linalg.svd(t[:D * H].reshape(D, H), full_matrices=False)
    return np.concatenate([((U[:, :k] * S[:k]) @ Vt[:k]).ravel(), t[D * H:]])


print(f"θ0            acc {acc(th0, XAe, yAe):.4f} / {acc(th0, XBe, yBe):.4f}")
print(f"LoRA A 专家   acc {lacc(lA, XAe, yAe):.4f} / {lacc(lA, XBe, yBe):.4f}  (交叉)")
print(f"LoRA B 专家   acc {lacc(lB, XAe, yAe):.4f} / {lacc(lB, XBe, yBe):.4f}  (交叉)")
print(f"cos(τA,τB) = {tA @ tB / np.linalg.norm(tA) / np.linalg.norm(tB):+.4f}")
thr = 0.05 * np.abs(tA).max()
print(f"LoRA delta 元素级「近零」占比 = {np.mean(np.abs(tA) < thr):.4f}")
for nm, t in (("λ=0 (=θ0)", np.zeros_like(tA)), ("linear avg t=0.5", 0.5 * (tA + tB)),
              ("task arith λ=1", tA + tB), ("TIES trim20%", ties([tA, tB])),
              ("DARE-TIES p=.9", ties([tA, tB], drop=0.9, r=np.random.default_rng(7))),
              ("sum→trunc rank 8", trunc(tA + tB, R))):
    a, b, m = ev(t)
    print(f"{nm:18s} acc_A {a:.4f}  acc_B {b:.4f}  mean {m:.4f}")
```

本机实际输出：

```
θ0            acc 0.8363 / 0.8333
LoRA A 专家   acc 0.9580 / 0.6697  (交叉)
LoRA B 专家   acc 0.6740 / 0.9587  (交叉)
cos(τA,τB) = -0.4282
LoRA delta 元素级「近零」占比 = 0.4129
λ=0 (=θ0)          acc_A 0.8363  acc_B 0.8333  mean 0.8348
linear avg t=0.5   acc_A 0.8323  acc_B 0.8323  mean 0.8323
task arith λ=1     acc_A 0.8280  acc_B 0.8327  mean 0.8303
TIES trim20%       acc_A 0.8193  acc_B 0.7960  mean 0.8077
DARE-TIES p=.9     acc_A 0.7910  acc_B 0.7840  mean 0.7875
sum→trunc rank 8   acc_A 0.8280  acc_B 0.8327  mean 0.8303
```

## 💡 结论与盲区

**能站住的三条：**

1. **低秩降低了冲突，但没有改变合并的结局。** cos(τA,τB) 从全参数微调的 −0.5638 改善到 −0.4282，可它仍然是负数；`θ0 + 0.5(τA+τB) ≈ θ0` 这个代数后果依然成立。LoRA 上最好的合并结果 0.8325，低于 λ=0 的 0.8348。**合并回收的仍然是公约数。** 低秩买到的是便宜（1288 vs 2112 个可训练参数、2r 的存储），不是能力。
2. **低秩 ≠ 稀疏，这是 TIES/DARE 在 LoRA 上更不适用的直接原因。** LoRA delta 的元素级近零占比只有 0.4129（全参数 delta 是 0.6851），而它 89.82% 的奇异值能量集中在**单一方向**上——「少方向、满坐标」，正好与 TIES 的 trim 前提相反。实测 TIES −2.26pp、DARE-TIES −4.28pp（相对 task arith λ=1），且与 PEFT 博客「dare_ties wasn't giving good results」的现场观察一致。
3. **秩拼接是恒等式 + 一个免费的压缩机会。** `[U_A,U_B] @ [V_A;V_B] = U_A V_A + U_B V_B`，实测误差 5.551e-17；它不提供任何相加之外的表达力（PEFT 的 `cat` 定义本身也这么说）。它真正的用处是把 2r 压回 r：截断保留 94.97% 能量，准确率与不截断**完全相同**（0.8303）。而 element-wise 的 `(ΣwU)(ΣwV)` 会引入占模长 51.55% 的交叉项，w=1.0 时直接掉到 0.8120——要用就用 `cat` 或权重相加，别用未归一化的 element-wise。

**盲区（这才是这篇最重要的部分）：**

- **「低秩」在这个玩具里不够低。** r=8 对 H=64 是 12.5% 的秩，可训练参数 1288 相当于全参数 delta 的 61%。真实 LoRA 常配 r=8/16 在 4096 维的层上，r/d 是千分之几。我的 delta 最大奇异值占比 0.8982 已经接近秩 1，说明这个 setting 更像「秩 1 vs 满秩」的极端对比，而不是真实 LoRA 的比例。
- **只测了准确率，没测函数。** 实验四里 element-wise 与权重相加的参数差了 51.55% 的模长，准确率只差 0.02pp。这说明我这把尺子很钝——真实场景里这种差异会在 logit 校准、拒答率、输出长度、语言漂移上显现，而这三项我一个都没测。
- **单 seed、单 β、两任务。** 48 篇做了 3 seed × 4 档 β，这篇为了省算力只跑了 β=1.0 单 seed。数字的稳定性没有误差棒支撑，尤其是 TIES / DARE 那两行（−2~−5pp 的空隙里，seed 波动可能有 0.3~0.5pp）。
- **没扫 r、没扫 trim/density。** 我只跑了 r=8、trim=20%、p=0.9。PEFT 的建议是 ties 用 `density=0.5`、dare 系列用 `density=0.7~0.8`——我用的 20% 保留率比它保守得多。所以能说的只有「在这个保守设定下不行」。
- **`svd` 与 `cat` 在真实 LoRA 上的对比没做。** PEFT 博客明确说 `cat` 合并大量适配器会 OOM、`*_svd` 系列需要额外显存且慢约 1.5 分钟。省内存和保精度之间的权衡，是这个话题真正落到生产时的核心问题，我这次只量了两端。
- **没有跨 rank、跨任务类型、没有真正同源的适配器。** 我的两个适配器共享同一个 θ0、同一批输入分布，这已经是理想化的「同源」了；真实适配器常在不同数据、不同 rank、甚至不同基座上训练。

> 一句话：**低秩让合并更便宜，没有让它更划算。** 想拿回两个专家，得靠路由或 MoE 式的分开部署；想省存储，`cat` 加上秩截断是这次唯一被实测支持的免费午餐——它保留 94.97% 的能量，却一个准确率点都不掉。
