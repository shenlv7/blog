---
title: "λ 从 0.001 调到 100，参数走的还是同一条路：第 87 步过零、翻 3 次符号——Adam 的 L2 项被 1/√v̂ 吃掉了，同样的 λ=0.1 下，低梯度那 4 个权重只长到真值的 0.181，高梯度那 4 个到 0.959"
description: "同一个 λ=0.1：特征尺度差 10 倍的两组权重，Adam+L2 收敛到 0.959 / 0.181，test MSE 0.0411；wd=0 的对照是 0.0106，换成解耦的 AdamW 是 0.994 / 0.970、0.0107。给一个没有任何梯度的参数做纯衰减，λ 从 0.001 扫到 100，过零点、翻符号次数、200 步后的残值全部重合到第 5 位有效数字——因为耦合式的每一步都是 η·λ|θ|/√v̂，而这时 √v̂ 恰好等于 λ|θ|。实测每参数等效衰减跨度 6.6 倍，AdamW 那一侧恒等于 λ。"
pubDate: 2026-11-15
tags: ["AI核心概念", "优化器", "AdamW", "权重衰减", "L2正则", "自适应学习率"]
difficulty: "advanced"
series: "ai-concepts"
seriesOrder: 83
slug: "ai-concepts-83-decoupled-weight-decay"
---

> 给一个**没有任何数据梯度**的参数做纯衰减，把 λ 从 0.001 改成 100（十万倍）：
> 过零点都在**第 87 步**，翻符号都是 **3 次**，200 步后残值都在 **2.19e-5** —— 路径一模一样。
> 因为耦合式每步走的是 $\eta\lambda|\theta|/\sqrt{\hat v}$，而无梯度时 $\sqrt{\hat v}=\lambda|\theta|$，**λ 被自己约掉了**。
> 换成 AdamW，200 步后的残值严格等于 $(1-\eta\lambda)^{200}$：0.9960 / 0.6701 / 0.0176，和理论对到四位小数。
> 同一个 λ=0.1 丢进一个特征尺度差 10 倍的线性回归：耦合版把低梯度那 4 个权重压到真值的 **0.181**，高梯度那 4 个留在 **0.959**；解耦版是 **0.970 / 0.994**，而 wd=0 的对照是 **0.996 / 1.003**。

读这篇要会：Adam 的一阶/二阶矩估计与偏差修正（$m,v,\hat m,\hat v$）/ L2 正则在 SGD 下等价于权重衰减的推导 / 线性回归的最小二乘梯度 $\frac{2}{B}X^\top(Xw-y)$。

## 🎯 在 SGD 里，这两件事确实是一回事

L2 正则加在损失上，梯度里就多一项 $\lambda\theta$。写开一步 SGD：

$$
\theta_{t+1} = \theta_t - \eta\big(\nabla L(\theta_t) + \lambda\theta_t\big)
= (1-\eta\lambda)\,\theta_t - \eta\nabla L(\theta_t)
$$

右边那个整体乘上去的 $(1-\eta\lambda)$ 就是「衰减」这个动作本身：**它不看梯度，只看参数自己有多大**，每步砍掉固定比例 $\eta\lambda$。所以「加 L2」和「乘性收缩」在 SGD 下无法区分，叫法混用到今天。

Adam 把这层等价性破坏了，因为梯度不是原样用的，是被二阶矩归一化过的：

$$
\text{Adam + L2:}\quad \theta_{t+1} = \theta_t - \eta\,\frac{\nabla L(\theta_t) + \lambda\theta_t}{\sqrt{\hat v_t}+\epsilon}
\qquad
\text{AdamW:}\quad \theta_{t+1} = \theta_t - \eta\,\frac{\nabla L(\theta_t)}{\sqrt{\hat v_t}+\epsilon} - \eta\lambda\theta_t
$$

差别只在**那个除法作用在谁身上**。耦合版（左）把 $\lambda\theta$ 一并塞进了分母 $\sqrt{\hat v}$；解耦版（右）让衰减完全走一条独立的乘法通道。把耦合式里属于衰减的那一半单独拎出来，除以 $|\theta|$ 就得到**每参数的等效衰减系数**：

$$
\lambda_{\text{eff}} \;=\; \frac{\lambda}{\sqrt{\hat v_t}}
$$

$\hat v_t$ 是这一维梯度平方的指数滑动平均，量级就是「这一维梯度有多大」的平方。于是 $\lambda_{\text{eff}} \propto \lambda/s$，$s$ 是梯度尺度：**梯度小 10 倍的参数，被衰减 10 倍狠**。这一条是从上面那个公式直接读出来的，不需要训练就能验证。

## 🔬 没有梯度的参数：λ 被完整约掉

先做最干净的对照。取一个标量参数 $\theta_0=1$，**数据梯度恒为 0**（它在损失里不出现），只让正则项拉它。$\eta=0.02$，跑 200 步：

| λ | 耦合式过零步 | 耦合式翻符号 | 耦合式 200 步残值 | AdamW 残值 | 理论 $(1-\eta\lambda)^{200}$ |
|---|---|---|---|---|---|
| 0.001 | 87 | 3 | 2.1887e-05 | 0.9960 | 0.9960 |
| 0.01 | 87 | 3 | 2.1908e-05 | 0.9608 | 0.9608 |
| 0.1 | 87 | 3 | 2.1908e-05 | 0.6701 | 0.6701 |
| 1.0 | 87 | 3 | 2.1908e-05 | 0.0176 | 0.0176 |
| 100 | 87 | 3 | 2.1908e-05 | 1.0000 | 1.0000 |

右三列是解耦，左三列是耦合。**耦合那三列的五档 λ 重合到第 5 位有效数字**，解耦那三列和解析式完全一致。

为什么？没有数据梯度时，梯度只能来自正则项自己：$g=\lambda\theta$，于是 $\hat v=\lambda^2\theta^2$、$\sqrt{\hat v}=\lambda|\theta|$。代进耦合式：

$$
\Delta\theta = -\,\eta\,\frac{\lambda\theta}{\lambda|\theta|} = -\eta\,\mathrm{sign}(\theta)
$$

**每步走固定长度 $\eta$，λ 完全消失。** 前 5 步实测 0.9800 → 0.9600 → 0.9400 → 0.9201 → 0.9002，正是每步 $-0.02$ 的匀速下滑；走到 0 之后 sign 翻转，在零点附近抖 3 次才停。第 87 步过零而不是理论上的 $1/\eta=50$ 步，是因为 $\beta_2=0.999$ 让 $\hat v$ 滞后于 $\theta^2$，分母偏大、步长略小于 $\eta$ —— 滞后量级可预测，机制不变。

这件事的工程含义比数字本身刺眼：**在「梯度本来就小」的参数上，耦合实现的 λ 几乎不起作用**（它不是「轻微正则」，而是完全失控——衰减量由 $\eta$ 决定，不由 λ 决定）。而在这台机器上，梯度小恰恰是 embedding 低频行、bias、冻结层旁支的常态。

## 🏗️ 有梯度时，账记在 √v̂ 上

纯衰减是极端情形。加回数据，做一个 8 维线性回归：前 4 维输入尺度 1，后 4 维输入尺度 **0.1**，真值全为 1。这样两组权重的梯度量级天然差一个数量级，而 Adam 本身对尺度不敏感——**先把 wd=0 的对照跑出来，确认两组本来能收敛到同一个地方**，否则后面的差异全是收敛不足的假象：

| 方案 | 前 4 维（梯度大）| 后 4 维（梯度小）| train MSE | test MSE |
|---|---|---|---|---|
| wd=0（对照） | 1.003 | 0.996 | 0.0103 | 0.0106 |
| Adam + L2，λ=0.1 | 0.959 | **0.181** | 0.0448 | **0.0411** |
| AdamW，λ=0.1 | 0.994 | 0.970 | 0.0106 | 0.0107 |

$\eta=0.01$，2000 步，batch 64。

三行读下来：

- **对照行证明这个线性回归是可学的**：两组都到 1.0 附近（1.003 / 0.996），误差就是 $10^{-2}$ 的噪声底。这是后面所有比较的基准线。
- **耦合版把低梯度组压到 0.181**，同一份 λ 下高梯度组只掉到 0.959 —— **差了 5.35 倍**。test MSE 0.0411 是对照的 **3.84 倍**。
- **解耦版几乎没有伤到任何一组**（0.994 / 0.970），test MSE 0.0107，和对照 0.0106 差一位小数。

换句话说，那个 3.84 倍的测试误差不是「正则太强」，是**正则挑人**：它按 $1/\sqrt{\hat v}$ 分配，谁梯度小就罚谁，跟谁需要正则化毫无关系。

第三个测量直接把 $\lambda_{\text{eff}}$ 量出来（取第 100–400 步的均值，避开暖机）：

| 分组 | $\lambda_{\text{eff}}=\lambda|\theta|/\sqrt{\hat v}$ |
|---|---|
| 前 4 维（梯度大） | 0.112 |
| 后 4 维（梯度小） | 0.738 |
| 跨度 | **6.6 倍** |
| AdamW 那一侧 | 两组都恒等于 λ = 0.100 |

特征尺度差 10 倍（末态梯度范数实测 0.165 对 0.034，差 4.9 倍），等效衰减差 6.6 倍——**同一个超参 λ，在两组参数上执行的实际强度差了一位数**。这就是「Adam + weight_decay」这个写法在 2017 年之前到处都是、却没人说得清它到底在干什么的原因。

## 💻 四十行量完两种写法

核心就是两个分支的差别：耦合把 $\lambda\theta$ 加进梯度，解耦从参数里直接扣。下面这段是上面两张表的全部代码：

```python
import numpy as np

class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.decoupled = wd, decoupled
        self.m = self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta); self.v = np.zeros_like(theta)
        self.t += 1
        if not self.decoupled:
            grad = grad + self.wd * theta                  # L2 写进梯度
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        theta = theta - self.lr * mh / (np.sqrt(vh) + self.eps)
        if self.decoupled:
            theta = theta - self.lr * self.wd * theta      # 直接从参数里扣
        return theta

# A. 纯衰减：数据梯度恒为 0
for wd in [0.001, 0.1, 1.0, 100.0]:
    for dec in (False, True):
        th = np.array([1.0]); opt = Adam(lr=0.02, wd=wd, decoupled=dec)
        for _ in range(200):
            th = opt.step(th, np.zeros_like(th))
        print(f"λ={wd:<7} decoupled={dec!s:<5} θ_200={th[0]: .4e}")
# → 耦合：全部 2.19e-05   |   解耦：0.9960 / 0.6701 / 0.0176 / 1.0000

# B. 异尺度回归：后 4 维输入乘 0.1，真值全 1
rng = np.random.default_rng(1)
X = np.concatenate([rng.normal(size=(1024, 4)),
                    rng.normal(size=(1024, 4)) * 0.1], axis=1)
y = X @ np.ones(8) + rng.normal(scale=0.1, size=1024)

def train(wd, decoupled, steps=2000):
    r = np.random.default_rng(2); w = np.zeros(8)
    opt = Adam(lr=0.01, wd=wd, decoupled=decoupled)
    for _ in range(steps):
        i = r.integers(0, 800, 64)
        w = opt.step(w, 2 * X[i].T @ (X[i] @ w - y[i]) / 64)
    return w

for name, wd, dec in [("wd=0", 0.0, True), ("Adam+L2", 0.1, False), ("AdamW", 0.1, True)]:
    w = train(wd, dec)
    mse = np.mean((X[800:] @ w - y[800:]) ** 2)
    print(f"{name:<8} 前4维 {w[:4].mean():.3f}  后4维 {w[4:].mean():.3f}  test MSE {mse:.4f}")
```

完整脚本（含 $\lambda_{\text{eff}}$ 的逐参数统计、λ 扫描、轨迹打印）：
[demos/ai-concepts-83-decoupled-weight-decay.py](https://github.com/shenlv7/blog/blob/main/demos/ai-concepts-83-decoupled-weight-decay.py)

## 💡 四个能直接用在训练脚本上的结论

**一、真正起作用的是 $\eta\lambda$，不是 λ。** 解耦衰减每步乘 $(1-\eta\lambda)$，跑 $T$ 步总共乘 $e^{-\lambda\sum_t \eta_t}$ —— **衰减强度等于 λ 乘以学习率的积分**。所以 lr schedule 一动，有效正则就跟着动：$\eta=10^{-3}$、λ=0.1、跑 1 万步，权重还剩 $e^{-1}=0.368$；同一份 λ 换成 $\eta=10^{-4}$，1 万步后还剩 $e^{-0.1}=0.905$。**换 lr 就必须重扫 λ**，否则你以为只调了学习率。反过来，cosine 衰减到 0 的末尾几百步几乎不再正则，这也是「最后一小段 lr 极低时权重范数不再下降」的原因。

**二、bias 和归一化层的 gain 不该衰减。** 这两类参数的最优值不是「越小越好」，把它们拉向 0 是纯粹的破坏。这件事在解耦实现里是干净的参数分组（decay / no_decay 两个 param group），在耦合实现里做不干净——因为 L2 项一旦进入梯度，就被 $\sqrt{\hat v}$ 按尺度重新加权了，你没法保证「比例一致」。

**三、输入没归一化的时候，耦合版会系统性偏袒。** 上面那张表就是它的显影：特征尺度 0.1 的那 4 个权重被压到 0.181，另外 4 个留在 0.959。如果你的 pipeline 里有量纲不同的特征（价格 vs 比例、长文本 vs 短文本的 embedding），耦合实现在做的是**按梯度尺度分配正则**，而你的直觉是「一视同仁」。

**四、跨实现迁移 λ 必须先换算。** 耦合式的 λ 等价于 $\lambda/\sqrt{\hat v}$，量级随梯度尺度跳一个数量级；解耦式的 λ 是纯比例。旧代码 `Adam(..., weight_decay=1e-4)` 里的那个数，搬进 `AdamW` 直接照抄，实际正则强度可能差 10 倍以上——**方向还可能相反**（低梯度参数从「几乎不正则」变成「强正则」）。

**这几个数字的边界写在测量里**：以上是标量玩具与 8 维线性回归，用 numpy 手写、无框架，不是 transformer 训练。$\lambda_{\text{eff}}=\lambda/\sqrt{\hat v}$、以及 $\sqrt{\hat v}=\lambda|\theta|$ 时 λ 被约掉，这两条是公式的直接推论，与模型规模无关；而 0.181 对 0.970 的差距依赖于「两组梯度尺度差 10 倍」这个设定——尺度差 2 倍时差距会小到看不出来，这也是为什么这个问题在归一化良好的大模型上常被掩盖、在小模型和宽 embedding 表上突然致命。
