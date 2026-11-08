---
title: "lr=0.3 时 Adam 的第一步就把参数搬走 44%：eval loss 从 0.77 升到 1.83"
description: "同一个 batch、同一个 lr，SGD 把 257 个参数挪了 0.40，Adam 挪了 4.81——11.9 倍，占初始参数范数的 44.4%。实测证实这个位移恒等于 lr·√d：把梯度整体乘 1000 或除以 1000，位移从 4.8094 变到 4.7540（差的那点来自 ε 兜底），batch 从 8 换到 2048 也一动不动。原因是偏差修正后的 Adam 在 t=1 就是 sign 下降。warmup 压的正是这段「步长满格、方向只有 1 bit」的窗口。"
pubDate: 2026-11-08
tags: ["AI核心概念", "优化器", "Adam", "warmup", "训练稳定性", "signSGD"]
difficulty: "intermediate"
series: "ai-concepts"
seriesOrder: 76
slug: "ai-concepts-76-adam-first-step-warmup"
---

> 同一个 batch、同一个 lr：SGD 把 257 个参数挪了 0.403，Adam 挪了 4.809——11.9 倍。
> lr=0.3 时那是初始参数范数的 44.4%。走完这一步，eval loss 从 0.7671 变成 1.8296。
> 把梯度整体乘 1000 或除以 1000，位移仍然是 4.809，一个字节都不动。

**读这篇要会**：Adam 的一阶/二阶矩与偏差修正怎么写／对一个两层 MLP 手推反向传播／向量范数与余弦相似度。不需要跑过大规模训练。

## 🎯 第一步就走满全程

学率的直觉大多建立在这句话上：梯度小的时候步子小，梯度大的时候步子大。SGD 是这样，而 Adam——现在几乎所有 LLM 训练用的优化器——在第一步不是这样。

实验用的是我自己写的一个 257 参数的小网（2 维输入、64 个 tanh 隐单元、二分类），two-moons 数据加 10% 标签翻转。同一批 64 个样本、同一个初始化，只改优化器和学习率：

| lr | SGD 位移 \|Δθ\| | Adam 位移 \|Δθ\| | Adam/SGD | 占 \|θ₀\|=10.83 的比例 |
|---|---|---|---|---|
| 0.01 | 0.0134 | 0.1603 | 11.9× | 1.5% |
| 0.10 | 0.1345 | 1.6031 | 11.9× | 14.8% |
| 0.30 | 0.4034 | 4.8093 | 11.9× | 44.4% |

两列都随 lr 线性放大，但比例 11.9 是常数。换句话说：**Adam 的步长不含任何关于梯度大小或数据难度的信息**，它只由 lr 和参数个数决定。lr=0.3 意味着第一步就把整个模型在参数空间里搬走了 44% 的自身范数——模型被重写了一遍，方向由这 64 个样本说了算。

## 🔬 机制：偏差修正把 t=1 变成符号下降

Adam 的更新式（Kingma & Ba, *Adam: A Method for Stochastic Optimization*, ICLR 2015, arXiv:1412.6980）：

$$m_t=\beta_1 m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2$$

$$\hat m_t=\frac{m_t}{1-\beta_1^t},\qquad \hat v_t=\frac{v_t}{1-\beta_2^t},\qquad \theta_{t}\leftarrow\theta_{t-1}-\eta\,\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}$$

$m,v$ 初始化为 0 会把估计拉向 0，分母 $1-\beta^t$ 就是修这个的。问题出在 $t=1$：代入 $m_0=v_0=0$ 得到 $m_1=(1-\beta_1)g_1$、$\hat m_1=g_1$，同理 $\hat v_1=g_1^2$。于是

$$u_1=\eta\,\frac{g_1}{|g_1|+\varepsilon}=\eta\,\operatorname{sign}(g_1),\qquad \|u_1\|_2=\eta\sqrt{d}$$

逐坐标看，每个参数都恰好走 $\eta$，符号取自这个 batch 的梯度。梯度幅值的信息被 $\sqrt{\hat v}$ 除干净了。这不是近似：实测在 d=257、$\eta=0.3$ 下，$\eta\sqrt d=4.8094$，量到的位移是 4.8093。

于是有了下面这张反直觉的表——改动梯度本身，位移完全不变：

| 改动 | Adam 首步位移（lr=0.3） |
|---|---|
| 梯度 ×0.001 | 4.7540 |
| 梯度 ×1 | 4.8093 |
| 梯度 ×1000 | 4.8094 |
| batch = 8 | 4.8093 |
| batch = 64 | 4.8093 |
| batch = 2048 | 4.8092 |

两处细节值得说明。梯度 ×0.001 那一行掉了 1.2%，是因为 $\varepsilon=10^{-8}$ 开始在部分坐标上头起作用：$|g_i|$ 掉到 $10^{-8}$ 量级时，分母被 $\varepsilon$ 兜住，那一步就不再是精确的 $\eta$。这也是混合精度训练里 Adam 的状态必须用 fp32 master weights 存的原因（Narang et al., *Mixed Precision Training*, arXiv:1710.03740）：$\varepsilon=10^{-8}$ 在 fp16 里直接下溢成 0。第二处是 batch：**把 batch 从 8 加到 2048 不会让首步变小，只会让方向变准**。

方向准到什么程度，是另一回事。t=1 时 Adam 对每个参数只保留了 1 bit（符号），这和 signSGD（Bernstein et al., *signSGD: Compressed Optimisation for Non-Convex Problems*, ICML 2018, arXiv:1802.04434）是同一个更新规则。拿首步更新方向跟 8192 样本全量梯度的首步方向比余弦相似度：

| batch | cos(u₁, u_ref) |
|---|---|
| 8 | 0.253 ± 0.073 |
| 64 | 0.576 ± 0.042 |
| 512 | 0.782 ± 0.039 |
| 2048 | 0.703 ± 0.039 |

方向可信度确实随 batch 上升，但在 0.8 附近见顶，2048 反而掉回 0.70。原因是符号化：那些真实梯度本来就接近 0 的坐标，其符号由噪声决定，样本再多也在正负之间跳——放大的只是"有信号的坐标"的稳定性，而不是全部。最终效果是**一次满格步长，配一个部分可信的方向，作用在 257 维的参数上**。

后面的步会好起来，机制在 $v_t$：$\beta_2=0.999$ 对应约 1000 步的时间常数，$\sqrt{\hat v_t}$ 从「单样本梯度的绝对值」逐渐变成「梯度二阶矩的估计」。更新幅值到那时才开始反映曲率，Adam 才像自适应优化器。

一步之差的代价，直接量在 eval loss 上：

| lr（等价的 \|Δθ\|） | 一步之后的 eval loss |
|---|---|
| 起点 | 0.7671 |
| 0.003（0.048） | 0.7406 |
| 0.010（0.160） | 0.6895 |
| 0.030（0.481） | 0.6171 |
| 0.100（1.603） | 0.8031 |
| 0.300（4.809） | 1.8296 |
| 1.000（16.03） | 2.9513 |

一步之内存在最优 lr：0.03 是这一步能买到的最好结果，0.1 已经回吐，0.3 直接翻倍。注意这条曲线的自变量不是"学习率合不合适"，而是"这一步搬了多少参数"——Adam 里两者是同一个数乘常数。

## 🏗️ warmup 压的就是这段窗口

warmup 的做法是把第 $t$ 步的 lr 乘上 $t/W$：

$$\eta_t=\eta\cdot\min\left(1,\frac{t}{W}\right),\qquad \|u_t\|=\eta_t\sqrt d=\frac{\eta\sqrt d\,t}{W}$$

首步位移从 $\eta\sqrt d$ 变成 $\eta\sqrt d/W$。实测 $\eta=0.3$、$W=50$ 时首步位移 0.096（一步后 eval 0.7169，起点 0.7671），$W=200$ 时 0.024（0.7535）。前者相当于把 lr 有效限制在 0.006，后者接近原地不动。$W$ 步之后 lr 才恢复到满格，而那时 $v_t$ 已经积了几百个样本的信息。

按上式还能直接算出一个量：前 $W$ 步的累计路径长度是

$$\sum_{t=1}^{W}\frac{\eta\sqrt d\,t}{W}=\frac{\eta\sqrt d\,(W+1)}{2}$$

$W=200$ 时约 483，是初始参数范数的 45 倍。累计路径不等于净位移（方向在翻），但这个数字说明 warmup 阶段优化器并没有"闲着"——它走了很长一段路，只是每一步都没走出模型的信任范围。

Transformer 的原始配置就是这个形状：Vaswani 等（*Attention Is All You Need*, 2017）的 lr 是

$$\text{lr}=d_{\text{model}}^{-0.5}\cdot\min\left(\text{step}^{-0.5},\ \text{step}\cdot\text{warmup\_steps}^{-1.5}\right)$$

base 模型取 warmup_steps=4000，而 base 模型的训练总步数是 100,000——warmup 占 4%。RAdam（Liu et al., *On the Variance of the Adaptive Learning Rate and Beyond*, arXiv:1908.03265）后来把这件经验事做成了理论：早期 $\hat v_t$ 的方差极大，自适应学习率的有效步长方差同样极大，warmup 的作用是把这段高方差区间的实际步长压到接近 0。

工程上还有一个常被忽略的推论：**从 checkpoint 恢复训练，如果没存 optimizer state，$m,v$ 从 0 重新开始，第一步又是满速 sign 下降**。这正是"续训后 loss 突然冲高"最常见的来源之一，和 [loss 尖峰与 Adam 二阶矩](/blog/posts/ai-concepts-73-loss-spike-adam/) 是同一个机制的两端。恢复训练时重放几十步短 warmup，成本几乎为零。

## 💻 最小复现

完整脚本（含手写反向传播，无 PyTorch 依赖）：[demos/ai-concepts-76-adam-first-step-warmup.py](https://github.com/shenlv7/blog/blob/main/demos/ai-concepts-76-adam-first-step-warmup.py)。核心就是 Adam 的 step 返回更新向量，再直接量它的范数：

```python
class Adam:
    def __init__(self, p):
        self.m = np.zeros_like(flat(p)); self.v = np.zeros_like(flat(p)); self.t = 0
    def step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g ** 2
        mhat = self.m / (1 - b1 ** self.t)
        vhat = self.v / (1 - b2 ** self.t)
        return lr * mhat / (np.sqrt(vhat) + eps)   # 这就是参数位移

print(f'{np.linalg.norm(Adam(p).step(g, 0.3)):.4f}  vs  lr*sqrt(d) = {0.3*np.sqrt(257):.4f}')
# 4.8093  vs  4.8094
```

真实输出（本机跑，1.0 秒）：参数 d=257、|θ₀|=10.83、初始 eval loss=0.7671；lr=0.3 时 SGD 位移 0.4034 / Adam 位移 4.8093 / 比值 11.9×；梯度 ×1000 与 batch=2048 的位移都不变；一步后 eval loss 1.8296；warmup W=200 时首步位移 0.0240。

这里只报三步以内的量是刻意的。在这个小模型上跑 200 步以上，10% 标签翻转带来的 loss 方差大于 warmup 的效应，最终精度的对比不可复现；几何量是确定性的，才值得写进文章。

## 💡 实践清单

- **判断 lr 是否过大，别看 loss 抖不抖，看首步位移**。$\|u_1\|=\eta\sqrt d$，跟 $|\theta_0|$ 一比就知道这一步有多暴力。这个 257 参数的小模型上，单步位移 4.4%（lr=0.03）还是净收益，14.8%（lr=0.1）开始回吐，44%（lr=0.3）直接把 loss 翻倍。
- **batch 救不了首步**。4.8093（B=8）对 4.8092（B=2048）。加大 batch 只改善方向，不改善步长；要压步长只有降 lr 或加 warmup。
- **warmup 长度按 1%–2% 总步数给**，首步位移落在 $\eta\sqrt d/W$，$W$ 取到让这个数明显小于参数范数即可。
- **续训必须重放 warmup**，因为 optimizer state 丢了，$m,v$ 归零，第一步回到 sign 下降。
- **混合精度下检查 ε**。$\varepsilon=10^{-8}$ 在 fp16 里下溢，Adam 的二阶矩和 $\varepsilon$ 都要留在 fp32。

比起调衰减曲线，这一步的收益更确定：它决定的不是模型最后收敛到哪，而是训练最初几千步会不会被一个 batch 的方向带偏。相关的一张卡是 [Muon 与正交化更新](/blog/posts/ai-concepts-67-muon/)——那条路线干脆把"逐坐标归一化"换成"矩阵整体正交化"，动机正是这里暴露的：逐坐标满格步长，丢掉了梯度的大小与方向耦合信息。
