---
title: "余弦退火写死了训练长度：WSD 给中间 checkpoint 补 200 步退火拿到 0.0197，从头跑 2200 步余弦是 0.0451"
description: "单隐层 MLP + Adam、4096 样本、3 个种子的本机实测（非 GPU、非 LLM）：同样 3000 步，余弦退火验证 loss 0.0252、WSD 0.0133；WSD 一条 run 的 2000 步 checkpoint 再退火 200 步就到 0.0197，而从头跑 2200 步余弦是 0.0451（2.3 倍）。要把 loss-算力曲线的两个点都拿到，WSD 花 3300 步，余弦要重跑三条共 6300 步。给中间 checkpoint 续训时两者都还能降（0.0140 vs 0.0181），但这个玩具规模上没复现出「rewarm 尖峰」——论文在 210M 参数上观察到的是「打平 + 尖峰」。"
pubDate: 2026-11-26
tags: ["AI核心概念", "学习率调度", "WSD", "余弦退火", "预训练", "scaling law"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 94
slug: "ai-concepts-94-wsd-lr-schedule"
---

> 余弦退火的性能没问题，问题在它的形状把**总步数 N 写进了每一个 step 的学习率里**：跑到第 2000 步时，LR 已经只剩峰值的 27%，因为函数"知道"还有 1000 步就要结束。
> WSD 把退火压到最后 20%：前 80% 恒定在峰值，退火是一次**可以后置**的动作。于是中间 checkpoint 不是"半成品"，而是一个随时能退火定型的原始模型。
> 本机 3 个种子上，2000 步的 checkpoint 补 200 步 cooldown 得到验证 loss 0.0197；同样总步数的余弦从头跑是 0.0451。

读这篇要会：Adam 的更新式（一阶/二阶矩）/ warmup 在防什么 / 训练曲线里「loss 随时间下降」和「loss 随累计算力下降」是两条不同的曲线。

## 🎯 卡点：余弦调度把「计划」编译进了每一步

cosine decay 的定义里，总步数 $N$ 是分母：

$$\eta(t) = \eta_{\max}\cdot\frac{1}{2}\left(1+\cos\frac{\pi (t-t_w)}{N-t_w}\right)$$

这意味着两件事：第 $t$ 步的 LR 取决于"你还剩多少步"，而不是"你学到了多少"。工程上这带来三个具体麻烦，都不是性能问题：

1. **加数据续训要重新排程**。数据多了、想从 3000 步拉到 6000 步，余弦曲线得重画；从已退火到 0 的终点继续训练，等于把 LR 从 0 拉回峰值——这就是 rewarm，模型要在噪声里重新找一遍盆地。
2. **scaling law 实验要重跑**。想知道"训 1000 / 2000 / 4000 步各自的 loss"，余弦必须跑三次独立的 run，因为 1000 步那条的 LR 曲线和 4000 步那条根本不是同一条。
3. **中间 checkpoint 不可比**。2000 步存下来的权重，是"按 3000 步计划已经退火 40%"的状态，跟另一个预算下的同 step checkpoint 不在同一水平线上。

WSD（Warmup-Stable-Decay）把这三件事解耦：退火是一个**独立阶段**，你可以决定现在退、也可以决定再跑一万步再退。

## 🔬 两个调度，差别只在退火何时发生

WSD 的形式（MiniCPM 论文式 (1)，省掉 warmup 项）：

$$\eta(n)=\begin{cases}\eta_{\max}, & n \le T \\ f(n-T)\cdot\eta_{\max}, & T < n \le S\end{cases}$$

$T$ 是稳定段的结束点，$S$ 是总步数，$f$ 单调降到 0。本机实现里 $f$ 取线性、退火段占总步数 20%。余弦那条则从 warmup 结束就开始掉。

| 步数 | 余弦 LR（峰值 3e-3） | WSD LR |
|---|---|---|
| 1000 | 2.34e-3（78%） | 3.0e-3（100%） |
| 2000 | 7.97e-4（27%） | 3.0e-3（100%） |
| 2700 | 7.85e-5（3%） | 1.5e-3（50%，退火中） |

退火不是可选项：高 LR 下权重在损失面上跳，最后一段把 LR 压低，等于在低噪声下把参数"最后推一把"。MiniCPM 的论文里把这个画成 loss 在 decay 阶段的一次**台阶式下降**（*MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies*, Hu et al., arXiv:2404.06395）。他们也正是用 WSD 做的 data-model scaling law——因为一条长 run 的多个 checkpoint 可以各自退火，得到整条 loss-算力曲线。同一篇论文报告：把一个 0.036B 模型连续训练下去，可以追平 0.17B 模型（论文图表数字，本文未复现）。

更系统的对照来自 Hägele 等（*Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations*, NeurIPS 2024, arXiv:2405.18392）：210M 参数、SlimPajama 上，恒定 LR + 20% cooldown 与最优余弦**几乎完全打平**，cooldown 的最优峰值 LR 还略低一些；cooldown 的起点可以从训练中途的任意 checkpoint 取，所以 scaling 实验的算力被压下来。他们也明确写了 rewarm 的代价：LR 重新拉高会出现尖峰，而恢复很慢。更早的梯形调度（Zhai et al., 2022, *Scaling Vision Transformers*）是同一思路的 ViT 版本。

## 🏗️ 实验：玩具任务上的三个问题

本机 CPU、numpy、单隐层 tanh MLP（8 维输入、32 个隐藏单元）、4096 个训练样本、Adam、峰值 LR 3e-3、3000 步，目标函数是 4 个正弦/乘积项的叠加（不可约噪声 0.05）。3 个独立种子（数据与初始化都换）。这是**玩具规模**，只回答机制问题，不回答"LLM 上能省多少"：

- E1：同预算 3000 步，余弦 vs WSD 的终值；
- E2：从 WSD 的第 1000 / 2000 步 checkpoint 取权重 + 10% 步数的 cooldown（100 / 200 步），与"从头跑同样总步数的余弦"比；
- E3：两者各多给 1050 步续训，看谁接得更好。

## 📊 结果

E1 终值（3 个种子，验证 loss）：

| 调度 | 种子 A | 种子 B | 种子 C | 均值 |
|---|---|---|---|---|
| 余弦 3000 步 | 0.0207 | 0.0334 | 0.0216 | **0.0252** |
| WSD 3000 步 | 0.0134 | 0.0142 | 0.0123 | **0.0133** |

WSD 低 47%，但别急着外推：这个差距主要来自 LR 预算的分配，余弦从第 100 步就开始掉，全程折半等于少跑了一大截高 LR 的步。第 1000 步时两者打平（训练 loss 0.0373 vs 0.0359），差距是在 1000–3000 这段拉开的（0.0207 → 0.0149）。论文在 210M/真实语料上是**打平**，这里 WSD 赢，说明玩具规模的结论只能当机制演示。

E2 才是这篇的重点：

| 起点 | 补的步数 | 补完验证 loss | 从头跑同总步数余弦 |
|---|---|---|---|
| WSD 1000 步 checkpoint | 100（+3.3%） | 0.0407 | 0.1006（1100 步） |
| WSD 2000 步 checkpoint | 200（+6.7%） | **0.0197** | 0.0451（2200 步） |

（两个数字都是 3 个种子的均值。）要拿到 loss-算力曲线上的两个点：WSD 是"一条 3000 步的 run + 300 步退火 = 3300 步"；余弦是三条独立 run（1100 + 2200 + 3000）= 6300 步。差 1.9 倍，而且 WSD 那条 run 的 checkpoint 还能继续退火出更多点。

E3 续训（各 1050 步）：余弦终点要先 warmup 回峰值再跑完一个余弦周期，0.0252 → 0.0181；WSD 从 2000 步稳定段直接全速跑 750 步再退火 300 步，0.0197 → 0.0140。两者都还能降，WSD 一侧低 23%。**这个规模上没有观察到 rewarm 的 loss 尖峰**——最大 loss 就等于续训第一步的值，因为模型太小、Adam 的二阶矩还保留着。论文在 210M 参数上观察到的尖峰在这里不成立，别拿玩具反证论文。

## 💻 最小可复现

核心只有两个函数：调度和带 10% cooldown 的分支训练。

```python
def schedule(name, total, lr_max=3e-3, warmup=100, decay_frac=0.2):
    stable_end = total - int(total * decay_frac)
    def lr(t):
        if t < warmup:
            return lr_max * t / warmup
        if name == 'cosine':
            return lr_max * 0.5 * (1 + math.cos(math.pi * (t - warmup) / (total - warmup)))
        if t < stable_end:                      # WSD：稳定段恒为峰值
            return lr_max
        return lr_max * max(0.0, 1 - (t - stable_end) / (total - stable_end))
    return lr

def branch_cooldown(params, m, v, ckpt_step, decay_frac=0.1):
    dec = int(ckpt_step * decay_frac)
    total = ckpt_step + dec                    # 总步数只影响分母，不影响已走过的稳定段
    sched = schedule('wsd', total, warmup=1, decay_frac=dec / total)
    return train(sched, dec, p=params, m=m, v=v, bc0=ckpt_step)
```

`bc0` 是 Adam 的偏差修正计数器——分支训练常常忘了这一项，续训第一步就会用错 $\hat m$ 和 $\hat v$。完整脚本（含 E1/E2/E3 三组实验、3 个种子的入口）见 [demos/94-wsd-lr-schedule.py](https://github.com/shenlv7/blog/blob/main/demos/94-wsd-lr-schedule.py)。

## 💡 什么时候该换

- **要做 scaling law 或超参扫描**：WSD 是结构性优势，不是一个超参。一条长 run 的多个 checkpoint + 短 cooldown，就把"每个长度重跑一次"变成"跑一次，退火很多次"，论文报的算力节省就来自这里。
- **预训练会被续**（加数据、加训练步、继续 mid-training）：WSD 的 checkpoint 是未退火的原始权重，续训接得上；余弦的终点 LR 已经是 0，续训必须 rewarm，而 rewarm 的代价随规模上升。
- **只跑一次、预算固定、不打算复用**：余弦没有任何劣势，反而少一个需要调的 `decay_frac`。换调度的理由应该是"我要复用 checkpoint"，不是"新方法更好"。
- **交付前必须退火**。稳定段的模型是最差的出货形态：LR 还在峰值，权重带着噪声。MiniCPM 论文里那次台阶式下降就发生在 decay 段——省掉 cooldown 等于把最好的一段丢掉。

盲区写在这里：以上数字是单隐层 MLP 上的 3000 步、单任务、单峰值 LR；没有扫 `decay_frac`（论文扫过，20% 附近是常用区间），也没有测优化器换成 Muon 之后的行为。要判断在 7B 模型上的收益，得看论文的规模，不能看这张表。
