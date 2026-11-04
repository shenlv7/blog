---
title: "KV cache 砍到 20%：注意力峰值一个没丢，晚到的那条事实回收率 0.000"
description: "numpy 离散模拟（非 GPU 实测）：2048 token、预算 410(20%) 下，H2O 的峰值 token 留存率 1.000，随机 0.43、滑窗 0.47，但保留的注意力质量只有 0.649；把同一批 token 改成第 200 个 query 才相关，H2O 与滑窗的回收率都掉到 0.000，随机反而是 0.51~0.91。被淘汰的 token 分数被冻结，这条死路是驱逐策略的结构性盲区，不是调参问题。"
pubDate: 2026-11-04
tags: ["KV cache", "H2O", "推理优化", "长上下文", "模拟"]
difficulty: "advanced"
series: "ai-concepts"
seriesOrder: 72
slug: "ai-concepts-72-kv-cache-eviction"
---

> 2048 token 的 KV cache 砍到 20%，H2O 把每个 query 的注意力峰值 token 全部留住，留存率 **1.000**；随机淘汰 0.43，滑窗 0.47。
> 同一批 token 改成"第 200 个 query 才开始相关"之后，H2O 的回收率是 **0.000** —— 三个随机种子全为 0。
> 差别不在算法聪不聪明，而在一条结构性死路：被淘汰的 token 再也挣不到分数。

**读这篇要会**：softmax 与注意力分数的形状（`B × H × T × T`）／decode 阶段 KV cache 为什么随长度线性增长／top-k 选择的语义。

## 🎯 显存账单：谁在随长度线性涨

decode 是逐 token 的：每生成一个 token，都要拿新 token 的 Q 去和**全部历史**的 K 做点积、再加权求和历史的 V。历史 K/V 不能丢，否则计算量从 $O(T)$ 变成 $O(T^2)$ 的重算，所以它们被存下来，叫 KV cache。

它的体积是可以精确算出来的，每个 token：

$$
2 \times L \times H_{kv} \times d_{head} \times \text{bytes}
$$

以 Llama-3-8B 为例：32 层、GQA 下 8 个 KV 头、head_dim 128、fp16 —— 每 token 128 KiB。

| 上下文长度 | KV cache（单请求） |
|---|---|
| 8K | 1 GiB |
| 32K | 4 GiB |
| 128K | 16 GiB |

权重本身才 16 GiB。也就是说 128K 上下文、batch=1 时，KV cache 已经和权重一样大；batch=8 就爆了。

三条路：**不存**（重算 prefill）、**少存**（量化、低秩投影）、**选择性存**（驱逐）。这里只讲第三条，因为它对精度的影响最难从指标上看出来。

## 🔬 机制：heavy hitter 假设与那条死路

H2O 论文（Zhang et al., NeurIPS 2023）的核心观测是：注意力矩阵在推理时 **>95% 稀疏**，累积注意力分数服从幂律 —— 少数 token 吃掉大部分质量，作者叫它们 heavy hitter（H2）。于是驱逐规则可以写成一个固定预算的集合选择：

$$
S_t = \{\text{sink}\} \cup W_{\text{recent}} \cup \operatorname{TopK}_{i \notin W}\left(\sum_{\tau \le t} a_{\tau, i}\right), \quad |S_t| = B
$$

保留集合上的注意力被重新归一化，预测的误差就直接是丢掉的质量：

$$
\hat a_i = \frac{a_i}{\sum_{j \in S_t} a_j}, \qquad \text{err} = 1 - \sum_{i \in S_t} a_i
$$

（$a_{\tau,i}$ 是第 $\tau$ 个 query 分给第 $i$ 个 token 的注意力，$B$ 是预算。）

公式里藏着那条死路：$\sum_{\tau \le t} a_{\tau,i}$ 只累计**真正算过的**注意力。一个 token 一旦被踢出 $S_t$，它的分数就冻结在淘汰那一刻，永远拿不到新的分数 —— 后续再怎么相关，也进不了候选集。这不是实现细节，是"用过去预测未来"这个范式的必然。

## 🏗️ 模拟怎么搭

本节数字来自离散事件模拟，不是 GPU 实测：不模拟真实权重，只模拟注意力分布的**形状**。三条形状假设都取自公开观测——sink 位置恒高、少量 heavy hitter 对多数 query 稳定、近处 token 有局部性偏好——外加逐 query 独立噪声 σ=0.9。

- T=2048，24 个 heavy hitter（logit +4.5），sink +4.0，最近 64 个 token +1.2，其余为噪声
- 预算 B=410（20%）= 1 sink + 128 最近窗口 + 281 个按累积分数选的 token
- 对比：full cache（基线）、H2O、随机淘汰（同样预算）、滑窗（只留最后 410 个）
- 度量：**质量保留** = 保留集合上的 full 注意力质量；**峰值留存** = 该 query 注意力最高的 token 是否还在缓存；**峰值概率占比** = 驱逐后峰值 token 的概率 / full 里的概率

先跑满缓存观察 8 个 query 再开始淘汰 —— 真实系统也要先 prefill 再谈驱逐，否则一开始就没有分数可依据。

## 💻 代码

```python
import numpy as np

T, DV, NQ, BUD = 2048, 32, 256, 410

def build_logits(rng, nq=NQ, t=T, hh=24):
    hh_idx = rng.choice(np.arange(1, t), size=hh, replace=False)
    base = 0.9 * rng.normal(size=(nq, t))
    base[:, 0] += 4.0                    # attention sink
    base[:, hh_idx] += 4.5               # heavy hitters
    base[:, -64:] += 1.2                 # 局部性
    return base, hh_idx

def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()

def run(nq=NQ, budget=BUD, seed=0, delayed=False):
    rng = np.random.default_rng(seed)
    L, hh = build_logits(rng, nq=nq)
    recent, n_keep = 128, budget - 128 - 1
    h2o = set(range(T))                  # 先满缓存观察，再淘汰
    rnd = {0} | set(range(T - recent, T)) | set(
        rng.choice(np.arange(1, T - recent), size=n_keep, replace=False).tolist())
    win = set(range(T - budget, T))
    pol = {'h2o': h2o, 'random': rnd, 'window': win}
    acc = {k: dict(mass=[], needle=[], ppeak=[]) for k in ('full', *pol)}
    got, want, scores = {}, {}, np.zeros(T)
    late = np.arange(1000, 1008)
    for q in range(nq):
        lg = L[q].copy()
        if delayed:
            lg[late] += -8.0 if q < 200 else 6.0   # 前 200 个 query 完全无关，之后突然相关
        a_full = softmax(lg)
        peak, p_full = int(a_full.argmax()), a_full.max()
        acc['full']['mass'].append(1.0); acc['full']['needle'].append(1.0)
        acc['full']['ppeak'].append(1.0)
        if delayed and q >= 200:
            m = float(a_full[late].sum()); got['full'] = got.get('full', 0) + m
            want['full'] = want.get('full', 0) + m
        for name, s in pol.items():
            keep = np.zeros(T, bool); keep[list(s)] = True
            a = softmax(np.where(keep, lg, -np.inf))
            acc[name]['mass'].append(float(a_full[keep].sum()))
            acc[name]['needle'].append(float(keep[peak]))
            acc[name]['ppeak'].append(float(a[peak] if keep[peak] else 0) / p_full)
            if name == 'h2o':
                a_h2o = a
            if delayed and q >= 200:
                got[name] = got.get(name, 0.0) + float(a[late].sum())
                want[name] = want.get(name, 0.0) + float(a_full[late].sum())
        scores += a_h2o        # 只累计"看得见"的注意力：被淘汰的 token 拿不到新分
        if q >= 8:
            h2o = {0} | set(range(T - recent, T))
            h2o |= set((np.argsort(-scores[1:T - recent])[:n_keep] + 1).tolist())
            pol['h2o'] = h2o
    return {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}

if __name__ == '__main__':
    for seed in (0, 1, 2):
        r = run(seed=seed)
        for name in ('full', 'h2o', 'random', 'window'):
            d = r[name]
            print(f"seed={seed} {name:6s} 质量保留={d['mass']:.3f} "
                  f"峰值留存={d['needle']:.3f} 峰值概率占比={d['ppeak']:.3f}")
```

真实输出（本机 numpy CPU，约 2.5 秒跑完；下面代码可直接粘贴运行）：

```
seed=0 full   质量保留=1.000 峰值留存=1.000 峰值概率占比=1.000
seed=0 h2o    质量保留=0.649 峰值留存=1.000 峰值概率占比=1.561
seed=0 random 质量保留=0.272 峰值留存=0.430 峰值概率占比=1.401
seed=0 window 质量保留=0.302 峰值留存=0.465 峰值概率占比=1.383
seed=1 full   质量保留=1.000 峰值留存=1.000 峰值概率占比=1.000
seed=1 h2o    质量保留=0.648 峰值留存=1.000 峰值概率占比=1.561
seed=1 random 质量保留=0.240 峰值留存=0.395 峰值概率占比=1.409
seed=1 window 质量保留=0.301 峰值留存=0.488 峰值概率占比=1.447
seed=2 full   质量保留=1.000 峰值留存=1.000 峰值概率占比=1.000
seed=2 h2o    质量保留=0.655 峰值留存=1.000 峰值概率占比=1.545
seed=2 random 质量保留=0.298 峰值留存=0.488 峰值概率占比=1.470
seed=2 window 质量保留=0.267 峰值留存=0.453 峰值概率占比=1.500
```

## 📊 三个数字，三种含义

| 策略 | 质量保留 | 峰值留存 | 峰值概率占比 |
|---|---|---|---|
| full | 1.000 | 1.000 | 1.000 |
| H2O | 0.648 ~ 0.655 | **1.000** | 1.55 |
| 随机 | 0.240 ~ 0.298 | 0.395 ~ 0.488 | 1.40 |
| 滑窗 | 0.267 ~ 0.302 | 0.453 ~ 0.488 | 1.44 |

**第一，峰值留存 1.000 是 H2O 的真本事。** 三个种子、256 个 query，没有一次把注意力最高的那个 token 淘汰掉。随机淘汰和滑窗只有 0.43~0.49，因为这个任务里 heavy hitter 的位置是随机的，滑窗只保证"近的"，保证不了"对的"。

**第二，质量保留 0.65 听着不高，但丢掉的基本是噪声。** 被淘汰的 1638 个 token 合起来只占 35% 的质量，且没有一个在任何一个 query 里当过峰值。驱逐的代价高度不对称：删掉长尾几乎无损，删错一个 heavy hitter 就直接掉点（论文里 mask 掉 H2 后准确率骤降）。

**第三，峰值概率占比 1.55 是个陷阱。** 重新归一化之后，剩下的 token 分摊 100% 的概率质量，于是峰值 token 的相对权重被放大了约 55%。如果只盯输出的 top-1 token 是否变化，看到的是"没掉点"；但它背后的分布已经变了 —— 这也是为什么驱逐类方法的评测要报困惑度或长文任务准确率，而不是只看 top-1。

## 🕳️ 延迟相关：回收率 0.000

把 token 1000..1007 设成前 200 个 query 完全不相关（logit −8），第 200 个之后突然变成关键（logit +6）：

```
seed=0: full 1.000  h2o 0.000  random 0.526  window 0.000
seed=1: full 1.000  h2o 0.000  random 0.511  window 0.000
seed=2: full 1.000  h2o 0.000  random 0.912  window 0.000
```

H2O 和滑窗的回收率是 0.000，随机反而是 0.51~0.91 —— 因为随机淘汰**碰巧**留下了其中一部分，而按分数淘汰是**系统性地**把它们全删了：前 200 个 query 里它们一分未得，正好排在队尾。

这就是把"过去被关注"当成"未来会被需要"的代价。对话里它对应一种很常见的形态：用户在第 3 轮提到的一条约束，第 40 轮才需要；文档问答里对应"问题在第 5 页，答案在第 1 页但那段当时看起来像目录"。检索增强能绕开一部分，但如果模型必须在同一个上下文里连续推理，缓存里已经没有那段 K/V 了。

H2O 论文自己也没回避这件事：20% 预算在 zero/one-shot 任务上基本无损、显存降到 1/5，但 5-shot 任务需要 **30~40%** 的预算才不掉点 —— 上下文越长、越需要"翻旧账"，驱逐就越贵。

## 💡 工程上能做什么

1. **分两条队列，别用一个池子。** 分数驱逐只会奖励"已被证明重要"的 token，因此要给"尚未被证明、但结构上可能重要"的 token 一条保护通道：系统提示、角色设定、用户明确说"记住"的内容、检索命中的段落，直接钉在保护集里，不参与淘汰。
2. **分层而不是删除。** HBM 里放 H2 + 窗口，被淘汰的 K/V 下沉到 CPU 内存或 NVMe，命中时再拉回来。多一次 PCIe 传输比永久丢失信息便宜 —— 尤其在 agent 场景里，"以后会不会用到"几乎无法预测。
3. **需要时重算，而不是假装没丢。** 驱逐后的正确性无法靠策略保证，只能靠重放：把被淘汰的 chunk 重新 prefill 一遍（配合 prefix caching，代价是重算而不是全量重算）。
4. **评测要换指标。** top-1 一致率对驱逐几乎不敏感（首 token 往往被 sink 和最近窗口锁定），用长文任务的准确率、关键信息召回、或者对"远端事实"的定向探查来评。

一句话收束：KV cache 驱逐省的是显存，押上的是**时间维度上的可预测性**；压在 20% 上的系统，最好知道自己在赌哪一段未来不会被翻出来。

完整脚本（含对比策略与延迟相关实验）：[demos/72-kv-cache-eviction.py](https://github.com/shenlv7/blog/blob/main/demos/72-kv-cache-eviction.py)
