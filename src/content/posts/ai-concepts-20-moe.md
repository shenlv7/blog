---
title: "20 | 专家混合 MoE：671B 参数只算 37B——我把 4 个专家塞进 numpy，看它们怎么分工"
description: "稠密模型每次思考都要动用全部参数，MoE 改成了「专业的事交给专业的人」：DeepSeek-V3 有 671B 参数却每 token 只算 37B。本文用 100 行 numpy 从零实现 top-1 门控 MoE，实测专家专业化、专家坍缩（一个专家被饿死到 0%）和线性门控切不开圆周三个现象，并给出 Mixtral/DeepSeek-V3/Llama 4 的真实参数账。"
pubDate: 2026-09-13
tags: ["AI核心概念", "MoE", "专家混合", "稀疏激活", "路由", "Transformer"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 20
slug: ai-concepts-20-moe
---

# AI核心概念(20)：专家混合 MoE——671B 参数只算 37B，专家是怎么分工的

> "稠密模型的信条是「每一份知识都参与每一次思考」；MoE 的信条是「专业的事交给专业的人」。诱人的是后者省算力，麻烦的是——你得先学会怎么把活分均匀。"

## 🎯 什么是 MoE？

**专家混合（Mixture of Experts, MoE）**：把一个稠密的 FFN（前馈网络）层拆成 N 个并行的子网络——每个叫一个**专家（expert）**——再配一个**门控网络（router / gate）**。每个 token 进来，门控只挑 **top-k** 个专家干活，其余专家这次不参与计算。

关键在于：**"总参数量"和"每个 token 的算力"从此变成两个独立的旋钮。**

| | 稠密模型 | MoE |
|---|---|---|
| 每 token 参与计算的参数 | 全部 | 只有 top-k 个专家 |
| 知识容量 | ∝ 全部参数 | ∝ 全部参数（专家越多容量越大） |
| 每 token 算力（FLOPs） | ∝ 全部参数 | ∝ 激活参数（k 决定，与专家总数无关） |
| 显存占用 | ∝ 全部参数 | ∝ 全部参数（**省的是算力，不是显存**） |
| 训练稳定性 | 高 | 低（路由会抖、专家会偏） |

真实模型的账（都是官方公开数据，不是我估算的）：

| 模型 | 专家配置 | 总参数 | 每 token 激活 |
|---|---|---|---|
| Mixtral 8x7B（Mistral，2023-12） | 8 选 **2** | 46.7B | 12.9B |
| DeepSeek-V3（2024-12，arXiv:2412.19437） | 256 选 **8**，外加 1 个「共享专家」 | 671B | 37B |
| Llama 4 Maverick（Meta，2025-04） | 128 个专家 | 400B | 17B |

注意两件事。第一，**MoE 只替换 Transformer 里的 FFN 子层**，注意力部分（以及所有 LayerNorm、残差）仍然是稠密的——省下来的是 FFN 那一大块。第二，**省算力不省显存**：Mixtral 官方明确说过，部署时内存开销按 47B 算，但算力和延迟按 13B 算。你可以理解成"请了一支 47 人的专家团队待命，但每次会议只叫 2 个人进会议室"——**人得养着，只是不让他们同时说话。**

## 🔬 为什么它既是好主意，又是麻烦制造机

**好在哪：容量的成本被摊薄了。** 语言模型里大量的 FFN 参数其实是"知识存储"。稠密模型想让知识翻倍，算力就得翻倍；MoE 想让知识翻倍，只需要多摆几个专家，每 token 算力几乎不变。这就是 DeepSeek-V3 能用 671B 的体量、37B 的算力开销去对标同级稠密模型的原因。

**麻烦在哪：路由是"学"出来的，不是"规定"的。** 门控网络自己也不知道哪个 token 该给谁——它靠梯度学。于是就有一个正反馈陷阱：

1. 训练初期某个专家碰巧对某批 token 给了略好的输出
2. 门控于是更倾向把这类 token 给它 → 它拿到更多梯度 → 变得更强
3. 其余专家拿不到 token、拿不到梯度、停止进化

这叫**专家坍缩（expert collapse）**，或者更直白：**有专家被饿死。** 参数表里那一行权重冻结在原地，白占显存、毫无贡献。

标准解法是加一项**负载均衡损失（load balancing loss，也称辅助损失 auxiliary loss）**：统计每个专家分到的 token 比例，惩罚"有人吃太饱、有人吃不到"的分布。训练目标就从单纯的预测误差，变成"预测误差 + λ×不均衡度"。

有意思的是，**DeepSeek-V3 后来把这条路走反了一次**：它不用辅助损失，改成给每个专家维护一个动态**偏置项**，谁分到的 token 太多就悄悄压低它的路由分数。官方称之为 auxiliary-loss-free 的均衡策略——好处是不再和主任务抢梯度。另外它还留了一个**共享专家（shared expert）**：不管路由结果如何都参与每个 token 的计算，兜住通用能力，让其余 256 个专家专心去分化。**门控负责分工，共享专家负责兜底**，这个设计思路值得记住。

## 🧪 本机实测：4 个簇、4 个专家、4 秒跑完

我造了一个最小可验证的任务：输入是 4 维，样本分成 4 个簇（摆在四维立方体的四个角上），**每个簇有一条自己的线性规则 y = w_c·x**。单层线性模型注定学不好（它只有一条直线），但一个会分工的 MoE 应该能学会"把每个簇路由给它自己的那个专家"。

本机 numpy 实跑，控制变量四组，全部用固定随机种子，可复现：

| 实验 | 测试 MSE | 专家使用率 | 专家是否各司其职 |
|---|---|---|---|
| 稠密单专家（基线） | **0.1633** | — | 一条直线拟合不了 4 条规则（应该的） |
| MoE top-1 **+** 负载均衡损失 | **0.0013** | E0 28.7% / E1 21.9% / E2 21.2% / E3 28.1% | ✅ 4 个簇 → 4 个不同专家，完美 1:1 |
| MoE top-1 **无** 负载均衡损失 | 0.0093 | **E0 0.0%** / E1 21.2% / E2 50.0% / E3 28.7% | ❌ E0 被完全饿死，两个簇挤在同一个专家身上 |
| 同样 4 个簇，改摆到**圆周**上 | 0.7110 | E0 23.1% / E1 26.2% / E2 26.2% / E3 24.4% | ❌ 有簇被迫共用专家，误差暴涨 500 倍 |

三条结论，全部来自真实输出：

1. **专家真的会专业化。** 开了均衡损失之后，4 个簇中心被路由到 4 个**互不相同**的专家，测试误差 0.0013——比稠密基线低了两个数量级。
2. **关掉均衡损失，专家就坍缩。** E0 的使用率精确地掉到 **0.0%**：它一次都没被选中，一次梯度都没拿到，彻底饿死。同时 E2 吃掉了 50% 的流量、被迫顶起两个簇的活，误差反而升到 0.0093（是上一组的 7 倍）。**这就是"参数量还在，有效容量已经没了"的样子。**
3. **门控自己太弱，分发就是乱分。** 把同样的 4 个簇从"立方体四个角"挪到"圆周上"，线性门控（本质是 X·Wg 一次线性打分）切不开圆周这类分布，专业化立刻退化，误差从 0.0013 涨到 0.7110。真实模型的门控之所以能分好，靠的是它吃的是深层已经编码好的表示，而不是原始特征。

## 💻 动手 Demo：100 行 numpy 实现 top-1 门控 MoE

```python
"""02-moe.py — 纯 numpy 最小 MoE，CPU 约 4 秒。含反向传播与 Adam。"""
import numpy as np

DIM, K, N_PER = 4, 4, 200           # 输入维度 / 专家数 / 每簇样本数


def hypercube(K=K, seed=7):
    """4 个簇放在四维立方体的 4 个角上：任意两类之间都线性可分。"""
    corners = np.array([[2.5, 2.5, 2.5, 2.5],
                        [2.5, -2.5, 2.5, -2.5],
                        [-2.5, 2.5, -2.5, 2.5],
                        [-2.5, -2.5, -2.5, -2.5]])[:K]
    rng = np.random.default_rng(seed)
    rules = rng.normal(size=(K, DIM))  # 每个簇一条自己的规则（这才是要学的东西）
    c = np.repeat(np.arange(K), N_PER)
    X = corners[c] + rng.normal(scale=0.35, size=(len(c), DIM))
    y = (X * rules[c]).sum(axis=1, keepdims=True)
    idx = rng.permutation(len(X))      # 打散：否则 8:2 切分可能整块缺一个簇
    return X[idx], y[idx], c[idx], corners


def circle(K=K, seed=7):
    """同样的 4 个簇，但摆在圆周上：任意两类之间线性不可分。"""
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    corners = np.stack([3 * np.cos(ang), 3 * np.sin(ang),
                        -3 * np.cos(ang), -3 * np.sin(ang)], axis=1)
    rng = np.random.default_rng(seed)
    rules = rng.normal(size=(K, DIM))
    c = np.repeat(np.arange(K), N_PER)
    X = corners[c] + rng.normal(scale=0.35, size=(len(c), DIM))
    y = (X * rules[c]).sum(axis=1, keepdims=True)
    idx = rng.permutation(len(X))
    return X[idx], y[idx], c[idx], corners


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MoE:
    """top-k 门控 + 线性专家。真实模型里专家是 FFN，门控是 Linear→Softmax。"""

    def __init__(self, K=K, k=1, aux=0.01, seed=1):
        g = np.random.default_rng(seed)
        self.Wg = g.normal(scale=0.1, size=(DIM, K))   # 门控网络（router）
        self.We = g.normal(scale=0.1, size=(K, DIM))   # K 个专家，各管一片
        self.be = np.zeros(K)
        self.k, self.E, self.aux = k, K, aux
        self.st = {n: (np.zeros_like(p), np.zeros_like(p)) for n, p in self.params().items()}
        self.t = 0

    def params(self):
        return {"Wg": self.Wg, "We": self.We, "be": self.be}

    def forward(self, X):
        probs = softmax(X @ self.Wg)                       # 每个 token 对每个专家的亲和度
        order = np.argsort(-probs, axis=1)[:, :self.k]     # 挑出 top-k
        mask = np.zeros_like(probs)
        for j in range(self.k):
            mask[np.arange(len(X)), order[:, j]] = 1.0
        gate = probs * mask
        gate = gate / gate.sum(axis=1, keepdims=True)      # 归一化：被选中的专家按权重加和
        eo = X @ self.We.T + self.be                       # 教学写法：K 个专家全算（工程上只算选中的）
        return (gate * eo).sum(axis=1, keepdims=True), probs, gate, mask

    def step(self, X, y, lr=0.05):
        n = len(X)
        pred, probs, gate, mask = self.forward(X)
        dpred = 2.0 * (pred - y) / n                       # dMSE/dpred
        eo = X @ self.We.T + self.be
        dgate = dpred * eo
        dexpert = dpred * gate
        dWe, dbe = dexpert.T @ X, dexpert.sum(axis=0)
        gsum = gate.sum(axis=1, keepdims=True)
        dgate_n = (dgate - (dgate * gate).sum(axis=1, keepdims=True) / gsum) / gsum
        dprobs = dgate_n * mask                            # 只有被选中的专家拿到梯度
        if self.aux:                                       # 负载均衡损失（对 probs 求导，f 视为常数）
            f = mask.mean(axis=0)
            dprobs += self.aux * self.E * f[None, :] / n
        dz = probs * (dprobs - (dprobs * probs).sum(axis=1, keepdims=True))   # softmax 反向
        grads = {"Wg": X.T @ dz, "We": dWe, "be": dbe}
        self.t += 1
        for name, p in self.params().items():              # Adam
            m, v = self.st[name]
            m *= 0.9; m += 0.1 * grads[name]
            v *= 0.999; v += 0.001 * grads[name] ** 2
            p -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)
        return float(np.mean((pred - y) ** 2))


def train(data, aux=0.01, k=1, steps=800, seed=1):
    X, y, _, corners = data
    ntr = int(len(X) * 0.8)
    m = MoE(k=k, aux=aux, seed=seed)
    for _ in range(steps):
        m.step(X[:ntr], y[:ntr])
    p, probs, _, mask = m.forward(X[ntr:])
    _, cprobs, _, _ = m.forward(corners)                   # 拿 4 个簇中心问一次路由
    return m, float(np.mean((p - y[ntr:]) ** 2)), mask.mean(axis=0), np.argmax(cprobs, axis=1)


def dense_baseline(data):
    X, y, _, _ = data
    ntr = int(len(X) * 0.8)
    w = np.linalg.lstsq(np.c_[X[:ntr], np.ones(ntr)], y[:ntr], rcond=None)[0]
    return float(np.mean((np.c_[X[ntr:], np.ones(len(X) - ntr)] @ w - y[ntr:]) ** 2))


def report(tag, data, aux):
    m, mse, usage, top1 = train(data, aux=aux)
    print(f"{tag}\n  测试 MSE {mse:.4f} | 专家使用率 " +
          " ".join(f"E{i}:{u:5.1%}" for i, u in enumerate(usage)) +
          f"\n  4 个簇中心 → 专家 {top1.tolist()}（共用到 {len(set(top1.tolist()))} 个专家）")
    return mse


if __name__ == "__main__":
    hc, ci = hypercube(), circle()
    print("=== 稠密单专家（同输入）基线 ===")
    print(f"  测试 MSE {dense_baseline(hc):.4f}  ← 一条直线拟合不了 4 条不同规则")
    print("\n=== 实验 A：4 个簇摆成立方体四个角（线性可分）===")
    report("  开了负载均衡损失 aux=0.01", hc, 0.01)
    report("  关掉负载均衡损失 aux=0", hc, 0.0)
    print("\n=== 实验 B：同样 4 个簇，但摆在圆周上（线性不可分）===")
    report("  开了负载均衡损失 aux=0.01", ci, 0.01)
```

实测输出（本机原样贴，`python3 02-moe.py`，约 4 秒）：

```text
=== 稠密单专家（同输入）基线 ===
  测试 MSE 0.1633  ← 一条直线拟合不了 4 条不同规则

=== 实验 A：4 个簇摆成立方体四个角（线性可分）===
  开了负载均衡损失 aux=0.01
  测试 MSE 0.0013 | 专家使用率 E0:28.7% E1:21.9% E2:21.2% E3:28.1%
  4 个簇中心 → 专家 [2, 3, 0, 1]（共用到 4 个专家）
  关掉负载均衡损失 aux=0
  测试 MSE 0.0093 | 专家使用率 E0: 0.0% E1:21.2% E2:50.0% E3:28.7%
  4 个簇中心 → 专家 [1, 2, 3, 2]（共用到 3 个专家）

=== 实验 B：同样 4 个簇，但摆在圆周上（线性不可分）===
  开了负载均衡损失 aux=0.01
  测试 MSE 0.7110 | 专家使用率 E0:23.1% E1:26.2% E2:26.2% E3:24.4%
  4 个簇中心 → 专家 [0, 1, 2, 0]（共用到 3 个专家）
```

代码里有几个地方是"教学特化"的，看真实实现时要知道差别：

- **`eo = X @ self.We.T` 把 K 个专家全算了。** 真跑起来是 gather/scatter：只把 token 送到被选中的专家所在的卡上。我这么写是为了让反向传播在 30 行内看得清——**稀疏的部分在"哪些专家拿梯度"，不在"我算了几个矩阵乘"。**
- **门控梯度是稀疏的。** `dprobs = dgate_n * mask` 这一行就是 MoE 省算力的本质：没被选中的专家，这一步梯度恒为 0，参数一动不动。
- **专家是线性的**，真实模型里是一个 SwiGLU FFN；**top-k 我用的是 top-1**，真实模型普遍用 top-2（Mixtral）甚至 top-8（DeepSeek-V3）。

## 🏗️ 工程上真正的难点

| 难点 | 现象 | 常见做法 |
|---|---|---|
| **路由不均衡** | 少数专家吃满算力，其余 GPU 空转；吞吐被最忙的那个卡卡住 | 负载均衡损失、偏置项动态调整（DeepSeek-V3 走的就是这条）、容量因子 + token 丢弃 |
| **通信开销** | 专家分散在多卡上（专家并行），每个 MoE 层都要做 all-to-all 的收发 | 把专家按层分组、通信与计算重叠、只在 MoE 层做并行 |
| **训练不稳** | 路由抖动、loss spike | 小学习率预热、门控加噪、路由用更平滑的分数（如 sigmoid） |
| **显存不省** | 总参数全得装进显存/内存，才能保证任意 token 都能路由到任意专家 | 量化、专家卸载到 CPU、按热度分布专家 |
| **推理实现** | 一个 batch 里不同 token 去不同专家，尺寸不齐，GPU 利用率低 | 专用稀疏算子（Megablocks 之类）、continuous batching、把 expert 当成大稀疏矩阵乘 |

一句话总结工程取舍：**MoE 用显存和通信复杂度，换来了"同等算力下更大的知识容量"。** 反过来说，如果你的部署环境显存紧、batch 小、单卡跑——MoE 往往不划算，这也是为什么它主要出现在旗舰级模型上。

## 💡 小结

- MoE 把 FFN 拆成 N 个专家 + 一个门控，每个 token 只走 top-k 个：**总参数决定容量（也是显存），激活参数决定算力**，两者从此解耦
- 数字感受一下：Mixtral 8x7B = 8 选 2，46.7B 总 / 12.9B 激活；DeepSeek-V3 = 256 选 8 加 1 个共享专家，671B 总 / 37B 激活
- 本机 100 行 numpy 实测：开了均衡损失，4 个簇干净地分给 4 个专家（MSE 0.0013）；**关掉它，一个专家使用率精确地掉到 0.0%——被饿死，另有一个吃掉 50% 流量，误差涨 7 倍**
- 同样的 4 个簇摆到圆周上，线性门控就切不开，MSE 从 0.0013 涨到 0.7110：**路由质量的上限，取决于门控看到的表示有多可分**
- 心法：**MoE 不是"更大"，是"分工 + 分配机制"**——参数量翻十倍很便宜，把活分均匀才是那个难了一个数量级的活 😵

下次看到"某某模型 1000B 参数"的宣传，先问一句：**每 token 激活多少？** 一千亿参数听起来像天文数字，激活 30B 的话，跑起来其实是个 30B 的账单。 🧮

---

_概念卡片持续更新中，下一个概念见~_ 🔫
