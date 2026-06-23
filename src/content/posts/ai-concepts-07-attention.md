---
title: "07 | Attention：Transformer的灵魂机制"
description: "自注意力机制让每个Token都能'看到'其他Token并计算关联度。这是Transformer之所以强大的核心原因。"
pubDate: 2026-06-25
tags: ["AI核心概念", "Attention", "Transformer", "架构"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 7
---

## 一句话说清楚

> Attention（自注意力）让模型在处理每个词时，能"看到"句子中所有其他词，并计算它们之间的关联程度。

---

## 它是什么

考虑这句话：

```
"猫坐在垫子上，因为它是软的"
```

`"它"` 指的是什么？是 `"猫"` 还是 `"垫子"`？

人类一眼就知道是 `"垫子"`（软的是垫子的属性）。但模型怎么知道？

**Attention 的作用就是：让每个词和其他所有词建立关联，计算出"谁和谁更相关"。**

```
"它" 对每个词的注意力权重:
  猫:    0.15
  坐:    0.05
  在:    0.03
  垫子:  0.62  ← 最高！模型学到了"它"指的是"垫子"
  上:    0.02
  因为:  0.05
  是:    0.03
  软的:  0.05
```

---

## 它是怎么工作的

### Q、K、V 三件套

Attention 的核心是三个向量：**Query（查询）、Key（键）、Value（值）**

```
每个Token都生成三个向量:
  Q（我想找什么？）
  K（我能提供什么？）
  V（我的实际内容是什么？）
```

**类比：图书馆查书**
- **Q** = 你的搜索关键词
- **K** = 每本书的索引标签
- **V** = 书的实际内容

你用 Q 去匹配每本书的 K，找到最相关的书，然后读取它们的 V。

### 计算过程

```
输入: 3个Token的向量 X = [x1, x2, x3]

① 线性变换生成 Q, K, V:
   Q = X × W_Q    (每个Token生成一个Query向量)
   K = X × W_K    (每个Token生成一个Key向量)
   V = X × W_V    (每个Token生成一个Value向量)

② 计算注意力分数:
   scores = Q × K^T / √d_k
   (Q和K的点积，除以√d_k做缩放)

③ Softmax归一化:
   weights = softmax(scores)
   (把分数变成0-1之间的概率)

④ 加权求和:
   output = weights × V
   (用注意力权重对Value加权求和)
```

### 数值示例

```
假设句子: "AI 改变 世界"
每个Token的Q, K, V是2维向量

Q = [[1, 0], [0, 1], [1, 1]]  (3个Token的Query)
K = [[1, 0], [0, 1], [0.5, 0.5]]  (3个Token的Key)

注意力分数 = Q × K^T:
         AI   改变   世界
AI     [1.0,  0.0,  0.5]
改变   [0.0,  1.0,  0.5]
世界   [1.0,  1.0,  1.0]

Softmax后:
         AI    改变    世界
AI     [0.43,  0.15,  0.42]
改变   [0.15,  0.43,  0.42]
世界   [0.33,  0.33,  0.33]

→ "AI" 最关注自己和"世界"
→ "改变" 最关注自己和"世界"
→ "世界" 平均关注所有词
```

---

## 动手试试

### 实验 1：用 PyTorch 手写 Self-Attention

```python
import torch
import torch.nn.functional as F

def self_attention(X, d_k=2):
    """
    X: 输入矩阵 (seq_len, d_model)
    d_k: Key的维度
    """
    seq_len, d_model = X.shape

    # 随机初始化权重矩阵（实际训练中这些是学出来的）
    torch.manual_seed(42)
    W_Q = torch.randn(d_model, d_k)
    W_K = torch.randn(d_model, d_k)
    W_V = torch.randn(d_model, d_k)

    # ① 生成 Q, K, V
    Q = X @ W_Q   # (seq_len, d_k)
    K = X @ W_K   # (seq_len, d_k)
    V = X @ W_V   # (seq_len, d_k)

    # ② 计算注意力分数
    scores = Q @ K.T / (d_k ** 0.5)  # (seq_len, seq_len)

    # ③ Softmax归一化
    weights = F.softmax(scores, dim=-1)

    # ④ 加权求和
    output = weights @ V  # (seq_len, d_k)

    return output, weights

# 模拟3个Token的输入（每个Token是4维向量）
X = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # "AI"
    [0.0, 1.0, 0.0, 1.0],  # "改变"
    [1.0, 1.0, 1.0, 1.0],  # "世界"
])

output, weights = self_attention(X, d_k=2)

tokens = ["AI", "改变", "世界"]
print("注意力权重矩阵:\n")
print(f"{'':8s}", end="")
for t in tokens:
    print(f"{t:>8s}", end="")
print()

for i, t1 in enumerate(tokens):
    print(f"{t1:8s}", end="")
    for j in range(len(tokens)):
        print(f"{weights[i][j].item():8.3f}", end="")
    print()

print(f"\n输出向量:\n{output}")
```

### 实验 2：可视化 Attention 权重（需要 matplotlib）

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟一句话的注意力权重
tokens = ["The", "cat", "sat", "on", "the", "mat", "."]

# 模拟注意力权重（实际应该从模型获取）
np.random.seed(42)
weights = np.random.dirichlet(np.ones(7), size=7)

# 让"cat"和"mat"有更强的关联
weights[1] = [0.05, 0.35, 0.10, 0.05, 0.05, 0.35, 0.05]
weights[5] = [0.05, 0.30, 0.10, 0.05, 0.05, 0.40, 0.05]

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(weights, cmap='Blues')

ax.set_xticks(range(len(tokens)))
ax.set_yticks(range(len(tokens)))
ax.set_xticklabels(tokens, fontsize=12)
ax.set_yticklabels(tokens, fontsize=12)
ax.set_xlabel("Key (被关注的词)", fontsize=12)
ax.set_ylabel("Query (关注的词)", fontsize=12)

# 在每个格子里显示数值
for i in range(len(tokens)):
    for j in range(len(tokens)):
        text = f"{weights[i][j]:.2f}"
        color = "white" if weights[i][j] > 0.2 else "black"
        ax.text(j, i, text, ha="center", va="center", color=color, fontsize=9)

plt.title("Self-Attention 权重可视化", fontsize=14)
plt.colorbar(im, label="注意力权重")
plt.tight_layout()
plt.savefig("attention_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
```

### 实验 3：从模型中提取真实 Attention 权重

```python
import openai

client = openai.OpenAI()

# 注意：大多数API不直接返回attention权重
# 但可以通过一些技巧观察模型的"关注点"

# 使用logprobs观察模型对每个位置的概率分配
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "The capital of France is"}],
    max_tokens=5,
    logprobs=True,
    top_logprobs=5,
)

print("模型预测:")
for token_info in response.choices[0].logprobs.content:
    print(f"\n生成: '{token_info.token}'")
    print(f"  Top 5 候选:")
    for top in token_info.top_logprobs:
        import math
        prob = math.exp(top.logprob)
        print(f"    '{top.token}': {prob:.3f}")
```

---

## Multi-Head Attention（多头注意力）

实际的 Transformer 不只用一组 Q、K、V，而是用**多组**（通常 8 或 16 组）并行计算：

```
Head 1: 关注语法关系 (主语-谓语)
Head 2: 关注语义关系 (同义词)
Head 3: 关注位置关系 (相邻词)
...
Head 8: 关注其他模式

最终把所有Head的输出拼接起来
```

这就像**从不同角度看同一个句子**，每个 Head 学到不同的关注模式。

---

## Attention 的意义

| 特性 | 传统RNN | Attention |
|------|---------|-----------|
| 长距离依赖 | 困难（信息衰减） | 直接连接，无衰减 |
| 并行计算 | 必须顺序处理 | 全部并行 |
| 计算复杂度 | O(n) 每步 | O(n²) 但可并行 |
| 可解释性 | 黑箱 | 注意力权重可视化 |

**Attention 的核心贡献：** 让模型能直接"看到"任意距离的 Token，解决了 RNN 的长距离依赖问题。

---

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 经典可视化讲解
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 从零实现 GPT

---

> **下一篇预告：** [08 | Transformer：改变一切的架构](/blog/posts/ai-concepts-08-transformer) — 把Attention、FFN、位置编码组装起来，理解GPT的完整架构。
