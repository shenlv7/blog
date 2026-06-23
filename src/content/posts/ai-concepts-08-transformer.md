---
title: "08 | Transformer：改变一切的架构"
description: "从Attention到完整的Transformer，理解GPT、Claude、Llama的共同底层架构。"
pubDate: 2026-06-25
tags: ["AI核心概念", "Transformer", "架构"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 8
---

## 一句话说清楚

> Transformer 是一种用 Attention 机制处理序列数据的神经网络架构。几乎所有大语言模型都基于它。

---

## 它是什么

2017年，Google 发了一篇论文叫《Attention Is All You Need》，提出了 Transformer 架构。

在此之前，处理文本主要靠 RNN（循环神经网络）和 LSTM，它们的问题是：
- 必须一个词一个词顺序处理，无法并行
- 长句子会"忘记"前面的内容

Transformer 的革命性在于：
- **完全基于 Attention**，不需要循环结构
- **可以并行处理所有 Token**，训练速度大幅提升
- **能捕捉任意距离的依赖关系**

```
RNN:  词1 → 词2 → 词3 → ... → 词n   (必须顺序)
Transformer: 词1 ↔ 词2 ↔ 词3 ↔ ... ↔ 词n  (全部互相可见)
```

---

## 它是怎么工作的

### Decoder-only 架构（GPT 系列）

GPT、Claude、Llama 都用的是 **Decoder-only** 架构：

```
输入 Token
    ↓
┌─────────────────────┐
│  Token Embedding     │  把Token ID变成向量
│  + Position Encoding │  加上位置信息
└─────────────────────┘
    ↓
┌─────────────────────┐ × N 层
│  Multi-Head          │
│  Self-Attention      │  让Token互相"看到"
├─────────────────────┤
│  Feed-Forward        │
│  Network (FFN)       │  逐Token的非线性变换
├─────────────────────┤
│  Layer Norm          │  稳定训练
├─────────────────────┤
│  Residual Connection │  残差连接保梯度
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Linear + Softmax    │  输出下一个Token的概率
└─────────────────────┘
    ↓
输出 Token
```

### 每一层在做什么

**1. Multi-Head Self-Attention**
```
让每个Token看到所有其他Token，计算关联度
多组Q/K/V并行，学习不同的关注模式
```

**2. Feed-Forward Network (FFN)**
```
对每个Token独立做非线性变换
两层线性层 + 激活函数（通常是GELU）
作用：把Attention整合的信息做进一步处理
```

**3. Layer Normalization**
```
稳定每一层的输出分布
防止训练过程中数值爆炸或消失
```

**4. Residual Connection**
```
output = LayerNorm(x + SubLayer(x))
让梯度能直接"跳过"复杂层，防止梯度消失
```

### 模型规模

| 模型 | 参数量 | 层数 | 注意力头数 | 隐藏维度 |
|------|--------|------|-----------|---------|
| GPT-2 | 1.5B | 48 | 16 | 1600 |
| Llama 3 8B | 8B | 32 | 32 | 4096 |
| Llama 3 70B | 70B | 80 | 64 | 8192 |
| GPT-4 | ~1.8T (传闻) | ~120 | ~128 | ~12288 |

---

## 动手试试

### 实验 1：手写一个 Mini Transformer Block

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MiniAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # 生成Q, K, V并分头
        Q = self.W_Q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        scores = (Q @ K.transpose(-2, -1)) / (self.d_k ** 0.5)
        weights = F.softmax(scores, dim=-1)
        attention = (weights @ V).transpose(1, 2).contiguous().view(batch, seq_len, d_model)

        return self.W_O(attention), weights

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.attention = MiniAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Attention + Residual + Norm
        attn_out, weights = self.attention(x)
        x = self.norm1(x + attn_out)

        # FFN + Residual + Norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, weights

# 测试
batch_size = 2
seq_len = 10
d_model = 64
n_heads = 4
d_ff = 256

x = torch.randn(batch_size, seq_len, d_model)
block = TransformerBlock(d_model, n_heads, d_ff)
output, weights = block(x)

print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"注意力权重形状: {weights.shape}")
print(f"参数量: {sum(p.numel() for p in block.parameters()):,}")
```

### 实验 2：理解位置编码

```python
import numpy as np
import matplotlib.pyplot as plt

def positional_encoding(seq_len, d_model):
    """正弦位置编码"""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)  # 偶数维度用sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数维度用cos
    return pe

seq_len = 50
d_model = 64
pe = positional_encoding(seq_len, d_model)

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(pe.T, aspect='auto', cmap='RdBu')
ax.set_xlabel("Position (词的位置)")
ax.set_ylabel("Dimension (编码维度)")
ax.set_title("正弦位置编码可视化")
plt.colorbar(im)
plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=150, bbox_inches='tight')
plt.show()

# 验证：相邻位置的编码确实不同
print("位置0和位置1的编码差异:")
diff = np.linalg.norm(pe[0] - pe[1])
print(f"  L2距离: {diff:.4f}")

print("位置0和位置49的编码差异:")
diff = np.linalg.norm(pe[0] - pe[49])
print(f"  L2距离: {diff:.4f}")
```

### 实验 3：GPT-2 参数结构分析

```python
# 需要安装 transformers: pip install transformers
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")

print(f"模型: GPT-2")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"\n各层结构:")

for name, param in model.named_parameters():
    if "weight" in name and param.dim() == 2:
        print(f"  {name:50s} {str(list(param.shape)):>20s}  ({param.numel():>12,} params)")
```

---

## Encoder vs Decoder vs Encoder-Decoder

| 架构 | 代表模型 | 特点 | 适用场景 |
|------|---------|------|---------|
| **Encoder-only** | BERT | 双向注意力，看到全部上下文 | 分类、NER、理解任务 |
| **Decoder-only** | GPT, Claude, Llama | 单向注意力，只看左边 | 文本生成、对话 |
| **Encoder-Decoder** | T5, BART | 编码理解 + 解码生成 | 翻译、摘要 |

**为什么 GPT 选 Decoder-only？**
- 简单、高效、可扩展
- 生成任务只需要看左边的上下文
- 大规模训练时更容易并行

---

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 最佳可视化教程
- [GPT-2 论文](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Llama 3 技术博客](https://ai.meta.com/blog/meta-llama-3/)

---

> **下一篇预告：** [09 | Fine-tuning：花小钱办大事的模型定制](/blog/posts/ai-concepts-09-finetuning) — 不用从头训练，用少量数据让通用模型变成领域专家。
