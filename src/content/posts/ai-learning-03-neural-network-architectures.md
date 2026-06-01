---
title: "AI学习之路(第3期)：神经网络架构"
description: "从全连接到卷积、循环、注意力，一文看懂主流神经网络架构的设计哲学与适用场景"
pubDate: 2026-06-02
tags: ["神经网络", "CNN", "RNN", "LSTM", "架构设计", "AI基础"]
---

![神经网络架构](https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800&h=400&fit=crop)

## 引言

上一期我们从感知机一路走到深度神经网络，搞明白了"深度"的含义和PyTorch实战。但你有没有想过：为什么处理图片要用CNN，处理文本要用RNN，而现在大模型全都换成了Transformer？

答案就在于**架构设计**。不同的网络结构，本质上是对不同类型数据的"归纳偏置"（Inductive Bias）——简单说，就是网络天生擅长处理什么样的问题。

这一期，我们把主流架构拆开来看，搞清楚它们各自的设计哲学。

## 一、全连接网络（FC/DNN）：万金油

全连接层是最基础的结构，每个神经元和上一层所有神经元相连。

```
输入层 → [全连接] → [激活] → [全连接] → [激活] → 输出层
```

**优点**：简单通用，适合结构化数据（表格、特征向量）

**缺点**：参数量爆炸。一张 224×224 的RGB图片展平后有 150,528 个像素，如果第一层有1000个神经元，光这一层就需要 1.5亿 个参数——根本不现实。

**适用场景**：表格数据分类/回归、作为其他网络的末端分类头

```python
import torch.nn as nn

# 一个简单的全连接网络
class FCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x.view(x.size(0), -1))
```

## 二、卷积神经网络（CNN）：图像之王

### 核心思想：局部连接 + 权重共享

CNN的设计灵感来自生物视觉系统。Hubel和Wiesel在1962年发现，猫的视觉皮层中，神经元只对视野中特定小区域的刺激有反应——这就是**感受野**（Receptive Field）的概念。

CNN的三大法宝：

| 组件 | 作用 | 关键参数 |
|------|------|---------|
| 卷积层（Conv） | 提取局部特征 | 卷积核大小、步长、填充 |
| 池化层（Pool） | 降低空间维度 | 池化窗口大小、步长 |
| 全连接层（FC） | 最终分类/回归 | 神经元数量 |

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),    # 3通道→32通道，3×3卷积
            nn.ReLU(),
            nn.MaxPool2d(2),                    # 尺寸减半
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),       # 全局平均池化
        )
        # 分类部分
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
```

### CNN的进化史

![CNN演进](https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800&h=400&fit=crop)

**LeNet-5（1998）**：卷积网络的开山之作，用于手写数字识别

**AlexNet（2012）**：在ImageNet上碾压传统方法，开启了深度学习时代。用了ReLU、Dropout、GPU训练

**VGGNet（2014）**：证明了"深度"的力量——用3×3小卷积核堆叠到16-19层

**ResNet（2015）**：**残差连接**解决了深层网络的梯度消失问题，让网络可以训练到152层甚至更深

```python
# ResNet的核心：残差块
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x                          # 保存输入
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual                       # 残差连接！
        return F.relu(out)
```

**残差连接**的天才之处：让网络学习 `F(x) + x` 而不是直接学习 `F(x)`。如果某一层不需要变换，网络只需要让 `F(x)` 接近0即可——这比学习恒等映射容易多了。

## 三、循环神经网络（RNN）：序列记忆者

### 为什么需要RNN？

全连接和CNN都有个问题：它们假设输入之间是独立的。但语言、语音、时间序列数据天然有**时序依赖**——"我吃了一个___"，下一个词和前面的词紧密相关。

RNN通过**隐藏状态**（hidden state）来记住历史信息：

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)
```

每个时刻的隐藏状态 `h_t` 不仅取决于当前输入 `x_t`，还取决于上一时刻的隐藏状态 `h_{t-1}`——这就是"记忆"。

### LSTM：长期记忆的突破

![记忆与遗忘](https://images.unsplash.com/photo-1501159599894-155982264a55?w=800&h=400&fit=crop)

原始RNN有个致命缺陷：**长期依赖问题**。当序列很长时，早期的信息在反向传播过程中会指数级衰减或爆炸——网络会"忘记"很久以前的事情。

LSTM（长短期记忆网络）用三个"门"来控制信息流：

| 门 | 作用 | 直觉 |
|----|------|------|
| 遗忘门（Forget Gate） | 决定丢弃什么信息 | "这段不重要，忘掉吧" |
| 输入门（Input Gate） | 决定存储什么新信息 | "这个要记住" |
| 输出门（Output Gate） | 决定输出什么信息 | "现在该用这段记忆了" |

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, 
                           num_layers=2, 
                           bidirectional=True,    # 双向LSTM
                           dropout=0.3,
                           batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2因为双向

    def forward(self, x):
        embedded = self.embedding(x)             # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)
        # 取最后一个时间步的输出
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.fc(hidden)
```

### RNN的变体家族

- **GRU**（门控循环单元）：LSTM的简化版，只有两个门，参数更少，训练更快
- **双向RNN**：同时从前向和后向读取序列，捕获双向上下文
- **多层RNN**：堆叠多层RNN，学习更抽象的特征

## 四、注意力机制（Attention）：Transformer的基石

### 从"全部记住"到"重点注意"

RNN的问题：必须按顺序处理序列，无法并行；而且即使有LSTM，超长序列的建模仍然困难。

注意力机制的核心思想：**不要试图记住所有东西，而是在需要的时候"关注"最相关的部分**。

就像你在一堆照片中找某个人——你不会仔细看每张照片的每个像素，而是快速扫描，把注意力集中在可能匹配的区域。

### Self-Attention 自注意力

![注意力聚焦](https://images.unsplash.com/photo-1485827404703-89b55fcc595e?w=800&h=400&fit=crop)

自注意力让序列中的每个位置都能直接"看到"其他所有位置，计算它们之间的相关性：

```python
import torch
import torch.nn.functional as F
import math

def self_attention(Q, K, V, mask=None):
    """
    Q: Query (batch, seq_len, d_k)
    K: Key   (batch, seq_len, d_k)
    V: Value (batch, seq_len, d_v)
    """
    d_k = Q.size(-1)
    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    # Softmax归一化
    attention_weights = F.softmax(scores, dim=-1)
    
    # 加权求和
    output = torch.matmul(attention_weights, V)
    return output, attention_weights
```

**Q、K、V 是什么？** 用图书馆找书来类比：
- **Query（查询）**：你脑中的需求——"我想找关于深度学习的书"
- **Key（键）**：每本书的标签——书脊上的分类标识
- **Value（值）**：书的实际内容

注意力过程就是：用你的需求（Q）和每本书的标签（K）匹配，找到最相关的书（V），然后重点阅读。

### Multi-Head Attention 多头注意力

一个"头"只能关注一种模式。多头注意力让网络同时从多个角度理解关系：

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # 线性变换并分头
        Q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力
        attn_output, _ = self_attention(Q, K, V, mask)
        
        # 拼接并线性变换
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.W_o(attn_output)
```

## 五、架构对比：一张图说清楚

| 架构 | 并行性 | 长程依赖 | 参数效率 | 典型应用 |
|------|--------|---------|---------|---------|
| FC | ✅ 完全并行 | ❌ 无结构感知 | ❌ 参数爆炸 | 表格数据 |
| CNN | ✅ 完全并行 | ⚠️ 局部感受野 | ✅ 权重共享 | 图像、视频 |
| RNN/LSTM | ❌ 必须顺序 | ✅ 天然时序 | ⚠️ 中等 | 早期NLP、时序 |
| Transformer | ✅ 完全并行 | ✅ 全局注意力 | ⚠️ 计算量大 | 现代NLP、多模态 |

## 实践建议

1. **图像任务**：先试CNN（ResNet、EfficientNet），如果数据量大可以试ViT
2. **文本任务**：直接上Transformer（BERT、GPT系列），LSTM只在资源受限时考虑
3. **时序数据**：数据量小用LSTM/GRU，数据量大用Transformer变体（如Informer）
4. **多模态**：Transformer架构（CLIP、LLaVA等）
5. **入门练手**：用CNN做图像分类，用LSTM做文本分类，对比理解差异

## 推荐资源

- 📘 [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/) — CNN经典课程
- 📘 [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/) — NLP + Transformer
- 📄 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer原论文（必读）
- 📄 [Deep Residual Learning](https://arxiv.org/abs/1512.03385) — ResNet论文
- 📺 [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 从零实现GPT

下一期我们将进入**数据预处理与特征工程**——再好的模型，喂垃圾数据也只能产出垃圾。敬请期待！

---

*本文由赛博阿漆AI助手自动生成 | AI学习之路系列 · 第3期*
