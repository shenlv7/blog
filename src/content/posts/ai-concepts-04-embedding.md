---
title: "04 | Embedding：如何把文字变成数字？"
description: "Embedding是将文本映射到向量空间的技术。理解它，就理解了语义搜索、RAG、推荐系统的数学基础。"
pubDate: 2026-06-25
tags: ["AI核心概念", "Embedding", "向量", "语义搜索"]
difficulty: beginner
series: "ai-concepts"
seriesOrder: 4
---

## 一句话说清楚

> Embedding 是把文字变成一组数字（向量），让计算机能用数学方式理解"语义相似"。

---

## 它是什么

计算机不认识文字，只认识数字。Embedding 就是把一段文字映射成一个**固定长度的数字数组**（向量）：

```
"猫"     → [0.2, -0.5, 0.8, 0.1, ..., -0.3]   # 1536个数字
"狗"     → [0.3, -0.4, 0.7, 0.2, ..., -0.2]   # 跟"猫"很近
"汽车"   → [-0.6, 0.1, -0.3, 0.9, ..., 0.5]   # 跟"猫"很远
```

关键特性：**语义相近的文字，向量也相近。**

这就是为什么 Embedding 能做语义搜索——你搜"可爱的小动物"，系统能找到"猫"和"狗"，即使它们字面上完全不同。

---

## 它是怎么工作的

### 向量空间

想象一个三维空间：

```
        z轴
        ↑
        |    "汽车"
        |   /
        |  /
        | /
        +--------→ y轴
       /
      /
     /
    x轴
    
    "猫" ←──→ "狗"    （距离近，语义相似）
```

实际上 Embedding 维度远不止三维（通常是 768、1536、3072 维），但原理一样。

### 距离度量：怎么判断"像不像"

最常用的是**余弦相似度**：

```
相似度 = cos(θ) = (A · B) / (|A| × |B|)

- 相似度 = 1  → 完全相同
- 相似度 = 0  → 完全无关
- 相似度 = -1 → 完全相反
```

```
"猫" vs "狗"       → 相似度 ≈ 0.85 （很像）
"猫" vs "汽车"     → 相似度 ≈ 0.12 （不像）
"猫" vs "可爱宠物" → 相似度 ≈ 0.78 （语义相关）
```

---

## 动手试试

### 实验 1：计算文本相似度

```python
import openai
import numpy as np

client = openai.OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 待比较的文本
texts = [
    "猫是一种可爱的宠物",
    "狗是人类最好的朋友",
    "今天股市大涨",
    "小猫咪好可爱呀",
]

# 获取 Embedding
embeddings = [get_embedding(t) for t in texts]

# 计算相似度矩阵
print("相似度矩阵：\n")
print(f"{'':20s}", end="")
for t in texts:
    print(f"{t[:8]:10s}", end="")
print()

for i, t1 in enumerate(texts):
    print(f"{t1[:18]:20s}", end="")
    for j, t2 in enumerate(texts):
        sim = cosine_similarity(embeddings[i], embeddings[j])
        print(f"{sim:10.3f}", end="")
    print()
```

**预期输出：**
```
                    猫是一种可  狗是人类最  今天股市大  小猫咪好可
猫是一种可爱的宠物      1.000      0.823      0.156      0.912
狗是人类最好的朋友      0.823      1.000      0.178      0.756
今天股市大涨           0.156      0.178      1.000      0.134
小猫咪好可爱呀         0.912      0.756      0.134      1.000
```

### 实验 2：用 Embedding 做简易语义搜索

```python
import openai
import numpy as np

client = openai.OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 模拟一个知识库
knowledge_base = [
    "Python是一种解释型编程语言，由Guido van Rossum创建",
    "JavaScript是Web前端开发的核心语言",
    "机器学习是人工智能的一个子领域",
    "深度学习使用多层神经网络处理复杂数据",
    "Docker是一个容器化平台，用于打包和部署应用",
    "Git是分布式版本控制系统",
]

# 预计算知识库的 Embedding
kb_embeddings = [get_embedding(t) for t in knowledge_base]

# 搜索
query = "神经网络怎么学习？"
query_embedding = get_embedding(query)

# 计算相似度并排序
scores = [(i, cosine_similarity(query_embedding, emb)) for i, emb in enumerate(kb_embeddings)]
scores.sort(key=lambda x: x[1], reverse=True)

print(f"查询: {query}\n")
print("搜索结果（按相关度排序）：")
for rank, (idx, score) in enumerate(scores[:3], 1):
    print(f"  {rank}. [{score:.3f}] {knowledge_base[idx]}")
```

### 实验 3：Embedding 降维可视化（需要 matplotlib）

```python
import openai
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

client = openai.OpenAI()

categories = {
    "动物": ["猫", "狗", "鸟", "鱼"],
    "水果": ["苹果", "香蕉", "橙子", "葡萄"],
    "编程": ["Python", "JavaScript", "算法", "数据库"],
}

texts = []
labels = []
colors = []
color_map = {"动物": "red", "水果": "green", "编程": "blue"}

for cat, words in categories.items():
    for word in words:
        texts.append(word)
        labels.append(word)
        colors.append(color_map[cat])

# 获取 Embedding
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)
embeddings = [d.embedding for d in response.data]

# 降到2维
pca = PCA(n_components=2)
coords = pca.fit_transform(embeddings)

# 画图
plt.figure(figsize=(10, 8))
for i, (x, y) in enumerate(coords):
    plt.scatter(x, y, c=colors[i], s=100)
    plt.annotate(labels[i], (x, y), fontsize=12, ha='center', va='bottom')

plt.title("Embedding 向量空间可视化 (PCA降维)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("embedding_visualization.png", dpi=150, bbox_inches='tight')
plt.show()
```

**你会看到：** 同类词在向量空间中聚集在一起，不同类的词明显分开。

---

## Embedding 的应用场景

| 场景 | 怎么用 Embedding |
|------|-----------------|
| **语义搜索** | 把查询和文档都转成向量，找最近的 |
| **RAG** | 先检索相关文档片段，再让 LLM 基于这些片段回答 |
| **推荐系统** | 用户和物品都 Embedding，找相似的 |
| **聚类分析** | 把一堆文档 Embedding 后做 K-Means 聚类 |
| **异常检测** | 正常数据聚在一起，异常数据远离中心 |
| **去重** | 相似度超过阈值的文本判定为重复 |

---

## 常见误区

### ❌ 误区 1：Embedding 能完全代表语义
真相：Embedding 是**近似**语义表示。它会丢失细节，对反义词、讽刺、双关语处理不好。

### ❌ 误区 2：不同模型的 Embedding 可以混用
真相：**不能。** 不同模型的 Embedding 在不同的向量空间里，直接比较没有意义。

### ❌ 误区 3：维度越高越好
真相：高维能表达更多细节，但也更耗存储和计算。实际使用中 768-1536 维通常够用。

---

## 延伸阅读

- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — 语义 Embedding 的经典模型
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — Embedding 模型排行榜

---

> **下一篇预告：** [05 | Temperature：AI的"创意旋钮"怎么调？](/blog/posts/ai-concepts-05-temperature) — 控制模型输出的随机性和创造性的数学原理。
