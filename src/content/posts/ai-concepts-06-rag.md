---
title: "06 | RAG：让AI不再胡说八道的外挂知识库"
description: "RAG（检索增强生成）是当前最实用的AI应用范式。先检索相关知识，再让模型基于事实回答，从根源上减少幻觉。"
pubDate: 2026-06-25
tags: ["AI核心概念", "RAG", "检索增强生成", "应用"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 6
---

## 一句话说清楚

> RAG = 先搜资料，再回答问题。让模型基于真实文档生成，而不是靠"编"。

---

## 它是什么

LLM 有两个致命问题：
1. **知识有截止日期** — 不知道训练数据之后发生的事
2. **会"幻觉"** — 不知道的事也会一本正经地编

RAG（Retrieval-Augmented Generation，检索增强生成）是目前最实用的解决方案：

```
用户提问
   ↓
① 检索：从知识库中找到相关文档片段
   ↓
② 增强：把检索到的内容塞进 Prompt
   ↓
③ 生成：模型基于真实文档回答
```

本质就是**给模型开卷考试**。模型不用"背"所有知识，只需要"读"检索到的资料然后总结。

---

## 它是怎么工作的

### 完整流程

```
┌─────────────────────────────────────────────────┐
│                   离线索引阶段                     │
│                                                  │
│  文档 → 分块 → Embedding → 存入向量数据库         │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│                   在线查询阶段                     │
│                                                  │
│  用户问题 → Embedding → 向量搜索 → Top-K 相关片段 │
│                                                  │
│  相关片段 + 用户问题 → 组装 Prompt → LLM 生成回答  │
└─────────────────────────────────────────────────┘
```

### 关键步骤详解

**1. 文档分块（Chunking）**
```
原始文档: 5000字的技术文档
         ↓ 按段落/句子/固定长度切分
分块结果: [chunk_1: 500字, chunk_2: 480字, ..., chunk_10: 520字]

为什么分块？
- Embedding 模型有长度限制（通常 8192 Token）
- 小块检索精度更高
- 可以精确引用来源
```

**2. 向量检索**
```
用户问题: "Python怎么处理JSON？"
         ↓ Embedding
查询向量: [0.2, -0.5, 0.8, ...]
         ↓ 与知识库中所有chunk的向量计算余弦相似度
结果:     chunk_47 (相似度 0.92) ← 最相关
         chunk_23 (相似度 0.87)
         chunk_91 (相似度 0.85)
```

**3. Prompt 组装**
```
System: 你是公司内部知识库助手。请基于以下参考资料回答问题。
       如果参考资料中没有相关信息，请明确说"我找不到相关信息"。

参考资料:
[1] Python标准库json模块提供了json.dumps()和json.loads()方法...
[2] 处理JSON文件时建议使用with语句确保文件正确关闭...
[3] 对于大型JSON数据，可以使用ijson库进行流式解析...

用户问题: Python怎么处理JSON？

请基于以上参考资料回答：
```

---

## 动手试试

### 实验 1：最简 RAG 系统

```python
import openai
import numpy as np

client = openai.OpenAI()

# ============ 知识库 ============
knowledge_base = [
    "Python的GIL（全局解释器锁）确保同一时刻只有一个线程执行Python字节码。这意味着多线程无法利用多核CPU进行并行计算。",
    "asyncio是Python 3.4引入的异步编程框架，使用async/await语法，适合I/O密集型任务。",
    "multiprocessing模块通过创建独立进程来绕过GIL限制，每个进程有自己的Python解释器和内存空间。",
    "GIL在CPython实现中存在，但Jython和IronPython等其他实现没有GIL。",
    "对于CPU密集型任务，推荐使用multiprocessing而非threading。",
    "Python 3.12开始尝试通过PEP 703逐步移除GIL。",
]

# ============ 索引阶段 ============
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

# 预计算知识库向量
kb_embeddings = [get_embedding(doc) for doc in knowledge_base]

# ============ 检索阶段 ============
def retrieve(query, top_k=3):
    query_emb = get_embedding(query)
    scores = []
    for i, doc_emb in enumerate(kb_embeddings):
        score = np.dot(query_emb, doc_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(doc_emb)
        )
        scores.append((i, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [(knowledge_base[i], score) for i, score in scores[:top_k]]

# ============ 生成阶段 ============
def rag_answer(question):
    # 检索相关文档
    results = retrieve(question, top_k=3)

    # 组装 Prompt
    context = "\n".join(f"[{i+1}] {doc}" for i, (doc, _) in enumerate(results))

    prompt = f"""请基于以下参考资料回答问题。如果资料中没有相关信息，请说"我找不到相关信息"。

参考资料:
{context}

问题: {question}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content, results

# ============ 测试 ============
question = "为什么Python多线程不能利用多核CPU？"
answer, sources = rag_answer(question)

print(f"问题: {question}\n")
print(f"回答: {answer}\n")
print("参考来源:")
for i, (doc, score) in enumerate(sources, 1):
    print(f"  [{i}] (相似度: {score:.3f}) {doc[:60]}...")
```

### 实验 2：对比有无 RAG 的效果

```python
import openai

client = openai.OpenAI()

question = "2026年最新的Python版本有什么新特性？"

# 无 RAG：直接问模型
response_no_rag = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": question}],
    temperature=0,
)

# 有 RAG：提供参考资料
context = """
Python 3.13（2024年10月发布）新特性：
- 实验性自由线程模式（no-GIL）
- 改进的交互式解释器
- 更好的错误信息
- typing模块增强
"""

response_with_rag = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{
        "role": "user",
        "content": f"请基于以下资料回答：\n{context}\n\n问题：{question}"
    }],
    temperature=0,
)

print("【无RAG】")
print(response_no_rag.choices[0].message.content[:300])
print("\n" + "="*50)
print("【有RAG】")
print(response_with_rag.choices[0].message.content[:300])
```

---

## RAG 的核心挑战

### 1. 分块策略

```
太大的块 → 检索不精确，包含大量无关信息
太小的块 → 丢失上下文，语义不完整

常见策略:
- 按段落分块（自然边界）
- 固定长度 + 重叠（如500字，重叠50字）
- 递归分割（先按大块分，再细分）
```

### 2. 检索质量

```
检索不到相关文档 → 模型没有参考资料 → 幻觉
检索到不相关文档 → 模型被误导 → 错误答案

提升方法:
- 混合检索（向量 + 关键词）
- 重排序（Reranking）
- 查询改写（Query Rewriting）
```

### 3. 上下文窗口限制

```
检索到的文档 + 问题 + 系统提示 ≤ 上下文窗口

GPT-4o: 128K tokens
如果每个chunk 500 tokens → 最多塞 200+ 个chunk（实际不会这么多）
需要在信息量和相关性之间平衡
```

---

## 常见误区

### ❌ 误区 1：RAG 能完全消除幻觉
真相：RAG 能**减少**幻觉，但不能完全消除。模型仍可能误解检索到的内容，或者在资料不足时编造。

### ❌ 误区 2：向量数据库越大越好
真相：数据质量 > 数据数量。垃圾文档只会增加噪音，降低检索精度。

### ❌ 误区 3：RAG 可以替代微调
真相：RAG 解决的是"知识"问题，微调解决的是"能力"问题。两者互补，不替代。

---

## 延伸阅读

- [RAG 原始论文](https://arxiv.org/abs/2005.11401) — Facebook AI Research
- [LangChain RAG 教程](https://python.langchain.com/docs/tutorials/rag/)
- [向量数据库对比：Pinecone vs Milvus vs Chroma](https://www.pinecone.io/learn/vector-database/)

---

> **下一篇预告：** [07 | Attention：Transformer的灵魂机制](/blog/posts/ai-concepts-07-attention) — 理解自注意力机制，打开Transformer的引擎盖。
