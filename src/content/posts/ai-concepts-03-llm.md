---
title: "03 | LLM：大语言模型到底在做什么？"
description: "LLM的本质是一台文字接龙机器。理解这一点，就能推导出它所有的能力边界。"
pubDate: 2026-06-25
tags: ["AI核心概念", "LLM", "大语言模型", "基础"]
difficulty: beginner
series: "ai-concepts"
seriesOrder: 3
---

## 一句话说清楚

> LLM 的本质是一台**文字接龙机器**——给它一段文本，它预测下一个最可能的 Token，然后不断重复。

---

## 它是什么

所有花哨的名词——GPT、Claude、Llama、千问——底层做的是同一件事：

```
输入: "今天天气"
输出: "真"        ← 概率最高的下一个Token

输入: "今天天气真"
输出: "好"        ← 概率最高的下一个Token

输入: "今天天气真好"
输出: "，"        ← 概率最高的下一个Token

... 如此循环，直到生成结束标记
```

**它不是在"思考"，不是在"理解"，是在做概率运算。**

这听起来很简单，但这个简单机制加上足够大的数据和参数量，就涌现出了令人惊叹的能力——写代码、做翻译、写论文、推理数学题。

---

## 它是怎么工作的

### 三步理解 LLM

```
┌──────────────────────────────────────────┐
│  Step 1: 输入编码                          │
│  "帮我写代码" → [token_1, token_2, ...]   │
├──────────────────────────────────────────┤
│  Step 2: 前向推理                          │
│  每个Token经过Transformer层计算           │
│  输出：下一个Token的概率分布               │
├──────────────────────────────────────────┤
│  Step 3: 采样输出                          │
│  从概率分布中选一个Token（或取最高概率）   │
│  把选中的Token拼回输入，重复Step 2         │
└──────────────────────────────────────────┘
```

### 概率分布长什么样

假设输入是 `"中国的首都是"`：

```
Token        概率
─────────────────────
"北京"       0.85
"上海"       0.02
"一个"       0.01
"在"         0.01
...          ...

模型选 "北京" → 输出 "北京"
```

如果 Temperature 设得很高（比如 1.5），概率分布会被"拉平"：

```
Token        概率（高温）
─────────────────────
"北京"       0.35
"上海"       0.15
"一个"       0.10
"在"         0.08
...

模型可能选 "上海" → 输出 "上海"（低概率事件发生了）
```

这就是 Temperature 参数的本质：**控制模型在高概率和低概率选项之间的倾向程度。**

### 模型规模的含义

```
GPT-4:     ~1.8 万亿参数（传闻）
Llama 3:   700 亿参数
Qwen 2:    720 亿参数

参数越多 → 能记住的模式越多 → 能力越强（通常）
```

但规模不是越大越好。小模型在特定任务上可以超过大模型，关键是看你怎么用。

---

## 动手试试

### 实验 1：逐 Token 生成过程可视化

```python
import openai

client = openai.OpenAI()

prompt = "人工智能的未来是"

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    max_tokens=20,
    temperature=0,
    stream=False,
)

print(f"输入: {prompt}")
print(f"输出: {response.choices[0].message.content}")
print(f"\n使用 Token 数: {response.usage.prompt_tokens} (输入) + {response.usage.completion_tokens} (输出)")
```

### 实验 2：Temperature 对输出的影响

```python
import openai
from collections import Counter

client = openai.OpenAI()
prompt = "用一个词形容AI"

temperatures = [0, 0.5, 1.0, 1.5]

for temp in temperatures:
    results = []
    for _ in range(10):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=temp,
        )
        results.append(response.choices[0].message.content.strip())

    print(f"\nTemperature={temp}")
    print(f"  10次结果: {results}")
    print(f"  唯一答案数: {len(set(results))}")
```

**预期结果：**
- Temperature=0 → 10次结果几乎一样
- Temperature=1.5 → 10次结果五花八门

### 实验 3：流式输出（Streaming）

```python
import openai

client = openai.OpenAI()

stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "用100字解释量子计算"}],
    stream=True,
)

print("输出：", end="", flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

**注意看：** 文字是一个 Token 一个 Token 冒出来的，不是一次性返回。这就是"文字接龙"的直观体现。

---

## LLM 的三个硬约束

理解了"文字接龙"的本质，就能推导出 LLM 的三个先天限制：

### 约束 1：它不是在思考

它是根据概率选择下一个 Token。涉及严密逻辑推理的场景，它会"一本正经地胡说"：

```
问: "1234 × 5678 = ?"
答: "7006652"   ← 错的，正确答案是 7006652... 等等，其实它经常算错
```

### 约束 2：知识有截止日期

训练数据收录到某个时间点，之后发生的事它完全不知道：

```
问: "2026年世界杯冠军是谁？"
答: (它会编一个听起来合理的答案)
```

### 约束 3：同样输入，输出可能不同

因为它在多个高概率 Token 之间**随机采样**。想要输出稳定，需要：
- 设置 `temperature=0`
- 或者用 `seed` 参数固定随机种子

---

## 常见误区

### ❌ 误区 1：LLM 有"记忆"
真相：LLM 没有真正的记忆。每次对话都是从头开始，它看到的只是你发送的文本。所谓的"上下文"就是在输入里塞了之前的对话记录。

### ❌ 误区 2：模型越大越聪明
真相：模型大只代表参数多、训练数据多。在特定任务上，小模型 + 好 Prompt 可以超过大模型 + 烂 Prompt。

### ❌ 误区 3：LLM 理解你说的话
真相：它不理解，它在做模式匹配。它输出的每一个字都是概率运算的结果。看起来像"理解"只是因为模式匹配做得足够好。

---

## 延伸阅读

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 原始论文
- [OpenAI GPT-4 技术报告](https://arxiv.org/abs/2303.08774)
- [Llama 3 技术报告](https://ai.meta.com/blog/meta-llama-3/)

---

> **下一篇预告：** [04 | Embedding：如何把文字变成数字？](/blog/posts/ai-concepts-04-embedding) — 理解向量表示，这是语义搜索、RAG、推荐系统的数学基础。
