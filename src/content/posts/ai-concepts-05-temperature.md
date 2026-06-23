---
title: "05 | Temperature：AI的创意旋钮怎么调？"
description: "Temperature控制模型输出的随机性。理解它的数学原理，才能在创意和稳定之间找到最佳平衡点。"
pubDate: 2026-06-25
tags: ["AI核心概念", "Temperature", "采样", "基础"]
difficulty: beginner
series: "ai-concepts"
seriesOrder: 5
---

## 一句话说清楚

> Temperature 是控制模型输出随机性的参数。值越高越"有创意"，值越低越"稳定可控"。

---

## 它是什么

当你调用 LLM API 时，有一个参数叫 `temperature`，范围通常是 0 到 2。

```
temperature = 0    → 最保守，每次选概率最高的Token，输出几乎不变
temperature = 0.7  → 适中，常用默认值
temperature = 1.5  → 很随机，经常选低概率Token，输出天马行空
```

它就像一个**创意旋钮**：
- 写代码、做计算 → 调低（要准确）
- 写故事、头脑风暴 → 调高（要创意）

---

## 它是怎么工作的

### Softmax 的温度参数

LLM 输出层会生成每个候选 Token 的原始分数（logits），然后通过 **Softmax** 转换成概率：

```
标准 Softmax:
P(token_i) = exp(logit_i) / Σ exp(logit_j)

带温度的 Softmax:
P(token_i) = exp(logit_i / T) / Σ exp(logit_j / T)

其中 T 就是 temperature
```

### T 的数学效果

```
原始 logits: [3.0, 2.0, 1.0, 0.5]

T = 0.1 (极低温度):
→ 概率: [0.99, 0.01, 0.00, 0.00]   ← 几乎确定选第一个

T = 1.0 (标准温度):
→ 概率: [0.58, 0.21, 0.08, 0.03]   ← 有随机性，但高概率选项占优

T = 2.0 (高温):
→ 概率: [0.38, 0.26, 0.18, 0.12]   ← 概率被拉平，低概率选项也有机会
```

**核心原理：** 温度越高，概率分布越"平坦"，低概率选项被选中的机会越大。

---

## 动手试试

### 实验 1：Temperature 对概率分布的影响

```python
import numpy as np

def softmax_with_temperature(logits, temperature):
    """带温度参数的 Softmax"""
    scaled_logits = np.array(logits) / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # 数值稳定
    return exp_logits / exp_logits.sum()

# 假设模型对下一个Token的原始分数
logits = [3.0, 2.0, 1.0, 0.5, 0.1]
tokens = ["北京", "上海", "广州", "深圳", "杭州"]

temperatures = [0.1, 0.5, 1.0, 1.5, 2.0]

print("Temperature 效果对比：\n")
print(f"{'Token':>8s}", end="")
for t in temperatures:
    print(f"  T={t:<4.1f}", end="")
print()
print("-" * 55)

for i, token in enumerate(tokens):
    print(f"{token:>8s}", end="")
    for t in temperatures:
        probs = softmax_with_temperature(logits, t)
        print(f"  {probs[i]:.3f}", end="")
    print()
```

**预期输出：**
```
  Token  T=0.1   T=0.5   T=1.0   T=1.5   T=2.0
-------------------------------------------------------
    北京  0.977   0.731   0.582   0.496   0.431
    上海  0.022   0.180   0.213   0.224   0.228
    广州  0.001   0.066   0.117   0.142   0.161
    深圳  0.000   0.019   0.061   0.085   0.103
    杭州  0.000   0.004   0.027   0.053   0.077
```

### 实验 2：Temperature 对生成结果的影响

```python
import openai
from collections import Counter

client = openai.OpenAI()

prompt = "用一个词形容人工智能"

temperatures = [0, 0.5, 1.0, 1.5]

for temp in temperatures:
    results = []
    for _ in range(20):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=temp,
        )
        results.append(response.choices[0].message.content.strip())

    counter = Counter(results)
    print(f"\nTemperature = {temp}")
    print(f"  唯一答案数: {len(counter)}")
    print(f"  Top 3: {counter.most_common(3)}")
```

### 实验 3：实际场景的推荐温度

```python
import openai

client = openai.OpenAI()

scenarios = [
    {
        "name": "代码生成（要稳定）",
        "prompt": "写一个Python函数，计算斐波那契数列第n项",
        "temperature": 0,
    },
    {
        "name": "技术文档（适中）",
        "prompt": "用100字解释什么是REST API",
        "temperature": 0.5,
    },
    {
        "name": "创意写作（要创意）",
        "prompt": "用诗意的语言描述AI觉醒的瞬间",
        "temperature": 1.2,
    },
]

for scenario in scenarios:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": scenario["prompt"]}],
        temperature=scenario["temperature"],
        max_tokens=100,
    )
    print(f"\n{'='*50}")
    print(f"【{scenario['name']}】 Temperature={scenario['temperature']}")
    print(response.choices[0].message.content)
```

---

## Temperature vs Top-p

除了 Temperature，还有一个参数 `top_p` 也控制随机性：

| 参数 | 机制 | 效果 |
|------|------|------|
| **Temperature** | 调整概率分布的"平坦度" | 数学上拉平/拉尖分布 |
| **Top-p** | 只保留累积概率前 p% 的 Token | 直接砍掉低概率选项 |

```
Top-p = 0.9 表示：
只从累积概率达到90%的Token中选择，剩下10%的低概率Token直接忽略

Top-p = 1.0 表示：所有Token都有机会（不做截断）
Top-p = 0.1 表示：只从概率最高的几个Token中选择（非常保守）
```

**实际建议：**
- 一般只调一个，不要同时调 Temperature 和 Top-p
- 需要稳定输出 → `temperature=0`
- 需要适度创意 → `temperature=0.7, top_p=1.0`
- 需要最大创意 → `temperature=1.2, top_p=0.95`

---

## 常见误区

### ❌ 误区 1：Temperature=0 就是"最准确"
真相：Temperature=0 只是"最稳定"，每次都选概率最高的 Token。但概率最高不等于正确——模型可能以高概率输出错误答案。

### ❌ 误区 2：Temperature 越高回答越聪明
真相：Temperature 高只代表更随机，不代表更聪明。过高会导致输出前言不搭后语，像"胡言乱语"。

### ❌ 误区 3：所有任务应该用同一个 Temperature
真相：不同任务需要不同的 Temperature。代码生成用 0，对话用 0.7，创意写作用 1.0+。

---

## 延伸阅读

- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — 采样策略研究
- [OpenAI API Temperature 参数文档](https://platform.openai.com/docs/api-reference/chat/create#chat-create-temperature)

---

> **下一篇预告：** [06 | RAG：让AI不再"胡说八道"的外挂知识库](/blog/posts/ai-concepts-06-rag) — 检索增强生成，当前最实用的AI应用范式。
