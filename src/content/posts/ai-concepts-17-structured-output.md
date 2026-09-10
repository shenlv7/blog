---
title: "17 | 结构化输出：怎么让AI的回答变成程序能读的数据？"
description: "AI的回答只有人能看懂？结构化输出(Structured Output)让模型按JSON格式返回数据，程序直接解析使用"
pubDate: 2026-09-10
tags: ["AI核心概念", "结构化输出", "JSON Mode", "API", "提示工程"]
difficulty: beginner
series: "ai-concepts"
seriesOrder: 17
slug: ai-concepts-17-structured-output
---

# AI核心概念(17)：结构化输出——让AI的回答变成程序能读的数据

> "人看自然语言，程序读结构化数据。结构化输出就是那台翻译机。"

## 🎯 什么是结构化输出？

**结构化输出（Structured Output）**：让AI模型按照**你指定的格式**（通常是 JSON）返回结果，而不是一段自由发挥的文字。

```
自由文本（人能懂，程序难办）：
"这个产品的名称是智能音箱，分类属于智能家居，价格大概是299元。"

结构化输出（程序直接能用）：
{
  "name": "智能音箱",
  "category": "智能家居",
  "price": 299
}
```

**本质**：把AI的输出从"给人类看的文章"变成"给程序吃的数据"。

## 🔬 为什么需要它？

想象你要用AI自动处理1000条用户评价——提取"产品名、情感倾向、问题类型"三项。

- **自由文本**：模型返回一句话，你得写正则表达式去猜、去切分——格式稍有变化就解析失败
- **结构化输出**：模型直接返回 `{"product": "...", "sentiment": "negative", "issue": "..."}`，程序一行 `json.loads()` 搞定

这是AI从"聊天玩具"变成"生产工具"的关键一步——**能被程序稳定消费，才能接入业务流程**。

## 🏗️ 三种实现方式

| 方式 | 做法 | 可靠性 |
|------|------|--------|
| **提示词约束** | 在 prompt 里说"请用JSON回答" | 低，模型可能不听话 |
| **JSON Mode** | 调用API时开启 `response_format` | 中，保证是合法JSON |
| **Schema 约束** | 指定严格的字段结构 | 高，字段名/类型都保证 |

服务商对严格模式的支持不一，**JSON Mode 是目前兼容性最好的起点**。

## 💻 动手Demo

```python
import json
from openai import OpenAI

client = OpenAI(
    api_key="你的API Key",
    base_url="https://api.deepseek.com"
)

resp = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system",
         "content": "必须以JSON回答，字段：name(概念名), category(分类), one_line(一句话解释)"},
        {"role": "user", "content": "介绍 提示词缓存 Prompt Caching"}
    ],
    response_format={"type": "json_object"}   # 关键：开启JSON模式
)

data = json.loads(resp.choices[0].message.content)
print(data["name"], "→", data["category"])
```

实测输出（作者机器，DeepSeek API）：

```json
{
  "name": "提示词缓存",
  "category": "大语言模型优化技术",
  "one_line": "提示词缓存是一种将重复使用的提示词前缀...从而减少计算量、降低延迟和成本的技术。"
}
```

直接 `json.loads()` 解析成功，无需任何文本清洗。

## 💡 实用要点

- 用 JSON Mode 时，**system 提示里必须提到"JSON"**，否则部分服务商会直接报错
- 字段描述写清楚类型（`price` 是数字还是字符串），避免程序端类型错误
- 拿到数据后**仍要校验**（字段是否齐全、数值是否合理）——模型可能字段填得不对
- 需要严格保证结构时，用服务商的 Schema 约束模式或函数调用

结构化输出不性感，但它是把AI接进真实系统的地基。下次你的程序需要AI给数据而不是给文章，记得它。

---

_概念卡片持续更新中，下一个概念见~_ 🔫
