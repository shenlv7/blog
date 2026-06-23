---
title: "02 | Token：AI眼中的文字长什么样？"
description: "Token是LLM处理文本的最小单位。理解Token，才能理解为什么AI会'看不懂'某些文字，以及为什么API账单比你预期的贵。"
pubDate: 2026-06-25
tags: ["AI核心概念", "Token", "Tokenizer", "基础"]
difficulty: beginner
series: "ai-concepts"
seriesOrder: 2
---

## 一句话说清楚

> Token 是大语言模型处理文本的最小单位。模型不认字，只认 Token。

---

## 它是什么

你输入 `"今天天气真好"`，模型看到的不是这6个字，而是一串 Token 编号：

```
"今天天气真好" → [token_id_1, token_id_2, token_id_3, ...]
```

**Token 不等于字，也不等于词。** 它是介于两者之间的东西——由 Tokenizer（分词器）根据语料库的统计规律切分出来的子单元。

英文的例子更直观：

```
"unhappiness" → ["un", "happiness"]
"tokenization" → ["token", "ization"]
"I love AI" → ["I", " love", " AI"]
```

注意空格也是 Token 的一部分。`" love"` 和 `"love"` 是两个不同的 Token。

中文也类似：

```
"人工智能" → ["人工", "智能"]     # 2个Token
"今天天气真好" → ["今天", "天气", "真好"]  # 3个Token（大约）
```

---

## 它是怎么工作的

### BPE：最常见的分词算法

GPT 系列模型用的是 **BPE（Byte Pair Encoding）** 算法。核心思路很简单：

1. 从单个字符开始（最小粒度）
2. 统计语料中哪些字符对最常出现
3. 把最常见的字符对合并成一个新 Token
4. 重复步骤 2-3，直到 Token 表达到目标大小

```
语料: "low lower lowest"

初始: l o w   l o w e r   l o w e s t
第1轮: lo w   lo w e r   lo w e s t    (合并 lo)
第2轮: low   low e r   low e s t       (合并 low)
第3轮: low   lower   lowest            (合并 er, est)
```

最终模型学会了 `low`、`lower`、`lowest` 这些常见词作为完整 Token，而罕见词会被拆成更小的片段。

### Token 表大小

GPT-4 的 Token 表大约有 **100,256 个 Token**。这意味着：
- 常见词 → 1个 Token（如 `"the"`、`"是"`）
- 常见词组 → 1-2个 Token（如 `"人工智能"`）
- 罕见词/代码 → 多个 Token（如 `"Tokenization"` → `"Token"` + `"ization"`）

---

## 动手试试

### 实验 1：看看你的文字被切成了什么

```python
import tiktoken

# GPT-4o 使用的编码器
enc = tiktoken.encoding_for_model("gpt-4o")

texts = [
    "Hello, world!",
    "今天天气真好",
    "unhappiness",
    "I love artificial intelligence",
    "print('Hello World')",
]

for text in texts:
    tokens = enc.encode(text)
    token_strs = [enc.decode([t]) for t in tokens]
    print(f"\n原文: {text!r}")
    print(f"Token数: {len(tokens)}")
    print(f"切分结果: {token_strs}")
    print(f"Token ID: {tokens}")
```

**预期输出：**
```
原文: 'Hello, world!'
Token数: 4
切分结果: ['Hello', ',', ' world', '!']

原文: '今天天气真好'
Token数: 3
切分结果: ['今天', '天气', '真好']
```

### 实验 2：计算 Token 数量和费用

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

# 一篇典型的文章
article = """
人工智能（AI）正在改变我们的世界。从自动驾驶汽车到智能语音助手，
AI技术已经渗透到日常生活的方方面面。大语言模型（LLM）是AI领域
最新的突破，它能够理解和生成自然语言文本。
"""

tokens = enc.encode(article.strip())
token_count = len(tokens)

# GPT-4o 价格（示例）
input_price_per_1k = 0.0025   # $0.0025 / 1K tokens
output_price_per_1k = 0.01    # $0.01 / 1K tokens

print(f"Token 数量: {token_count}")
print(f"输入成本: ${token_count / 1000 * input_price_per_1k:.6f}")
print(f"如果生成同等长度输出: ${token_count / 1000 * output_price_per_1k:.6f}")

# 换算
chars_per_token = len(article.strip()) / token_count
print(f"\n平均每个Token对应 {chars_per_token:.1f} 个字符")
print(f"1000个Token ≈ {1000 * chars_per_token:.0f} 个字符 ≈ {1000 * chars_per_token / 2:.0f} 个中文字")
```

### 实验 3：不同语言的 Token 效率差异

```python
import tiktoken

enc = tiktoken.encoding_for_model("gpt-4o")

samples = {
    "英文": "Artificial intelligence is transforming the world.",
    "中文": "人工智能正在改变世界。",
    "日文": "人工知能は世界を変えています。",
    "代码": "def hello():\n    print('Hello, World!')",
    "Emoji": "🤖 AI is awesome! 🚀🎉",
}

for lang, text in samples.items():
    tokens = enc.encode(text)
    efficiency = len(text) / len(tokens)
    print(f"{lang:6s} | {len(tokens):3d} tokens | {len(text):3d} chars | 效率: {efficiency:.1f} char/token")
```

**你会发现：** 中文的 Token 效率比英文低——同样意思，中文消耗更多 Token。这是因为 GPT 的 Token 表是基于英文语料训练的，中文字符的合并机会更少。

---

## Token 的实际影响

### 1. 直接决定 API 账单

```
API 按 Token 计费：
  输入（你发给模型的文字）→ 按 Token 数收费
  输出（模型生成的文字）→ 按 Token 数收费（通常更贵）

省 Token = 省钱
```

### 2. 上下文窗口是 Token 数限制

```
GPT-4o: 128K tokens
Claude 3.5: 200K tokens

你的输入 + 输出 ≤ 上下文窗口
超了就会被截断，模型"忘记"前面的内容
```

### 3. 影响响应速度

```
Token 数越多 → 模型要处理的信息越多 → 响应越慢
尤其输出 Token，每生成一个都要跑一次前向推理
```

---

## 常见误区

### ❌ 误区 1：1个汉字 = 1个Token
真相：大多数汉字是 1-2 个 Token，取决于是否在 Token 表中。生僻字可能被拆成更多 Token。

### ❌ 误区 2：空格和换行不占 Token
真相：**占的。** 空格、换行、制表符都是 Token。格式化很漂亮的代码比压缩过的消耗更多 Token。

### ❌ 误区 3：Token 越多回答越准
真相：Token 多只代表输入信息多，不代表质量高。关键信息密度才是重点。

---

## 延伸阅读

- [OpenAI Tokenizer](https://platform.openai.com/tokenizer) — 在线可视化分词工具
- [tiktoken GitHub](https://github.com/openai/tiktoken) — OpenAI 官方 Token 计算库
- [BPE 论文](https://arxiv.org/abs/1508.07909) — 原始论文

---

> **下一篇预告：** [03 | LLM：大语言模型到底在做什么？](/blog/posts/ai-concepts-03-llm) — 回到最根本的问题：这些"大模型"的本质是什么？
