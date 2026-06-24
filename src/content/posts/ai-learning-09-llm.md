---
title: "AI学习之路(第9期)：大语言模型(LLM)——语言的炼金术"
slug: ai-learning-09-llm
pubDate: 2026-06-24
description: "从GPT到开源大模型，深入解析大语言模型的架构、训练范式与核心能力"
image: "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=1200"
tags: ["AI学习", "LLM", "大语言模型", "GPT", "Transformer", "深度学习"]
series: "AI学习之路"
episode: 9
---

# AI学习之路(第9期)：大语言模型(LLM)——语言的炼金术

![大语言模型](https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800)

> "语言是思想的衣服。" —— 塞缪尔·约翰逊

## 前言

上一期我们探索了GANs——两个AI互相博弈的艺术。这一期，我们进入当今AI最炙手可热的领域：**大语言模型（Large Language Models, LLMs）**。

从ChatGPT横空出世到如今百花齐放的开源大模型，LLM已经彻底改变了我们与AI交互的方式。它不仅仅是"聊天机器人"——它是理解、生成、推理的语言引擎，是通往通用人工智能（AGI）的关键路径之一。

这一期，我们不聊炒作，只聊本质：LLM到底是什么？它怎么训练出来的？为什么它能做到那么多看似"智能"的事情？

---

## 1. 什么是大语言模型？

### 1.1 本质定义

大语言模型，本质上是一个**超大规模的概率语言模型**。它的核心任务极其简单：

```
给定前面的文字，预测下一个词（token）。
```

就这么简单？就这么简单。但当模型参数从百万涨到千亿，训练数据从几GB涨到几TB，这个简单的"预测下一个词"任务就涌现出了惊人的能力——翻译、写作、编程、推理、甚至"理解"人类意图。

### 1.2 与传统NLP的区别

| 维度 | 传统NLP模型 | 大语言模型 |
|------|------------|-----------|
| 任务适配 | 一任务一模型 | 一个模型，多种任务 |
| 训练方式 | 监督学习为主 | 预训练 + 微调 + 对齐 |
| 参数规模 | 百万级 | 百亿到万亿级 |
| 数据需求 | 标注数据 | 海量无标注文本 |
| 使用方式 | 需要专业知识 | 自然语言提示（Prompt） |

### 1.3 关键里程碑

```
2017  Transformer架构诞生（"Attention is All You Need"）
2018  GPT-1（1.17亿参数）& BERT（3.4亿参数）
2019  GPT-2（15亿参数）—— "太危险不敢发布"
2020  GPT-3（1750亿参数）—— Few-shot学习的震撼
2022  ChatGPT —— 人机交互的范式转换
2023  GPT-4 —— 多模态能力
2024  开源大模型爆发（Llama 3, Qwen 2, DeepSeek等）
2025  Agent时代 —— LLM从"对话"走向"行动"
```

---

## 2. 核心架构：Transformer Decoder

### 2.1 回顾Transformer

在第8期（GANs）之前，我们还没正式讲过Transformer。简单回顾一下：

Transformer的核心创新是**自注意力机制（Self-Attention）**，它让模型能够并行处理序列中所有位置之间的关系，而不是像RNN那样逐步处理。

### 2.2 Decoder-Only架构

现代LLM几乎清一色采用**Decoder-Only**架构（GPT系列的路线）：

```
输入: "今天天气"
                ↓
    ┌─────────────────────┐
    │   Token Embedding   │  ← 将文字转为向量
    └─────────────────────┘
                ↓
    ┌─────────────────────┐
    │  Positional Encoding │  ← 告诉模型词序
    └─────────────────────┘
                ↓
    ┌─────────────────────┐
    │  Transformer Block  │  ← 重复N次
    │  ┌───────────────┐  │
    │  │ Masked Self-   │  │  ← 只能看到前面的词
    │  │ Attention      │  │
    │  └───────────────┘  │
    │  ┌───────────────┐  │
    │  │ Feed-Forward   │  │
    │  │ Network        │  │
    │  └───────────────┘  │
    └─────────────────────┘
                ↓
    ┌─────────────────────┐
    │  Output Projection  │  ← 预测下一个token
    └─────────────────────┘
                ↓
输出: "真好"（概率最高的token）
```

### 2.3 关键超参数

- **层数（n_layers）**：模型深度，通常24-128层
- **隐藏维度（d_model）**：向量维度，通常4096-12288
- **注意力头数（n_heads）**：多头注意力的并行头数
- **上下文窗口（context_length）**：能处理的最大token数

GPT-4据传有约1.8万亿参数，分布在120层中，上下文窗口128K tokens。

---

## 3. 训练三部曲

LLM的训练分为三个阶段，每阶段解决不同的问题：

### 3.1 第一阶段：预训练（Pre-training）

**目标**：学习语言的通用知识

```python
# 简化的预训练目标
def pretrain_loss(model, text_tokens):
    predictions = model(text_tokens[:-1])  # 输入前N-1个token
    target = text_tokens[1:]               # 预测后N-1个token
    return cross_entropy(predictions, target)
```

**数据来源**：
- 互联网爬取（Common Crawl等）
- 书籍、论文
- 代码仓库（GitHub）
- 维基百科

**数据量级**：通常1-15万亿tokens（约几TB文本）

**计算成本**：GPT-3预训练据估计算力成本约460万美元；GPT-4可能超过1亿美元

### 3.2 第二阶段：指令微调（Instruction Tuning / SFT）

**目标**：让模型学会"听话"

预训练后的模型只会"续写"，不会"回答问题"。需要通过人工标注的指令-回答对进行微调：

```json
{
  "instruction": "请用简单的语言解释什么是量子计算。",
  "input": "",
  "output": "量子计算是一种利用量子力学原理进行计算的技术..."
}
```

这一步让模型从"文本补全机器"变成"有帮助的助手"。

### 3.3 第三阶段：人类偏好对齐（RLHF/DPO）

**目标**：让模型的输出符合人类偏好

这是最关键的一步，解决"正确但有害"或"有用但不安全"的问题。

**RLHF（基于人类反馈的强化学习）**：
1. 让模型生成多个回答
2. 人类标注员对回答进行排序
3. 训练一个奖励模型（Reward Model）
4. 用PPO算法优化模型，最大化奖励

**DPO（直接偏好优化）**：更简洁的替代方案，直接从偏好数据优化，无需单独训练奖励模型。

```python
# DPO损失函数的核心思想
def dpo_loss(policy, ref_model, preferred, rejected):
    # 让policy更喜欢preferred回答，更不喜欢rejected回答
    loss = -log(sigmoid(
        beta * (log_prob(policy, preferred) - log_prob(policy, rejected))
        - beta * (log_prob(ref_model, preferred) - log_prob(ref_model, rejected))
    ))
    return loss
```

---

## 4. 核心能力涌现

当模型规模超过某个阈值时，会"涌现"出训练时未明确教授的能力：

### 4.1 In-Context Learning（上下文学习）

不需要更新参数，仅通过在提示中给出几个示例，模型就能学会新任务：

```
# Few-shot示例
输入："苹果 → Apple, 香蕉 → Banana, 橙子 →"
模型输出："Orange"
```

### 4.2 Chain-of-Thought（思维链推理）

通过"让我们一步步思考"这样的提示，模型能展示推理过程：

```
Q: 一个商店有15个苹果，卖掉了8个，又进了12个，还剩多少？
A: 让我们一步步思考：
   1. 起初有15个苹果
   2. 卖掉8个：15 - 8 = 7个
   3. 进了12个：7 + 12 = 19个
   答案：19个苹果
```

### 4.3 指令遵循

经过对齐训练后，模型能理解并执行复杂的多步骤指令，甚至能拒绝不合理的请求。

### 4.4 工具使用

现代LLM能生成结构化输出（如JSON），调用外部工具（计算器、搜索引擎、API），这为Agent奠定了基础。

---

## 5. 主流大模型对比

### 5.1 闭源模型

| 模型 | 公司 | 参数规模 | 特点 |
|------|------|---------|------|
| GPT-4o | OpenAI | 未公开 | 多模态，推理强 |
| Claude 3.5 | Anthropic | 未公开 | 安全对齐，长上下文 |
| Gemini Ultra | Google | 未公开 | 原生多模态 |

### 5.2 开源模型

| 模型 | 公司 | 参数规模 | 特点 |
|------|------|---------|------|
| Llama 3.1 | Meta | 8B/70B/405B | 开源标杆 |
| Qwen 2.5 | 阿里 | 0.5B-72B | 中文能力强 |
| DeepSeek V3 | DeepSeek | 671B MoE | 高性价比，MoE架构 |
| Mistral Large | Mistral | 未公开 | 欧洲代表 |

### 5.3 架构趋势

```
2023: Dense模型（所有参数都参与计算）
     ↓
2024: MoE混合专家（Mixture of Experts）
     → 激活参数少，总参数多，推理成本低
     ↓
2025: 超长上下文 + 多模态 + Agent能力
     → 128K-1M上下文窗口成为标配
     → 图片/视频/音频原生理解
     → 工具调用、代码执行、多步规划
```

---

## 6. 关键技术细节

### 6.1 Tokenization（分词）

LLM不直接处理文字，而是处理**token**：

```python
# BPE分词示例
"大语言模型" → ["大", "语言", "模型"]
"Transformer" → ["Trans", "former"]
"unhappiness" → ["un", "happi", "ness"]
```

主流分词算法：BPE（Byte-Pair Encoding）、SentencePiece

### 6.2 位置编码（Positional Encoding）

Transformer本身没有"位置感"，需要额外注入位置信息：

- **RoPE（旋转位置编码）**：当前主流，支持长度外推
- **ALiBi**：线性偏置，训练短推理长

### 6.3 KV Cache

推理时，已计算过的Key和Value可以缓存复用，避免重复计算：

```
生成第1个token: 计算所有层的K,V → 缓存
生成第2个token: 只计算新token的K,V，拼接到缓存
生成第3个token: 同上
...
```

这是LLM推理优化的核心技术之一。

### 6.4 量化（Quantization）

将模型权重从FP16压缩到INT8/INT4，大幅减少内存和计算需求：

```python
# 量化前后对比
FP16: 70B参数 × 2字节 = 140GB显存
INT4: 70B参数 × 0.5字节 = 35GB显存  ← 4倍压缩！
```

常用工具：GPTQ、AWQ、GGUF（llama.cpp）

---

## 7. 实战：用代码理解LLM推理

### 7.1 最简推理流程

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# 加载模型
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 编码输入
prompt = "请用一句话解释什么是大语言模型："
inputs = tokenizer(prompt, return_tensors="pt")

# 逐token生成
outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    temperature=0.7,       # 控制随机性
    top_p=0.9,             # 核采样
    do_sample=True
)

# 解码输出
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### 7.2 自定义采样策略

```python
import torch

def custom_generate(model, tokenizer, prompt, max_tokens=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    for _ in range(max_tokens):
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # 最后一个位置的logits
        
        # Top-p采样
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过top_p的token
        sorted_indices_to_remove = cumulative_probs > 0.9
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        # 采样
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 拼接
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # 检查结束符
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(input_ids[0])
```

---

## 8. LLM的局限与挑战

### 8.1 幻觉（Hallucination）

LLM会"一本正经地胡说八道"——生成看似合理但实际错误的内容：

```
用户: "谁发明了Python语言？"
LLM: "Python语言由Guido van Rossum于1989年在CWI发明。"
     ↑ 基本正确，但细节可能有误
```

**缓解方法**：RAG（检索增强生成）、事实性检查、置信度校准

### 8.2 上下文窗口限制

虽然上下文窗口在扩大，但：
- 越长的上下文，推理越慢（O(n²)复杂度）
- "中间遗忘"问题：模型倾向于记住开头和结尾，忽略中间内容

### 8.3 推理成本

一个70B参数的模型推理一次需要约140GB显存（FP16），这意味着：
- 个人用户难以本地运行大模型
- API调用有延迟和成本
- 实时应用受限

### 8.4 知识时效性

模型的知识截止于训练数据的时间点，无法获知最新信息。RAG和工具调用是主要解决方案。

---

## 9. 未来方向

### 9.1 推理模型（Reasoning Models）

2024-2025年的趋势：让模型"思考更长时间"来提高推理质量。

OpenAI的o1/o3系列、DeepSeek-R1等模型展示了"思维链+自我验证"的强大能力。模型不再是"想到什么说什么"，而是"先想后说"。

### 9.2 多模态融合

从"文字进文字出"到"图片/视频/音频进，任意模态出"：
- GPT-4o：原生多模态
- Gemini：视频理解
- 开源：LLaVA、Qwen-VL

### 9.3 小模型的逆袭

不是所有任务都需要千亿参数：
- Phi-3（3.8B）在特定任务上媲美GPT-3.5
- Gemma 2B在手机上运行
- 专用场景：小模型 > 通用大模型

### 9.4 长期记忆与个性化

当前LLM没有"记忆"——每次对话都是从零开始。未来方向：
- 外部记忆库
- 用户画像持久化
- 持续学习（不遗忘旧知识的前提下学习新知识）

---

## 10. 动手实践

### 10.1 本地运行大模型

```bash
# 使用ollama运行开源模型
curl -fsSL https://ollama.com/install.sh | sh
ollama run qwen2.5:7b

# 使用llama.cpp（更底层）
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
./main -m models/qwen2.5-7b-q4_k_m.gguf -p "你好，请介绍一下自己" -n 200
```

### 10.2 使用Hugging Face

```python
from transformers import pipeline

# 最简单的使用方式
generator = pipeline("text-generation", model="Qwen/Qwen2.5-7B-Instruct")
result = generator("给我讲一个关于AI的笑话：", max_length=200)
print(result[0]["generated_text"])
```

### 10.3 API调用

```python
import openai

client = openai.OpenAI(
    base_url="https://api.siliconflow.cn/v1",  # 或其他兼容API
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "请用简单的语言解释什么是量子计算。"}
    ],
    temperature=0.7,
    max_tokens=500
)

print(response.choices[0].message.content)
```

---

## 总结

| 概念 | 核心要点 |
|------|---------|
| LLM本质 | 超大规模的"下一个token预测"模型 |
| 架构 | Transformer Decoder-Only |
| 训练 | 预训练 → SFT → RLHF/DPO |
| 能力涌现 | In-context learning、思维链、指令遵循 |
| 关键挑战 | 幻觉、上下文限制、推理成本 |
| 未来方向 | 推理增强、多模态、小模型、个性化 |

大语言模型不是魔法，而是**工程+规模+数据**的胜利。理解它的原理，才能更好地使用它、超越它。

---

## 下期预告

**第10期：AI Agent与工具使用** —— 当LLM学会"动手"，从对话助手变成真正的行动者。我们将探索Agent架构、工具调用、多步规划，以及如何构建你自己的AI Agent。

---

*赛博阿漆 · AI学习之路 · 2026年6月24日*
*系列：AI学习之路 | 第9期 / 共12期*
