---
title: "09 | Fine-tuning：花小钱办大事的模型定制"
description: "Fine-tuning是在预训练模型基础上，用少量领域数据微调，让通用模型变成领域专家的高效方法。"
pubDate: 2026-06-25
tags: ["AI核心概念", "Fine-tuning", "微调", "LoRA", "SFT"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 9
---

## 一句话说清楚

> Fine-tuning 是用你的数据在预训练模型上"再教一遍"，让通用模型变成你的领域专家。

---

## 它是什么

预训练好的 LLM（如 GPT、Llama）是一个"通才"——什么都知道一点，但在你的特定场景可能不够好。

Fine-tuning 就是在这个通才基础上，用你的数据做针对性训练：

```
预训练模型（通才）
    + 你的领域数据
    ↓ Fine-tuning
你的专属模型（专家）
```

**类比：**
- 预训练 = 大学通识教育
- Fine-tuning = 专业实习

---

## 它是怎么工作的

### SFT（监督微调）

最常见的微调方式，用"指令-回答"对训练：

```json
{
  "messages": [
    {"role": "system", "content": "你是一名法律助手"},
    {"role": "user", "content": "什么是合同违约？"},
    {"role": "assistant", "content": "合同违约是指当事人不履行合同义务或履行不符合约定..."}
  ]
}
```

训练过程：
```
输入: "什么是合同违约？"
模型预测: "合同违约是..."
正确答案: "合同违约是指当事人不履行合同义务..."

计算差距(损失) → 反向传播 → 更新参数
重复几千到几万次
```

### LoRA（低秩适配）

全量微调需要更新所有参数，成本很高。LoRA 的思路是：**只训练一小部分"额外"参数**

```
原始模型参数: W (d × d) = 1,000,000 个参数

LoRA:
  冻结原始 W 不动
  新增两个小矩阵:
    A (d × r) = 1,000 × 16 = 16,000 个参数
    B (r × d) = 16 × 1,000 = 16,000 个参数
  
  输出 = W×x + A×B×x
  只训练 A 和 B
  可训练参数: 32,000 / 1,000,000 = 3.2%
```

**LoRA 的优势：**
- 训练参数量减少 90%+
- 显存占用大幅降低
- 效果接近全量微调
- 可以给同一基础模型套不同的 LoRA 适配器

---

## 动手试试

### 实验 1：用 OpenAI API 做 Fine-tuning

```python
import openai

client = openai.OpenAI()

# ① 准备训练数据（JSONL格式）
training_data = [
    {
        "messages": [
            {"role": "system", "content": "你是一个Python代码审查助手"},
            {"role": "user", "content": "review这段代码: x = [i for i in range(10) if i % 2 == 0]"},
            {"role": "assistant", "content": "这段代码用列表推导式筛选偶数，写法简洁。建议：\n1. 变量名建议改为 evens = ...\n2. 如果只需要一个结果，考虑用 generator: (i for i in range(10) if i % 2 == 0)"}
        ]
    },
    # ... 更多训练样本
]

# 保存为 JSONL
import json
with open("training_data.jsonl", "w") as f:
    for item in training_data:
        f.write(json.dumps(item) + "\n")

# ② 上传训练文件
file = client.files.create(
    file=open("training_data.jsonl", "rb"),
    purpose="fine-tune",
)
print(f"文件上传成功: {file.id}")

# ③ 创建微调任务
job = client.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-4o-mini-2024-07-18",
    hyperparameters={
        "n_epochs": 3,
    },
)
print(f"微调任务已创建: {job.id}")

# ④ 查看状态
job = client.fine_tuning.jobs.retrieve(job.id)
print(f"状态: {job.status}")
```

### 实验 2：LoRA 微调示例（使用 transformers + peft）

```python
# 需要安装: pip install transformers peft datasets accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch

# ① 加载基础模型
model_name = "Qwen/Qwen2-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# ② 配置 LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                    # 低秩维度
    lora_alpha=32,          # 缩放系数
    lora_dropout=0.1,       # Dropout
    target_modules=["q_proj", "v_proj"],  # 应用LoRA的层
)

# ③ 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# 输出: trainable params: 1,234,567 || all params: 1,500,000,000 || trainable%: 0.08%

# ④ 准备数据
data = {
    "text": [
        "### 指令: 什么是机器学习?\n### 回答: 机器学习是人工智能的一个分支...",
        "### 指令: 解释过拟合\n### 回答: 过拟合是指模型在训练数据上表现很好...",
    ]
}
dataset = Dataset.from_dict(data)

# ⑤ 训练（简化示例）
training_args = TrainingArguments(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-4,
    logging_steps=10,
)

# 实际训练代码需要使用 Trainer...
print("LoRA配置完成，可以开始训练")
print(f"原始参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### 实验 3：对比微调前后的效果

```python
import openai

client = openai.OpenAI()

test_prompt = "解释一下什么是梯度下降"

# 原始模型
response_base = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": test_prompt}],
    temperature=0,
)

# 微调后的模型（替换为你的微调模型ID）
# response_finetuned = client.chat.completions.create(
#     model="ft:gpt-4o-mini-2024-07-18:your-org::your-model-id",
#     messages=[{"role": "user", "content": test_prompt}],
#     temperature=0,
# )

print("【原始模型】")
print(response_base.choices[0].message.content[:300])
print("\n【微调模型】")
# print(response_finetuned.choices[0].message.content[:300])
print("（需要先完成微调，替换模型ID）")
```

---

## 什么时候该用 Fine-tuning

| 场景 | 用 Prompt Engineering | 用 RAG | 用 Fine-tuning |
|------|---------------------|--------|---------------|
| 改变输出风格 | ✅ 试试先 | ❌ | ✅ |
| 加入领域知识 | ❌ | ✅ | ⚠️ 可以但RAG更灵活 |
| 特定格式输出 | ✅ Few-shot | ❌ | ✅ |
| 提升推理能力 | ⚠️ 效果有限 | ❌ | ✅ |
| 降低成本 | ❌ | ❌ | ✅ 小模型微调替代大模型 |

**经验法则：** 先试 Prompt Engineering → 不够就加 RAG → 还不够才上 Fine-tuning

---

## 常见误区

### ❌ 误区 1：数据越多越好
真相：数据**质量**远比数量重要。100条高质量数据 > 10000条垃圾数据。

### ❌ 误区 2：微调能让模型"学会"新知识
真相：微调更多是教模型"怎么用"已有知识，而不是"注入"新知识。注入新知识用 RAG 更好。

### ❌ 误区 3：全量微调一定比 LoRA 好
真相：在大多数场景下，LoRA 效果接近全量微调，但成本低一个数量级。除非你有充足的算力预算，否则优先用 LoRA。

---

## 延伸阅读

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [OpenAI Fine-tuning Guide](https://platform.openai.com/docs/guides/fine-tuning)
- [Hugging Face PEFT 库](https://huggingface.co/docs/peft)
- [QLoRA 论文](https://arxiv.org/abs/2305.14314) — 4-bit 量化的 LoRA

---

> **下一篇预告：** [10 | Agent：让AI自己"动手"干活](/blog/posts/ai-concepts-10-agent) — 从"对话"到"行动"，AI Agent 的核心范式。
