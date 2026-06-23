---
title: "10 | Agent：让AI自己动手干活"
description: "AI Agent是能自主决策、调用工具、执行任务的智能体。从对话到行动，这是AI应用的下一个范式。"
pubDate: 2026-06-25
tags: ["AI核心概念", "Agent", "工具调用", "ReAct"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 10
---

## 一句话说清楚

> Agent = LLM + 工具 + 记忆 + 规划。让AI不只是"说"，还能"做"。

---

## 它是什么

普通的 LLM 对话是这样的：

```
用户: "今天北京天气怎么样？"
AI:   "我无法获取实时天气信息..."  ← 只能说，不能做
```

Agent 是这样的：

```
用户: "今天北京天气怎么样？"
AI(思考): 需要查询实时天气，调用天气API
AI(行动): 调用 get_weather(city="北京")
AI(观察): 返回 {"temp": 28, "weather": "晴"}
AI(回答): "今天北京28度，晴天，适合出门。"  ← 能说能做
```

**核心区别：Agent 能使用工具、能观察结果、能自主决策下一步。**

---

## 它是怎么工作的

### ReAct 范式

最经典的 Agent 框架是 **ReAct（Reasoning + Acting）**：

```
Thought: 我需要查一下北京的天气
Action:  get_weather(city="北京")
Observation: {"temp": 28, "weather": "晴"}
Thought: 拿到天气信息了，可以回答用户
Answer:  今天北京28度，晴天。
```

循环过程：

```
┌──────────────────────────────────────┐
│                                      │
│   ┌─────────┐    ┌─────────┐        │
│   │ Thought │ →  │ Action  │        │
│   │ (思考)   │    │ (行动)   │        │
│   └─────────┘    └────┬────┘        │
│        ↑              ↓             │
│        │        ┌───────────┐       │
│        │        │ Observation│       │
│        │        │ (观察结果)  │       │
│        │        └─────┬─────┘       │
│        │              ↓             │
│        └──────────────┘             │
│                                      │
│   直到任务完成或达到最大步数          │
└──────────────────────────────────────┘
```

### Function Calling

现代 LLM 通过 **Function Calling** 来调用工具：

```json
// 1. 你告诉模型有哪些工具可用
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "查询指定城市的天气",
        "parameters": {
          "type": "object",
          "properties": {
            "city": {"type": "string", "description": "城市名"}
          },
          "required": ["city"]
        }
      }
    }
  ]
}

// 2. 模型决定调用什么工具
{
  "tool_calls": [
    {
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"北京\"}"
      }
    }
  ]
}

// 3. 你执行工具，把结果告诉模型
{
  "role": "tool",
  "content": "{\"temp\": 28, \"weather\": \"晴\"}"
}

// 4. 模型基于工具结果生成最终回答
"今天北京28度，晴天，适合出门。"
```

---

## 动手试试

### 实验 1：最简 Agent（Function Calling）

```python
import openai
import json

client = openai.OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "执行数学计算",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前时间",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]

# 模拟工具实现
def execute_tool(name, args):
    if name == "calculate":
        try:
            result = eval(args["expression"])  # 仅用于演示，生产环境不要用eval
            return str(result)
        except Exception as e:
            return f"计算错误: {e}"
    elif name == "get_current_time":
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return "未知工具"

# Agent 循环
def agent_chat(user_message, max_steps=5):
    messages = [{"role": "user", "content": user_message}]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        # 如果模型要调用工具
        if msg.tool_calls:
            messages.append(msg)
            for tool_call in msg.tool_calls:
                result = execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                print(f"  🔧 调用工具: {tool_call.function.name}({tool_call.function.arguments})")
                print(f"  📋 结果: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
        else:
            # 模型给出最终回答
            return msg.content

    return "达到最大步数，任务未完成"

# 测试
print("问题: 现在几点了？3+5等于多少？")
answer = agent_chat("现在几点了？另外帮我算一下3+5等于多少")
print(f"\n回答: {answer}")
```

### 实验 2：带记忆的 Agent

```python
import openai
import json

client = openai.OpenAI()

class SimpleAgent:
    def __init__(self, system_prompt="你是一个有用的助手"):
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tools = []

    def add_tool(self, name, description, parameters, func):
        self.tools.append({
            "definition": {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": parameters,
                }
            },
            "func": func,
        })

    def chat(self, user_message):
        self.messages.append({"role": "user", "content": user_message})

        tool_defs = [t["definition"] for t in self.tools]

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.messages,
            tools=tool_defs if tool_defs else None,
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            self.messages.append(msg)
            for tc in msg.tool_calls:
                for tool in self.tools:
                    if tool["definition"]["function"]["name"] == tc.function.name:
                        result = tool["func"](json.loads(tc.function.arguments))
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(result),
                        })
            # 让模型基于工具结果生成回答
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.messages,
            )
            msg = response.choices[0].message

        self.messages.append({"role": "assistant", "content": msg.content})
        return msg.content

# 使用
agent = SimpleAgent("你是一个天气查询助手，记住用户之前问过的城市")

agent.add_tool(
    "get_weather",
    "查询城市天气",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    lambda args: f"{args['city']}今天28度，晴天"
)

# 多轮对话
print(agent.chat("北京今天天气怎么样？"))
print(agent.chat("那上海呢？"))           # Agent记得在问天气
print(agent.chat("比一下这两个城市"))      # Agent记得之前查过的城市
```

### 实验 3：规划能力演示

```python
import openai

client = openai.OpenAI()

# 让模型展示规划能力
prompt = """
我要完成以下任务：组织一次公司团建活动

请用ReAct格式展示你的思考过程：
1. Thought: 分析需要做什么
2. Action: 决定下一步行动
3. Observation: 假设行动的结果
... 重复直到完成规划
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3,
)

print(response.choices[0].message.content)
```

---

## Agent 的核心组件

```
┌─────────────────────────────────────┐
│            AI Agent                 │
│                                     │
│  ┌─────────┐  ┌─────────────────┐  │
│  │   LLM   │  │   Tools         │  │
│  │ (大脑)   │  │ (手脚)          │  │
│  │         │  │ - API调用       │  │
│  │ 推理    │  │ - 代码执行      │  │
│  │ 规划    │  │ - 数据库查询    │  │
│  │ 决策    │  │ - 文件操作      │  │
│  └─────────┘  └─────────────────┘  │
│                                     │
│  ┌─────────┐  ┌─────────────────┐  │
│  │ Memory  │  │ Planning        │  │
│  │ (记忆)   │  │ (规划)          │  │
│  │         │  │                 │  │
│  │ 短期    │  │ 任务分解        │  │
│  │ 长期    │  │ 优先级排序      │  │
│  │ 工具结果│  │ 动态调整        │  │
│  └─────────┘  └─────────────────┘  │
└─────────────────────────────────────┘
```

| 组件 | 作用 | 典型实现 |
|------|------|---------|
| LLM | 大脑，负责推理和决策 | GPT-4, Claude, Llama |
| Tools | 执行具体操作 | Function Calling, 代码执行 |
| Memory | 存储历史信息 | 对话历史, 向量数据库 |
| Planning | 分解和规划任务 | CoT, Tree of Thoughts |

---

## Agent 框架对比

| 框架 | 特点 | 适用场景 |
|------|------|---------|
| **LangChain** | 生态最全，组件最多 | 通用Agent开发 |
| **LlamaIndex** | RAG见长，Agent能力也不错 | 知识库Agent |
| **AutoGen** | 微软出品，多Agent协作 | 复杂任务分解 |
| **CrewAI** | 角色扮演式多Agent | 模拟团队协作 |
| **原生 Function Calling** | 最简单直接 | 单工具调用场景 |

---

## 常见误区

### ❌ 误区 1：Agent 能解决所有问题
真相：Agent 适合**需要多步决策和工具调用**的场景。简单问答用普通 LLM 就够了。

### ❌ 误区 2：Agent 很可靠
真相：Agent 的每一步都可能出错，错误会累积。需要设置最大步数、超时、人工审核等兜底机制。

### ❌ 误区 3：Agent = AutoGPT
真相：AutoGPT 是早期 Agent 探索，容易陷入死循环。现代 Agent 更注重**受控的工具调用**，而不是完全自主。

---

## 延伸阅读

- [ReAct 论文](https://arxiv.org/abs/2210.03629) — Reasoning + Acting
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [LangChain Agent 文档](https://python.langchain.com/docs/modules/agents/)
- [Building Effective Agents - Anthropic](https://docs.anthropic.com/en/docs/build-with-claude/agentic-systems)

---

> 🎉 **恭喜！** 你已经完成了 AI 核心概念系列基础篇的全部内容。后续我们将继续深入模型架构、训练优化、安全治理等主题。敬请期待！
