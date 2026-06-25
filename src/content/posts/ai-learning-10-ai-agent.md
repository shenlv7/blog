---
title: "AI学习之路(第10期)：AI Agent——从对话到行动"
slug: ai-learning-10-ai-agent
pubDate: 2026-06-25
description: "从ChatGPT到自主Agent，探索AI如何从被动回答进化为主动执行，附实战代码"
image: "https://images.unsplash.com/photo-1677442136019-21780ecad995?w=1200"
tags: ["AI学习", "AI Agent", "工具调用", "ReAct", "LLM", "深度学习"]
series: "AI学习之路"
episode: 10
---

# AI学习之路(第10期)：AI Agent——从对话到行动

![AI Agent](https://images.unsplash.com/photo-1677442136019-21780ecad995?w=800)

> "The best interface is no interface." —— Golden Krishna

## 前言

上一期我们深入探索了大语言模型——理解了它如何通过"预测下一个词"涌现出惊人的能力。但你有没有想过：**如果LLM不仅能说话，还能做事呢？**

这就是 **AI Agent** 的核心命题。从ChatGPT的插件系统到AutoGPT的自主任务执行，从Cursor的代码助手到Devin的软件工程AI，Agent正在重新定义人与AI的交互范式。

这一期，我们不谈概念炒作，只聊本质：Agent到底是什么？它怎么工作？怎么从零搭建一个？

---

## 1. 什么是AI Agent？

### 1.1 从Chatbot到Agent

传统Chatbot是**被动**的——你问一句，它答一句。Agent是**主动**的——你给一个目标，它自己规划、执行、调整。

```
Chatbot模式：
  用户: "帮我查一下北京明天的天气"
  AI:   "北京明天晴，25°C"
  （结束）

Agent模式：
  用户: "帮我规划明天的北京出行"
  AI:   思考 → 查天气 → 查交通 → 查景点 → 生成行程 → 反馈给用户
  （自主完成多步推理和工具调用）
```

### 1.2 Agent的核心定义

一个AI Agent至少包含四个核心组件：

```
┌─────────────────────────────────────────────┐
│                  AI Agent                    │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │  LLM大脑  │  │  记忆系统  │  │  工具箱   │  │
│  │ (推理决策) │  │ (上下文/长 │  │ (API/代码 │  │
│  │          │  │  期记忆)  │  │  /搜索)   │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│         │            │            │          │
│         └────────────┼────────────┘          │
│                      ▼                       │
│              ┌──────────────┐                │
│              │   规划引擎    │                │
│              │ (Plan/ReAct) │                │
│              └──────────────┘                │
└─────────────────────────────────────────────┘
```

| 组件 | 作用 | 类比 |
|------|------|------|
| **LLM大脑** | 推理、决策、理解意图 | 人的大脑 |
| **记忆系统** | 短期上下文 + 长期知识 | 人的笔记本 |
| **工具箱** | 执行具体操作 | 人的工具箱 |
| **规划引擎** | 分解任务、制定计划 | 人的项目管理能力 |

---

## 2. ReAct：Agent的核心推理框架

### 2.1 什么是ReAct？

ReAct（Reasoning + Acting）是目前最主流的Agent推理框架，由Yao et al. 2022年提出。核心思想：**让LLM交替进行"思考"和"行动"**。

```
Thought: 用户想知道明天北京的天气，我需要调用天气API
Action: call_weather_api(city="北京", date="明天")
Observation: 北京明天晴，25°C，微风
Thought: 我已经获取了天气信息，可以回复用户了
Answer: 北京明天天气晴朗，气温25°C，微风，适合出行。
```

### 2.2 ReAct vs 纯推理

| 模式 | 流程 | 优点 | 缺点 |
|------|------|------|------|
| **纯推理** | Thought → Answer | 简单快速 | 无法获取实时信息 |
| **ReAct** | Thought → Action → Observation → ... → Answer | 能使用工具、获取实时信息 | 较慢、需要更多token |

### 2.3 代码实现：ReAct循环

```python
import openai
import json

class ReActAgent:
    """一个简单的ReAct Agent"""
    
    def __init__(self, model="gpt-4"):
        self.client = openai.OpenAI()
        self.model = model
        self.tools = {}
        self.max_steps = 10
    
    def register_tool(self, name, func, description):
        """注册工具"""
        self.tools[name] = {
            "function": func,
            "description": description
        }
    
    def think(self, task, history):
        """LLM思考：决定下一步行动"""
        tools_desc = "\n".join(
            f"- {name}: {tool['description']}" 
            for name, tool in self.tools.items()
        )
        
        prompt = f"""你是一个ReAct Agent。你可以使用以下工具：
{tools_desc}

任务：{task}

历史步骤：
{history}

请按以下格式输出：
Thought: [你的思考]
Action: [工具名称(参数)]
或
Answer: [最终回答]

请选择下一步行动："""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    
    def act(self, action_str):
        """执行工具调用"""
        # 解析 Action: tool_name(param1=value1, param2=value2)
        import re
        match = re.match(r'(\w+)\((.*)\)', action_str)
        if not match:
            return "工具调用格式错误"
        
        tool_name = match.group(1)
        if tool_name not in self.tools:
            return f"未知工具: {tool_name}"
        
        # 简单参数解析
        args_str = match.group(2)
        kwargs = {}
        if args_str:
            for arg in args_str.split(','):
                k, v = arg.strip().split('=')
                kwargs[k.strip()] = v.strip().strip('"\'')
        
        return self.tools[tool_name]["function"](**kwargs)
    
    def run(self, task):
        """执行任务"""
        history = ""
        
        for step in range(self.max_steps):
            response = self.think(task, history)
            print(f"\n--- Step {step + 1} ---")
            print(response)
            
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
                return answer
            
            if "Action:" in response:
                action = response.split("Action:")[-1].strip()
                observation = self.act(action)
                history += f"\n{response}\nObservation: {observation}\n"
        
        return "达到最大步数限制，任务未完成"
```

---

## 3. 工具调用（Function Calling）

### 3.1 OpenAI Function Calling

现代LLM（如GPT-4）原生支持工具调用，不需要手动解析：

```python
import openai
import json

client = openai.OpenAI()

# 定义工具
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 工具实现
def get_weather(city):
    # 实际项目中调用真实API
    weather_data = {"北京": "晴 25°C", "上海": "多云 28°C"}
    return weather_data.get(city, f"{city}天气未知")

def search_web(query):
    return f"搜索结果: 关于'{query}'的相关信息..."

# 工具映射
available_tools = {
    "get_weather": get_weather,
    "search_web": search_web
}

# 对话循环
messages = [
    {"role": "user", "content": "北京和上海明天天气怎么样？"}
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    tools=tools,
    tool_choice="auto"
)

# 处理工具调用
while response.choices[0].message.tool_calls:
    tool_calls = response.choices[0].message.tool_calls
    messages.append(response.choices[0].message)
    
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        func_args = json.loads(tool_call.function.arguments)
        result = available_tools[func_name](**func_args)
        
        print(f"调用工具: {func_name}({func_args})")
        print(f"返回结果: {result}")
        
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        tools=tools,
        tool_choice="auto"
    )

print(response.choices[0].message.content)
```

### 3.2 工具调用的执行流程

```
用户: "北京天气怎么样？"
        │
        ▼
┌───────────────────┐
│   LLM 推理        │
│   "需要查天气"     │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  生成工具调用       │
│  get_weather(北京) │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  执行函数代码       │
│  返回: "晴 25°C"  │
└───────┬───────────┘
        │
        ▼
┌───────────────────┐
│  LLM 生成最终回答   │
│  "北京今天晴..."    │
└───────────────────┘
```

---

## 4. 记忆系统设计

### 4.1 短期记忆：上下文窗口

LLM的上下文窗口就是它的"工作记忆"。但窗口有限，需要管理：

```python
class ConversationMemory:
    """对话记忆管理"""
    
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add(self, role, content):
        self.messages.append({"role": role, "content": content})
        self._trim()
    
    def _trim(self):
        """保持在token限制内，保留系统消息和最近对话"""
        total = sum(len(m["content"]) for m in self.messages)
        while total > self.max_tokens and len(self.messages) > 2:
            # 保留第一条系统消息，删除最旧的对话
            removed = self.messages.pop(1)
            total -= len(removed["content"])
    
    def get_messages(self):
        return self.messages.copy()
```

### 4.2 长期记忆：向量数据库

```python
import chromadb
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    """基于向量数据库的长期记忆"""
    
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("agent_memory")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def remember(self, text, metadata=None):
        """存储记忆"""
        embedding = self.encoder.encode(text).tolist()
        self.collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[f"mem_{self.collection.count()}"],
            metadatas=[metadata or {}]
        )
    
    def recall(self, query, top_k=3):
        """检索相关记忆"""
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        return results["documents"][0]

# 使用示例
memory = LongTermMemory()
memory.remember("用户喜欢简洁的回答风格", {"type": "preference"})
memory.remember("用户是Python开发者", {"type": "profile"})

# 后续对话中检索
relevant = memory.recall("用户的技术背景")
print(relevant)  # ["用户是Python开发者"]
```

---

## 5. 经典Agent架构

### 5.1 Plan-and-Execute

先制定计划，再逐步执行：

```python
class PlanAndExecuteAgent:
    """规划-执行 Agent"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
    
    def plan(self, task):
        """制定执行计划"""
        prompt = f"""任务: {task}
        
请制定一个分步执行计划，每步使用一个工具：
1. [工具名] [具体操作]
2. [工具名] [具体操作]
..."""
        
        response = self.llm(prompt)
        return self._parse_plan(response)
    
    def execute_step(self, step, context):
        """执行单个步骤"""
        # 根据步骤选择工具并执行
        tool_name = step["tool"]
        result = self.tools[tool_name](**step["args"])
        return result
    
    def run(self, task):
        """完整执行流程"""
        plan = self.plan(task)
        print(f"计划: {plan}")
        
        context = []
        for i, step in enumerate(plan):
            print(f"执行步骤 {i+1}: {step}")
            result = self.execute_step(step, context)
            context.append({"step": step, "result": result})
        
        return context[-1]["result"]
```

### 5.2 Multi-Agent协作

多个Agent各司其职，协作完成复杂任务：

```python
class MultiAgentSystem:
    """多Agent协作系统"""
    
    def __init__(self):
        self.agents = {}
    
    def register_agent(self, name, agent):
        self.agents[name] = agent
    
    def route(self, task):
        """根据任务类型路由到合适的Agent"""
        # 简化版：关键词匹配
        if "代码" in task or "编程" in task:
            return self.agents["coder"]
        elif "搜索" in task or "查找" in task:
            return self.agents["researcher"]
        elif "写作" in task or "文案" in task:
            return self.agents["writer"]
        else:
            return self.agents["general"]
    
    def run(self, task):
        agent = self.route(task)
        return agent.run(task)

# 创建专业化Agent
coder_agent = ReActAgent()
coder_agent.register_tool("execute_python", execute_code, "执行Python代码")

researcher_agent = ReActAgent()
researcher_agent.register_tool("search", web_search, "搜索互联网")

writer_agent = ReActAgent()
writer_agent.register_tool("write", write_file, "写入文件")

# 组装多Agent系统
system = MultiAgentSystem()
system.register_agent("coder", coder_agent)
system.register_agent("researcher", researcher_agent)
system.register_agent("writer", writer_agent)
```

---

## 6. 实战：搭建一个完整的Agent

### 6.1 需求分析

搭建一个**个人助理Agent**，能够：
- 查询天气、新闻
- 搜索网络信息
- 管理待办事项
- 回答问题

### 6.2 完整实现

```python
import openai
import json
import requests
from datetime import datetime

class PersonalAssistantAgent:
    """个人助理Agent"""
    
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        self.todo_list = []
        self.conversation_history = []
        
        # 定义工具
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "获取城市天气",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "城市名"}
                        },
                        "required": ["city"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_todo",
                    "description": "添加待办事项",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {"type": "string", "description": "待办内容"},
                            "priority": {
                                "type": "string", 
                                "enum": ["high", "medium", "low"],
                                "description": "优先级"
                            }
                        },
                        "required": ["task"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_todos",
                    "description": "查看待办列表",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
        
        self.available_functions = {
            "get_weather": self._get_weather,
            "add_todo": self._add_todo,
            "get_todos": self._get_todos
        }
    
    def _get_weather(self, city):
        # 模拟天气API
        weather_db = {
            "北京": "☀️ 晴 25°C，微风",
            "上海": "⛅ 多云 28°C，东南风",
            "深圳": "🌤️ 晴间多云 31°C，湿度75%",
            "杭州": "🌧️ 小雨 22°C，东北风"
        }
        return weather_db.get(city, f"暂无{city}的天气数据")
    
    def _add_todo(self, task, priority="medium"):
        todo = {
            "task": task,
            "priority": priority,
            "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "done": False
        }
        self.todo_list.append(todo)
        return f"✅ 已添加待办: [{priority}] {task}"
    
    def _get_todos(self):
        if not self.todo_list:
            return "📋 待办列表为空"
        
        result = "📋 待办列表:\n"
        for i, todo in enumerate(self.todo_list, 1):
            status = "✅" if todo["done"] else "⬜"
            result += f"{i}. {status} [{todo['priority']}] {todo['task']}\n"
        return result
    
    def chat(self, user_message):
        """与Agent对话"""
        self.conversation_history.append({
            "role": "user", 
            "content": user_message
        })
        
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=self.conversation_history,
            tools=self.tools,
            tool_choice="auto"
        )
        
        # 处理工具调用
        while response.choices[0].message.tool_calls:
            tool_calls = response.choices[0].message.tool_calls
            self.conversation_history.append(response.choices[0].message)
            
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)
                
                print(f"🔧 调用工具: {func_name}({func_args})")
                result = self.available_functions[func_name](**func_args)
                
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=self.conversation_history,
                tools=self.tools,
                tool_choice="auto"
            )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        return assistant_message

# 使用示例
agent = PersonalAssistantAgent("your-api-key")

# 对话
print(agent.chat("北京今天天气怎么样？"))
print(agent.chat("帮我添加一个待办：学习Agent开发，优先级高"))
print(agent.chat("看看我的待办列表"))
print(agent.chat("再加一个：写博客文章"))
print(agent.chat("我的待办列表是什么？"))
```

---

## 7. Agent的挑战与未来

### 7.1 当前挑战

| 挑战 | 描述 |
|------|------|
| **幻觉问题** | Agent可能基于错误信息采取行动 |
| **安全性** | 执行工具可能带来风险（如删除文件） |
| **成本** | 多轮推理消耗大量token |
| **可控性** | 自主Agent的行为难以完全预测 |

### 7.2 安全实践

```python
# 工具执行前的安全检查
class SafeAgent(ReActAgent):
    def act(self, action_str):
        # 危险操作白名单检查
        dangerous_actions = ["delete_file", "execute_command", "send_email"]
        
        tool_name = action_str.split('(')[0]
        if tool_name in dangerous_actions:
            print(f"⚠️ 危险操作，需要用户确认: {tool_name}")
            confirm = input("确认执行？(y/n): ")
            if confirm.lower() != 'y':
                return "操作已取消"
        
        return super().act(action_str)
```

### 7.3 未来趋势

```
2023  Agent概念爆发（AutoGPT、BabyAGI）
2024  工具调用标准化（OpenAI Function Calling、MCP）
2025  多Agent协作框架成熟（CrewAI、AutoGen）
2026  Agent操作系统化 → 你现在在这里
未来  AGI？Agent即通用智能体
```

---

## 总结

| 概念 | 核心要点 |
|------|---------|
| **Agent** | 从被动对话到主动执行 |
| **ReAct** | 推理与行动交替的执行框架 |
| **Function Calling** | LLM原生的工具调用能力 |
| **记忆系统** | 短期上下文 + 长期向量存储 |
| **Multi-Agent** | 多个专业Agent协作 |

**下一步：** 本期是AI学习之路的第10期，我们从"理解AI"进入了"让AI行动"的阶段。下一期，我们将探讨一个至关重要的话题——AI伦理与安全。

---

## 参考资料

- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)
- [Function Calling Guide - OpenAI](https://platform.openai.com/docs/guides/function-calling)
- [Building Effective Agents - Anthropic](https://docs.anthropic.com/claude/docs/building-effective-agents)
