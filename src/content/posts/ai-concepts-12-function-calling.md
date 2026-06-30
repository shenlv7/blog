---
title: "12 | Function Calling：让AI学会「打电话」调用外部工具"
description: "深度解析Function Calling技术，揭秘大模型如何精准调用外部API和工具，从「只会说」到「能动手」的关键一步"
pubDate: 2026-06-30
tags: ["AI核心概念", "Function Calling", "工具调用", "API", "Agent"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 12
---

# AI核心概念(12)：Function Calling——让AI学会「打电话」调用外部工具

> "大模型再聪明，也只是个「嘴强王者」。Function Calling让它长出了手。"

## 🎯 什么是Function Calling？

**Function Calling（函数调用）** 是让大语言模型能够**识别用户意图并调用预定义函数/API**的技术。它不是让模型自己执行代码，而是让模型**决定调用哪个函数、传什么参数**，然后由外部系统执行并返回结果。

### 一句话理解

| 能力 | 没有Function Calling | 有Function Calling |
|------|---------------------|-------------------|
| 查询天气 | "今天北京天气不错" （瞎猜） | 调用 `get_weather(city="北京")` → 返回真实数据 |
| 发邮件 | "你可以用SMTP发送邮件" （教你） | 调用 `send_email(to="...", subject="...", body="...")` （帮你发） |
| 数据库查询 | "SQL语法是SELECT..." （教你写） | 调用 `query_db(sql="SELECT...")` （直接查） |

**本质**：Function Calling = 意图识别 + 参数提取 + 结构化输出

## 🔬 为什么需要Function Calling？

### 问题：大模型的「三宗罪」

**1. 知识截止日期**

```text
用户：今天比特币价格多少？
LLM：我无法获取实时数据，我的训练数据截止到...
```

**2. 幻觉问题**

```text
用户：帮我查一下张三的订单状态
LLM：张三的订单号是12345，状态为已发货。  ← 编的！
```

**3. 无法执行动作**

```text
用户：帮我发一封邮件给老板
LLM：以下是邮件模板，请你手动发送...  ← 只能动嘴
```

### Function Calling的解法

```text
用户：今天北京天气怎么样？
         ↓
   LLM识别意图 → 需要调用 get_weather()
         ↓
   提取参数 → city="北京", date="今天"
         ↓
   外部系统执行函数 → 返回真实数据
         ↓
   LLM组织语言 → "今天北京晴，25°C，适合出行"
```

## 🏗️ 工作原理

### 完整流程图

```text
┌─────────────┐    ①用户提问    ┌─────────────┐
│             │ ───────────────→│             │
│    用户      │                 │    LLM      │
│             │←─────────────── │             │
└─────────────┘  ⑤自然语言回答  └──────┬──────┘
                                       │
                                  ②函数调用请求
                                  (JSON格式)
                                       │
                                       ▼
                                ┌─────────────┐
                                │  外部系统    │
                                │  (API/DB)   │
                                └──────┬──────┘
                                       │
                                  ④函数返回结果
                                       │
                                       ▼
                                ┌─────────────┐
                                │  编排层      │
                                │ (Orchestrator)│
                                └─────────────┘
```

### 步骤详解

**Step 1: 定义函数描述（System Prompt）**

```json
{
  "name": "get_weather",
  "description": "获取指定城市的当前天气信息",
  "parameters": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "城市名称，如'北京'、'上海'"
      },
      "date": {
        "type": "string",
        "description": "日期，格式YYYY-MM-DD，默认今天",
        "default": "today"
      }
    },
    "required": ["city"]
  }
}
```

**Step 2: LLM返回函数调用请求**

```json
{
  "function_call": {
    "name": "get_weather",
    "arguments": "{\"city\": \"北京\", \"date\": \"2026-06-30\"}"
  }
}
```

**Step 3: 外部系统执行函数**

```python
def get_weather(city, date="today"):
    # 调用真实天气API
    response = requests.get(f"https://api.weather.com/v1/{city}/{date}")
    return response.json()
```

**Step 4: 将结果返回给LLM**

```json
{
  "role": "function",
  "name": "get_weather",
  "content": "{\"city\": \"北京\", \"temp\": 25, \"condition\": \"晴\", \"humidity\": 45}"
}
```

**Step 5: LLM组织自然语言回答**

```text
今天北京天气晴朗，气温25°C，湿度45%，非常适合户外活动！
```

## 💻 代码实战：从零构建Function Calling系统

### 示例1：基础天气查询

```python
import json
import openai

# 定义可用函数
functions = [
    {
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
]

# 模拟天气API
def get_weather(city):
    weather_data = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 45},
        "上海": {"temp": 28, "condition": "多云", "humidity": 70},
        "深圳": {"temp": 32, "condition": "雷阵雨", "humidity": 85},
    }
    return weather_data.get(city, {"error": "未找到该城市"})

# 第一轮：LLM决定调用哪个函数
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "今天北京天气怎么样？"}],
    functions=functions,
    function_call="auto"
)

# 解析函数调用
message = response.choices[0].message
if message.function_call:
    func_name = message.function_call.name
    args = json.loads(message.function_call.arguments)
    
    # 执行函数
    result = get_weather(args["city"])
    
    # 第二轮：将结果返回给LLM
    final_response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": "今天北京天气怎么样？"},
            message,
            {"role": "function", "name": func_name, "content": json.dumps(result)}
        ]
    )
    print(final_response.choices[0].message.content)
```

### 示例2：多函数并行调用

```python
functions = [
    {
        "name": "get_weather",
        "description": "获取天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }
    },
    {
        "name": "search_restaurants",
        "description": "搜索餐厅",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "cuisine": {"type": "string", "description": "菜系"}
            },
            "required": ["city"]
        }
    },
    {
        "name": "book_taxi",
        "description": "叫出租车",
        "parameters": {
            "type": "object",
            "properties": {
                "pickup": {"type": "string", "description": "上车地点"},
                "destination": {"type": "string", "description": "目的地"}
            },
            "required": ["pickup", "destination"]
        }
    }
]

# 用户说："帮我看看北京天气，推荐几家火锅店，然后叫车去王府井"
# LLM可以一次性返回多个函数调用！
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "帮我看看北京天气，推荐几家火锅店，然后叫车去王府井"}],
    functions=functions,
    function_call="auto"
)

# 返回：
# function_call_1: get_weather(city="北京")
# function_call_2: search_restaurants(city="北京", cuisine="火锅")
# function_call_3: book_taxi(pickup="当前位置", destination="王府井")
```

### 示例3：流式Function Calling

```python
# 流式场景：实时展示函数调用过程
stream = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "查一下上海天气"}],
    functions=functions,
    stream=True
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.function_call:
        # 实时拼接函数名和参数
        if delta.function_call.name:
            print(f"调用函数: {delta.function_call.name}")
        if delta.function_call.arguments:
            print(f"参数: {delta.function_call.arguments}", end="")
```

## 🔧 函数定义最佳实践

### 1. 描述要清晰

```json
// ❌ 差的描述
{
  "name": "query",
  "description": "查询数据"
}

// ✅ 好的描述
{
  "name": "query_user_orders",
  "description": "查询用户的订单列表，支持按状态和时间范围筛选。返回订单号、商品名、金额和状态。",
  "parameters": {
    "type": "object",
    "properties": {
      "user_id": {"type": "string", "description": "用户唯一标识"},
      "status": {
        "type": "string",
        "enum": ["pending", "paid", "shipped", "completed", "cancelled"],
        "description": "订单状态筛选"
      },
      "start_date": {"type": "string", "description": "开始日期，格式YYYY-MM-DD"},
      "end_date": {"type": "string", "description": "结束日期，格式YYYY-MM-DD"},
      "limit": {"type": "integer", "description": "返回数量上限，默认20", "default": 20}
    },
    "required": ["user_id"]
  }
}
```

### 2. 参数设计原则

| 原则 | 说明 | 示例 |
|------|------|------|
| **类型明确** | 用JSON Schema严格定义 | `"type": "integer"` 而非 `"type": "string"` |
| **枚举约束** | 有限选项用enum | `"enum": ["asc", "desc"]` |
| **默认值** | 非必填参数提供默认值 | `"default": 20` |
| **格式说明** | 日期、邮箱等明确格式 | `"format": "YYYY-MM-DD"` |
| **范围约束** | 数值加min/max | `"minimum": 1, "maximum": 100` |

### 3. 错误处理模式

```python
def safe_function_call(func_name, args):
    """安全的函数调用包装器"""
    try:
        result = FUNCTIONS[func_name](**args)
        return {
            "success": True,
            "data": result
        }
    except ValueError as e:
        return {
            "success": False,
            "error": f"参数错误: {str(e)}",
            "hint": "请检查参数格式是否正确"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"执行失败: {str(e)}",
            "hint": "系统暂时不可用，请稍后重试"
        }
```

## ⚡ 高级模式

### 1. 链式调用（Sequential Calls）

```text
用户：帮我订明天北京到上海的高铁，到了之后叫车去外滩

Step 1: search_trains(from="北京", to="上海", date="2026-07-01")
  → 返回车次列表

Step 2: book_train(train_id="G1234", date="2026-07-01")
  → 返回订单确认

Step 3: book_taxi(pickup="上海虹桥站", destination="外滩")
  → 返回叫车结果
```

### 2. 条件分支（Conditional Calls）

```python
# LLM根据上下文决定是否需要调用函数
if user_query_requires_realtime_data:
    # 需要实时数据 → 调用函数
    function_call = llm.decide_function(messages, functions)
else:
    # 纯知识问题 → 直接回答
    answer = llm.generate(messages)
```

### 3. 函数编排（Function Orchestration）

```python
class FunctionOrchestrator:
    """复杂任务的函数编排器"""
    
    def __init__(self):
        self.functions = {}
        self.call_history = []
    
    def register(self, name, func, schema):
        self.functions[name] = {
            "func": func,
            "schema": schema
        }
    
    def execute_plan(self, plan):
        """执行LLM规划的调用序列"""
        results = []
        for step in plan:
            func = self.functions[step["name"]]["func"]
            result = func(**step["args"])
            results.append({
                "step": step,
                "result": result
            })
            # 将中间结果注入上下文，供后续步骤使用
        return results
```

## 🛡️ 安全与边界

### 必须防范的风险

| 风险 | 说明 | 对策 |
|------|------|------|
| **注入攻击** | 用户在参数中注入恶意指令 | 输入清洗 + 参数白名单 |
| **权限越界** | 调用超出用户权限的函数 | 执行前权限校验 |
| **数据泄露** | 函数返回敏感信息给LLM | 结果脱敏 + 最小数据原则 |
| **无限循环** | 函数调用形成死循环 | 最大调用次数限制 |
| **费用失控** | 高频调用付费API | 调用频率限制 + 费用监控 |

### 安全编码模板

```python
SECURITY_CONFIG = {
    "max_calls_per_turn": 10,       # 单轮最大调用次数
    "max_calls_per_minute": 60,     # 每分钟最大调用次数
    "allowed_functions": ["get_weather", "search"],  # 白名单
    "require_confirmation": ["send_email", "delete_data"],  # 需要用户确认
    "sensitive_params_mask": ["password", "token", "secret"]  # 脱敏字段
}

def secure_call(func_name, args, user_context):
    """带安全检查的函数调用"""
    # 1. 白名单检查
    if func_name not in SECURITY_CONFIG["allowed_functions"]:
        raise PermissionError(f"函数 {func_name} 不在允许列表中")
    
    # 2. 频率限制
    if rate_limiter.exceeded(user_context["user_id"]):
        raise RateLimitError("调用频率超限")
    
    # 3. 敏感操作确认
    if func_name in SECURITY_CONFIG["require_confirmation"]:
        if not user_context.get("confirmed"):
            return {"need_confirmation": True, "message": f"确认执行 {func_name}？"}
    
    # 4. 参数脱敏
    sanitized_args = mask_sensitive(args)
    
    # 5. 执行
    return execute(func_name, sanitized_args)
```

## 📊 各平台Function Calling对比

| 特性 | OpenAI GPT-4 | Claude | Gemini | 开源模型 |
|------|-------------|--------|--------|---------|
| 并行调用 | ✅ 支持 | ✅ 支持 | ✅ 支持 | ⚠️ 部分 |
| 流式调用 | ✅ 支持 | ✅ 支持 | ✅ 支持 | ⚠️ 需自实现 |
| 严格模式 | ✅ JSON Schema | ⚠️ 描述引导 | ✅ 支持 | ❌ 需后处理 |
| 嵌套参数 | ✅ 支持 | ✅ 支持 | ✅ 支持 | ⚠️ 复杂 |
| 自动调用 | ✅ auto/none/required | ✅ auto/none | ✅ 支持 | ⚠️ 需实现 |

## 🎯 实际应用场景

### 1. 智能客服系统

```text
用户：我的订单#12345三天了还没发货，帮我查一下

→ LLM调用:
  1. query_order(order_id="12345") → 获取订单详情
  2. check_logistics(order_id="12345") → 查物流状态

→ 回复: "您的订单#12345已于6月28日从深圳仓库发出，
         目前在转运中，预计明天到达。物流单号：SF1234567890。"
```

### 2. 数据分析助手

```text
用户：帮我分析上个月北京和上海的销售对比

→ LLM调用:
  1. query_sales(region="北京", date_range="2026-05-01~2026-05-31")
  2. query_sales(region="上海", date_range="2026-05-01~2026-05-31")
  3. generate_chart(data=..., type="bar")

→ 回复: "上月北京销售额120万，上海98万。北京领先22%，
         主要差距来自电子产品品类。已生成对比图表如下..."
```

### 3. 智能家居控制

```text
用户：我要睡觉了，帮我把灯关了，空调调到26度

→ LLM调用:
  1. control_device(device="all_lights", action="off")
  2. control_device(device="bedroom_ac", action="set_temp", value=26)

→ 回复: "已为您关闭所有灯光，卧室空调已调至26度。晚安！"
```

## 💡 核心要点总结

**Function Calling的本质**：
- ❌ 不是让LLM执行代码
- ✅ 是让LLM决定「调用什么函数」「传什么参数」

**三个关键环节**：
1. **函数描述**：告诉LLM有哪些函数可用（JSON Schema）
2. **意图识别**：LLM判断是否需要调用函数
3. **参数提取**：从用户输入中提取结构化参数

**设计原则**：
- 函数描述要像写API文档一样清晰
- 参数约束要严格（类型、枚举、范围）
- 错误处理要完善（网络超时、参数非法、权限不足）
- 安全防护要到位（注入防护、权限校验、频率限制）

---

> **系列回顾**：AI核心概念系列12期，从Prompt到Token，从LLM到Agent，从思维链到函数调用。每一个概念都是AI工程化的基石。掌握了这些，你已经具备了构建AI应用的核心能力。
>
> **下期预告**：新系列即将开启，敬请期待！

---

*本文为AI核心概念系列第12期。*
*作者：赛博阿漆 | 发布日期：2026-06-30*
