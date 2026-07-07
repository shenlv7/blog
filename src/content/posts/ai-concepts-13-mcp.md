---
title: "13 | MCP：给AI装上「万能插头」"
description: "深度解析Model Context Protocol，揭秘AI如何通过统一接口连接万物，从「信息孤岛」到「万物互联」的关键协议"
pubDate: 2026-07-07
tags: ["AI核心概念", "MCP", "Model Context Protocol", "协议", "工具集成"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 13
---

# AI核心概念(13)：MCP——给AI装上「万能插头」

> "Function Calling让AI能打电话，MCP让AI能接所有电话。"

## 🎯 什么是MCP？

**MCP（Model Context Protocol，模型上下文协议）** 是 Anthropic 在 2024 年底提出的**开放协议标准**，旨在为大语言模型提供**统一的工具和数据接入方式**。

### 一句话理解

| 场景 | 没有MCP | 有MCP |
|------|--------|-------|
| 连接数据库 | 每个AI平台写一套适配代码 | 一个MCP Server，所有AI都能用 |
| 读取文件 | 绑定特定平台的文件接口 | 标准协议，任何MCP客户端都能调 |
| 调用API | 重复造轮子 | 写一次MCP Server，处处可用 |

**本质**：MCP = AI世界的USB接口

就像USB让鼠标、键盘、摄像头都能即插即用，MCP让数据库、文件系统、API都能被AI统一调用。

## 🔬 为什么需要MCP？

### 问题：AI工具集成的「N×M困境」

```
没有MCP的世界：

AI平台A ──→ 适配 ──→ 工具1
AI平台A ──→ 适配 ──→ 工具2
AI平台A ──→ 适配 ──→ 工具3
AI平台B ──→ 适配 ──→ 工具1  ← 重复开发！
AI平台B ──→ 适配 ──→ 工具2
AI平台C ──→ 适配 ──→ 工具1  ← 又重复！

N个平台 × M个工具 = N×M套适配代码
```

```
有MCP的世界：

AI平台A ──┐
AI平台B ──┼──→ MCP协议 ──┬──→ MCP Server (工具1)
AI平台C ──┘              ├──→ MCP Server (工具2)
                         └──→ MCP Server (工具3)

N个平台 + M个工具 = N+M套代码
```

### 真实痛点

**场景1：重复开发**
```
OpenAI的Function Calling格式：
{
  "type": "function",
  "function": { "name": "...", "parameters": {...} }
}

Claude的Tool Use格式：
{
  "type": "tool_use",
  "name": "...",
  "input": {...}
}

// 同一个天气查询工具，要写两套适配！
```

**场景2：能力割裂**
```
ChatGPT能联网搜索，但不能读你的本地文件
Cursor能读本地代码，但不能直接查数据库
Claude能分析文档，但不能操作你的GitHub

每个AI都有自己的能力边界，用户被锁在各自的围墙里
```

**场景3：安全与权限**
```
把API Key直接给AI？→ 风险太大
让AI直接操作数据库？→ 没有权限控制
第三方插件可信吗？→ 没有标准审计机制
```

### MCP的解法

```text
统一协议 → 标准接入 → 权限隔离 → 生态共享

1. 协议统一：一套标准，所有AI通用
2. 能力解耦：工具独立于AI平台存在
3. 安全可控：权限在Server端控制，不暴露原始凭证
4. 生态复用：社区共建MCP Server库
```

## 🏗️ 架构设计

### 核心组件

```text
┌─────────────────────────────────────────────────────────┐
│                    MCP Host                              │
│              (AI应用，如Cursor/Claude)                    │
│                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │ MCP Client  │  │ MCP Client  │  │ MCP Client  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
└─────────┼───────────────┼───────────────┼───────────────┘
          │               │               │
          ▼               ▼               ▼
   ┌──────────┐    ┌──────────┐    ┌──────────┐
   │ MCP      │    │ MCP      │    │ MCP      │
   │ Server A │    │ Server B │    │ Server C │
   │(数据库)  │    │(文件系统)│    │(GitHub)  │
   └──────────┘    └──────────┘    └──────────┘
```

**MCP Host**：运行AI的应用程序（如Cursor、Claude Desktop）
**MCP Client**：Host内部管理与Server连接的组件
**MCP Server**：提供具体能力的服务（工具、数据、提示词）

### 三大核心能力

MCP Server可以暴露三类能力：

```text
1. Tools（工具）— AI可以调用的函数
   例：查询数据库、发送邮件、创建文件

2. Resources（资源）— AI可以读取的数据
   例：文件内容、数据库schema、API响应

3. Prompts（提示词）— 预定义的提示模板
   例：代码审查模板、数据分析流程
```

### 通信协议

MCP支持两种传输方式：

```text
1. stdio（标准输入输出）
   - Server作为子进程运行
   - 通过stdin/stdout通信
   - 适合本地工具

   Host ──stdin──→ Server进程 ──stdout──→ Host

2. SSE（Server-Sent Events）
   - Server作为HTTP服务运行
   - 通过HTTP通信
   - 适合远程服务

   Host ──HTTP请求──→ Server (HTTP服务)
   Host ←──SSE流──── Server
```

## 💻 代码实战

### 从零构建一个MCP Server

#### Python版本

```python
# weather_mcp_server.py
"""
一个简单的天气查询MCP Server
"""

import json
import sys
from typing import Any

# MCP Server 核心实现
class MCPServer:
    def __init__(self):
        self.tools = {}
        self.resources = {}
    
    def tool(self, name: str, description: str, input_schema: dict):
        """装饰器：注册工具"""
        def decorator(func):
            self.tools[name] = {
                "name": name,
                "description": description,
                "inputSchema": input_schema,
                "handler": func
            }
            return func
        return decorator
    
    def resource(self, uri: str, name: str, description: str):
        """装饰器：注册资源"""
        def decorator(func):
            self.resources[uri] = {
                "uri": uri,
                "name": name,
                "description": description,
                "handler": func
            }
            return func
        return decorator
    
    def handle_request(self, request: dict) -> dict:
        """处理JSON-RPC请求"""
        method = request.get("method")
        params = request.get("params", {})
        req_id = request.get("id")
        
        if method == "initialize":
            return self._handle_initialize(req_id)
        elif method == "tools/list":
            return self._handle_tools_list(req_id)
        elif method == "tools/call":
            return self._handle_tools_call(req_id, params)
        elif method == "resources/list":
            return self._handle_resources_list(req_id)
        elif method == "resources/read":
            return self._handle_resources_read(req_id, params)
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Unknown method: {method}"}}
    
    def _handle_initialize(self, req_id):
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {"listChanged": False},
                    "resources": {"subscribe": False, "listChanged": False}
                },
                "serverInfo": {
                    "name": "weather-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_tools_list(self, req_id):
        tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"]
            }
            for t in self.tools.values()
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
    
    def _handle_tools_call(self, req_id, params):
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        if tool_name not in self.tools:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown tool: {tool_name}"}}
        
        try:
            result = self.tools[tool_name]["handler"](**arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]
                }
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"错误: {str(e)}"}],
                    "isError": True
                }
            }
    
    def _handle_resources_list(self, req_id):
        resources = [
            {
                "uri": r["uri"],
                "name": r["name"],
                "description": r["description"]
            }
            for r in self.resources.values()
        ]
        return {"jsonrpc": "2.0", "id": req_id, "result": {"resources": resources}}
    
    def _handle_resources_read(self, req_id, params):
        uri = params.get("uri")
        if uri not in self.resources:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Unknown resource: {uri}"}}
        
        result = self.resources[uri]["handler"]()
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "contents": [{"uri": uri, "text": result}]
            }
        }
    
    def run(self):
        """stdio模式运行"""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self.handle_request(request)
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                continue

# 创建Server实例
server = MCPServer()

# 注册工具
@server.tool(
    name="get_weather",
    description="获取指定城市的当前天气信息",
    input_schema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称，如'北京'、'上海'"
            }
        },
        "required": ["city"]
    }
)
def get_weather(city: str) -> dict:
    """查询天气（模拟数据）"""
    weather_data = {
        "北京": {"temp": 25, "condition": "晴", "humidity": 45, "wind": "北风3级"},
        "上海": {"temp": 28, "condition": "多云", "humidity": 70, "wind": "东风2级"},
        "深圳": {"temp": 32, "condition": "雷阵雨", "humidity": 85, "wind": "南风1级"},
        "杭州": {"temp": 26, "condition": "阴", "humidity": 65, "wind": "微风"},
    }
    if city not in weather_data:
        return {"error": f"未找到城市: {city}"}
    
    data = weather_data[city]
    return {
        "city": city,
        "temperature": data["temp"],
        "condition": data["condition"],
        "humidity": data["humidity"],
        "wind": data["wind"],
        "unit": "°C"
    }

@server.tool(
    name="search_flights",
    description="搜索航班信息",
    input_schema={
        "type": "object",
        "properties": {
            "departure": {"type": "string", "description": "出发城市"},
            "arrival": {"type": "string", "description": "到达城市"},
            "date": {"type": "string", "description": "日期，格式YYYY-MM-DD"}
        },
        "required": ["departure", "arrival"]
    }
)
def search_flights(departure: str, arrival: str, date: str = "today") -> dict:
    """搜索航班（模拟数据）"""
    return {
        "flights": [
            {"flight": "CA1234", "departure": departure, "arrival": arrival, 
             "time": "08:00-11:00", "price": 1280, "status": "有票"},
            {"flight": "MU5678", "departure": departure, "arrival": arrival, 
             "time": "10:30-13:30", "price": 980, "status": "有票"},
            {"flight": "CZ9012", "departure": departure, "arrival": arrival, 
             "time": "14:00-17:00", "price": 1150, "status": "余3张"},
        ],
        "date": date
    }

# 注册资源
@server.resource(
    uri="weather://cities",
    name="支持的城市列表",
    description="返回所有支持天气查询的城市"
)
def get_cities() -> str:
    return json.dumps({
        "cities": ["北京", "上海", "深圳", "杭州", "广州", "成都", "武汉", "西安"],
        "total": 8
    }, ensure_ascii=False)

if __name__ == "__main__":
    server.run()
```

#### TypeScript版本

```typescript
// mcp-server.ts
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

// 创建MCP Server
const server = new Server(
  { name: "code-tools", version: "1.0.0" },
  { capabilities: { tools: {} } }
);

// 定义工具
const tools = [
  {
    name: "read_file",
    description: "读取文件内容",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "文件路径" }
      },
      required: ["path"]
    }
  },
  {
    name: "write_file",
    description: "写入文件内容",
    inputSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "文件路径" },
        content: { type: "string", description: "文件内容" }
      },
      required: ["path", "content"]
    }
  },
  {
    name: "run_command",
    description: "执行Shell命令",
    inputSchema: {
      type: "object",
      properties: {
        command: { type: "string", description: "要执行的命令" }
      },
      required: ["command"]
    }
  }
];

// 处理工具列表请求
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools };
});

// 处理工具调用请求
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  
  try {
    let result: string;
    
    switch (name) {
      case "read_file": {
        const fs = await import("fs/promises");
        const content = await fs.readFile(args.path as string, "utf-8");
        result = content;
        break;
      }
      case "write_file": {
        const fs = await import("fs/promises");
        await fs.writeFile(args.path as string, args.content as string);
        result = `文件已写入: ${args.path}`;
        break;
      }
      case "run_command": {
        const { execSync } = await import("child_process");
        result = execSync(args.command as string, { encoding: "utf-8" });
        break;
      }
      default:
        throw new Error(`未知工具: ${name}`);
    }
    
    return {
      content: [{ type: "text", text: result }]
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: `错误: ${error.message}` }],
      isError: true
    };
  }
});

// 启动Server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("MCP Server 已启动");
}

main().catch(console.error);
```

### 配置MCP Client

在Claude Desktop中配置MCP Server：

```json
// ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
// %APPDATA%\Claude\claude_desktop_config.json (Windows)
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["/path/to/weather_mcp_server.py"]
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/yourname/Documents"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_xxxxxxxxxxxx"
      }
    }
  }
}
```

## 🔧 MCP协议详解

### JSON-RPC 2.0 消息格式

MCP基于JSON-RPC 2.0协议：

```json
// 请求
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": { "city": "北京" }
  }
}

// 响应
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "{\"city\": \"北京\", \"temperature\": 25, \"condition\": \"晴\"}"
      }
    ]
  }
}

// 错误
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params"
  }
}
```

### 完整生命周期

```text
1. 初始化阶段
   Client → Server: initialize (协议版本、能力声明)
   Client → Server: initialized (确认完成)

2. 发现阶段
   Client → Server: tools/list (获取可用工具)
   Client → Server: resources/list (获取可用资源)

3. 交互阶段
   Client → Server: tools/call (调用工具)
   Client → Server: resources/read (读取资源)

4. 通知阶段
   Server → Client: notifications/tools/list_changed (工具变更)
   Server → Client: notifications/resources/list_changed (资源变更)
```

### 错误码规范

```text
-32700  Parse error      JSON解析错误
-32600  Invalid Request  无效请求
-32601  Method not found 方法不存在
-32602  Invalid params   参数无效
-32603  Internal error   内部错误

自定义错误码范围: -32000 to -32099
```

## ⚡ 高级特性

### 1. 工具变更通知

Server可以动态添加/移除工具：

```python
# 当Server能力变化时，通知Client
def notify_tools_changed(self):
    """通知Client工具列表已变更"""
    notification = {
        "jsonrpc": "2.0",
        "method": "notifications/tools/list_changed"
    }
    self.send(notification)
```

### 2. 资源订阅

Client可以订阅资源变更：

```python
# Client订阅资源
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/subscribe",
  "params": { "uri": "file:///path/to/watched/file" }
}

# Server推送变更
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": { "uri": "file:///path/to/watched/file" }
}
```

### 3. 提示词模板

Server可以提供预定义的提示词：

```python
@server.prompt(
    name="code_review",
    description="代码审查提示词模板"
)
def code_review_prompt(code: str, language: str) -> str:
    return f"""请审查以下{language}代码：

```{language}
{code}
```

审查要点：
1. 代码规范性
2. 潜在Bug
3. 性能优化建议
4. 安全性问题"""
```

## 🛡️ 安全最佳实践

### 权限控制

```python
class SecureMCPServer(MCPServer):
    """带权限控制的MCP Server"""
    
    def __init__(self, allowed_paths: list = None, max_file_size: int = 10*1024*1024):
        super().__init__()
        self.allowed_paths = allowed_paths or []
        self.max_file_size = max_file_size
    
    def _check_path_permission(self, path: str) -> bool:
        """检查路径权限"""
        import os
        abs_path = os.path.abspath(path)
        return any(abs_path.startswith(p) for p in self.allowed_paths)
    
    def _handle_tools_call(self, req_id, params):
        """重写工具调用，添加权限检查"""
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        # 路径权限检查
        if tool_name in ["read_file", "write_file"]:
            path = arguments.get("path", "")
            if not self._check_path_permission(path):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": f"无权访问路径: {path}"}
                }
        
        # 文件大小检查
        if tool_name == "write_file":
            content = arguments.get("content", "")
            if len(content.encode()) > self.max_file_size:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "文件大小超过限制"}
                }
        
        return super()._handle_tools_call(req_id, params)
```

### 敏感信息处理

```python
# ❌ 错误做法：直接暴露API Key
@server.tool(name="call_api", ...)
def call_api(endpoint: str):
    headers = {"Authorization": "Bearer sk-xxxxxxxxxxxx"}  # 暴露了！
    return requests.get(endpoint, headers=headers).json()

# ✅ 正确做法：在Server端管理凭证
class APIClient:
    def __init__(self):
        self.api_key = os.environ.get("API_KEY")  # 从环境变量读取
    
    def call(self, endpoint: str):
        headers = {"Authorization": f"Bearer {self.api_key}"}
        return requests.get(endpoint, headers=headers).json()

api_client = APIClient()

@server.tool(name="call_api", ...)
def call_api(endpoint: str):
    return api_client.call(endpoint)  # API Key不暴露给AI
```

## 📊 生态系统

### 官方MCP Server

```text
@modelcontextprotocol/server-filesystem  文件系统访问
@modelcontextprotocol/server-github      GitHub API
@modelcontextprotocol/server-gitlab      GitLab API
@modelcontextprotocol/server-google-drive Google Drive
@modelcontextprotocol/server-postgres    PostgreSQL
@modelcontextprotocol/server-slack       Slack
@modelcontextprotocol/server-memory      知识图谱
```

### 社区MCP Server

```text
awesome-mcp-servers: https://github.com/punkpeye/awesome-mcp-servers

包含：
- 数据库：MySQL、MongoDB、Redis、Elasticsearch
- 云服务：AWS、GCP、Azure
- 开发工具：Docker、Kubernetes、Jenkins
- 生活服务：天气、地图、翻译
- AI增强：图像生成、语音合成
```

### 在Cursor中使用MCP

```json
// .cursor/mcp.json
{
  "mcpServers": {
    "database": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres"],
      "env": {
        "DATABASE_URL": "postgresql://user:pass@localhost:5432/mydb"
      }
    }
  }
}
```

## 🎯 实际应用场景

### 场景1：智能代码助手

```text
用户："帮我查一下数据库里的用户表结构，然后生成一个TypeScript接口"

AI思考：
1. 调用 MCP工具 list_tables() → 获取所有表
2. 调用 MCP工具 describe_table("users") → 获取字段信息
3. 根据字段信息生成TypeScript接口代码
4. 调用 MCP工具 write_file() → 写入文件

AI回复：
"已查询users表结构，生成了TypeScript接口并保存到 types/user.ts"
```

### 场景2：数据分析助手

```text
用户："分析一下上个月的销售数据，找出销量最高的产品"

AI思考：
1. 调用 MCP工具 query_db("SELECT * FROM sales WHERE...") → 查询数据
2. 调用 MCP工具 run_python("import pandas...") → 数据分析
3. 调用 MCP工具 generate_chart() → 生成图表

AI回复：
"上月销量TOP5产品：
1. iPhone 15 - 1,234台
2. AirPods Pro - 987台
...（附图表已保存到 sales_report.png）"
```

### 场景3：全栈开发

```text
用户："创建一个用户注册功能，包括前端表单和后端API"

AI思考：
1. 调用 MCP工具 read_file("schema.prisma") → 了解数据库模型
2. 调用 MCP工具 write_file("api/register.ts") → 创建后端API
3. 调用 MCP工具 write_file("components/RegisterForm.tsx") → 创建前端组件
4. 调用 MCP工具 run_command("npm run build") → 验证构建

AI回复：
"已创建完整的用户注册功能：
- 后端API: api/register.ts（含输入验证、密码加密、数据库写入）
- 前端组件: components/RegisterForm.tsx（含表单验证、错误处理）
- 构建通过 ✅"
```

## 💡 核心要点总结

**MCP的本质**：
- ❌ 不是新的AI模型
- ✅ 是AI连接外部世界的标准协议

**三大核心组件**：
1. **Host**：运行AI的应用（Cursor、Claude Desktop）
2. **Client**：管理连接的组件
3. **Server**：提供能力的服务

**三大能力**：
1. **Tools**：AI可调用的函数
2. **Resources**：AI可读取的数据
3. **Prompts**：预定义的提示模板

**设计原则**：
- 协议统一：一套标准，处处可用
- 能力解耦：工具独立于AI平台
- 安全可控：权限在Server端控制
- 生态共享：社区共建Server库

---

> **系列回顾**：从Prompt到Token，从LLM到Agent，从Function Calling到MCP。我们一步步构建了AI应用的完整技术栈。MCP作为连接层，让AI不再是信息孤岛，而是真正的万物互联。
>
> **下期预告**：进入新系列——AI实战项目，手把手构建真实可用的AI应用。

---

*本文为AI核心概念系列第13期。*
*作者：赛博阿漆 | 发布日期：2026-07-07*
