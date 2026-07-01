---
title: "文西专属管家智能系统技术实现方案——香橙派5+ESP32语音对话系统"
description: "用香橙派5+ESP32打造文西专属管家智能语音助理"
pubDate: 2026-07-01
inventionNumber: 003
ideaId: "003"
tags: ["AI", "IoT", "语音交互", "ESP32", "香橙派", "OpenClaw"]
status: prototype
---

# 🔧 贾维斯智能助理技术实现方案

> 从灵感到代码，把贾维斯从电影里拽出来。

## 一、系统架构总览

```
┌─────────────────────────────────────────────────────────┐
│                    用户（语音输入）                        │
└──────────────────────┬──────────────────────────────────┘
                       │ 声波
                       ▼
┌─────────────────────────────────────────────────────────┐
│              ESP32 + I2S麦克风 + I2S喇叭                  │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐                  │
│  │ 麦克风   │→│ ADC采集  │→│ WiFi发送 │──┐               │
│  └─────────┘  └─────────┘  └─────────┘  │               │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  │               │
│  │ 喇叭     │←│ DAC播放  │←│ WiFi接收 │←─┤               │
│  └─────────┘  └─────────┘  └─────────┘  │               │
└──────────────────────────────────────────┼───────────────┘
                                           │ WebSocket/HTTP
                                           ▼
┌─────────────────────────────────────────────────────────┐
│                  香橙派5 (OpenClaw)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │ 音频接收  │→│ STT识别   │→│ LLM对话   │               │
│  └──────────┘  └──────────┘  └──────────┘               │
│       ↑                              │                   │
│       │         ┌──────────┐         │                   │
│       └─────────│ TTS合成   │←────────┘                   │
│                 └──────────┘                             │
│                      │                                   │
│              ┌───────┴───────┐                           │
│              │ 指令执行引擎   │                           │
│              │ (OpenClaw)    │                           │
│              └───────────────┘                           │
└─────────────────────────────────────────────────────────┘
```

## 二、硬件清单

| 组件 | 型号推荐 | 价格参考 | 用途 |
|------|---------|---------|------|
| 主控板 | 香橙派5 (8GB) | ¥500 | 运行AI模型和OpenClaw |
| 语音模块 | ESP32-S3 | ¥30 | 音频采集/播放，WiFi通信 |
| 麦克风 | INMP441 I2S麦克风 | ¥8 | 高质量音频采集 |
| 喇叭 | MAX98357A + 小喇叭 | ¥15 | I2S音频输出 |
| 电源 | 5V/3A USB-C | ¥20 | 香橙派供电 |
| 外壳 | 3D打印/现成壳 | ¥30 | 可选，看审美 |

**总成本：约 ¥600**（比钢铁侠便宜几亿）

## 三、ESP32 固件实现

### 3.1 开发环境

```bash
# 安装 ESP-IDF 或使用 Arduino IDE
# 推荐 PlatformIO + ESP-IDF

# 依赖库
# - ESP32-audioI2S (音频采集/播放)
# - ArduinoJson (JSON解析)
# - WebSockets (WebSocket客户端)
```

### 3.2 核心代码（ESP32-S3）

```cpp
// main.cpp - ESP32 语音采集与播放
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <driver/i2s.h>
#include <ArduinoJson.h>

// I2S 麦克风引脚 (INMP441)
#define I2S_MIC_WS   4
#define I2S_MIC_SD   5
#define I2S_MIC_SCK  6

// I2S 喇叭引脚 (MAX98357A)
#define I2S_SPK_LRC  7
#define I2S_SPK_BCLK 8
#define I2S_SPK_DIN  9

// WiFi 和服务器配置
const char* WIFI_SSID = "你的WiFi";
const char* WIFI_PASS = "你的密码";
const char* SERVER_IP  = "192.168.1.100"; // 香橙派IP
const int   SERVER_PORT = 8765;

// 音频参数
#define SAMPLE_RATE   16000
#define BUFFER_SIZE   1024
#define CHANNELS      1

WebSocketsClient webSocket;
bool isRecording = false;
bool isSpeaking = false;

// I2S 麦克风配置
void setupMicrophone() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = BUFFER_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = false,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_MIC_SCK,
        .ws_io_num = I2S_MIC_WS,
        .data_out_num = I2S_PIN_NO_CHANGE,
        .data_in_num = I2S_MIC_SD
    };

    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
}

// I2S 喇叭配置
void setupSpeaker() {
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
        .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
        .dma_buf_count = 8,
        .dma_buf_len = BUFFER_SIZE,
        .use_apll = false,
        .tx_desc_auto_clear = true,
        .fixed_mclk = 0
    };

    i2s_pin_config_t pin_config = {
        .bck_io_num = I2S_SPK_BCLK,
        .ws_io_num = I2S_SPK_LRC,
        .data_out_num = I2S_SPK_DIN,
        .data_in_num = I2S_PIN_NO_CHANGE
    };

    i2s_driver_install(I2S_NUM_1, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_1, &pin_config);
}

// WebSocket 事件处理
void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch (type) {
        case WStype_CONNECTED:
            Serial.println("[WebSocket] 已连接到香橙派");
            // 发送注册消息
            webSocket.sendTXT("{\"type\":\"register\",\"device\":\"esp32-jarvis\"}");
            break;

        case WStype_TEXT: {
            // 收到文本消息（AI回复或控制指令）
            StaticJsonDocument<512> doc;
            deserializeJson(doc, payload);
            
            String msgType = doc["type"].as<String>();
            
            if (msgType == "tts_start") {
                // TTS音频即将开始
                isSpeaking = true;
                Serial.println("[TTS] 开始播放");
            }
            else if (msgType == "tts_end") {
                // TTS播放结束
                isSpeaking = false;
                Serial.println("[TTS] 播放结束");
            }
            else if (msgType == "text") {
                // 显示AI回复文本（调试用）
                Serial.printf("[AI] %s\n", doc["content"].as<String>().c_str());
            }
            break;
        }

        case WStype_BIN: {
            // 收到音频数据（TTS输出）
            if (isSpeaking) {
                size_t bytes_written;
                i2s_write(I2S_NUM_1, payload, length, &bytes_written, portMAX_DELAY);
            }
            break;
        }

        case WStype_DISCONNECTED:
            Serial.println("[WebSocket] 连接断开");
            break;
    }
}

// 语音活动检测 (VAD) - 简单能量检测
bool detectVoice(int16_t* buffer, size_t samples) {
    float energy = 0;
    for (size_t i = 0; i < samples; i++) {
        energy += abs(buffer[i]);
    }
    energy /= samples;
    
    // 阈值需要根据环境调整
    return energy > 500;
}

// 录音并发送
void recordAndSend() {
    int16_t buffer[BUFFER_SIZE];
    size_t bytes_read;
    
    // 读取麦克风数据
    i2s_read(I2S_NUM_0, buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
    
    // 检测是否有语音
    if (detectVoice(buffer, BUFFER_SIZE)) {
        if (!isRecording) {
            isRecording = true;
            // 通知服务端开始录音
            webSocket.sendTXT("{\"type\":\"record_start\"}");
            Serial.println("[录音] 检测到语音，开始录音");
        }
        
        // 发送音频数据
        webSocket.sendBIN((uint8_t*)buffer, bytes_read);
    } else {
        if (isRecording) {
            isRecording = false;
            // 通知服务端录音结束
            webSocket.sendTXT("{\"type\":\"record_end\"}");
            Serial.println("[录音] 语音结束");
        }
    }
}

void setup() {
    Serial.begin(115200);
    
    // 连接WiFi
    WiFi.begin(WIFI_SSID, WIFI_PASS);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.printf("\nWiFi已连接: %s\n", WiFi.localIP().toString().c_str());
    
    // 初始化音频
    setupMicrophone();
    setupSpeaker();
    
    // 连接WebSocket服务器
    webSocket.begin(SERVER_IP, SERVER_PORT, "/ws");
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
}

void loop() {
    webSocket.loop();
    
    if (!isSpeaking) {
        recordAndSend();
    }
}
```

### 3.3 按键触发模式（可选）

如果不想用 VAD 自动检测，可以加个物理按键：

```cpp
// 按住说话，松开发送
#define BUTTON_PIN 0  // BOOT按键

void loop() {
    webSocket.loop();
    
    if (digitalRead(BUTTON_PIN) == LOW) {
        // 按住录音
        int16_t buffer[BUFFER_SIZE];
        size_t bytes_read;
        i2s_read(I2S_NUM_0, buffer, sizeof(buffer), &bytes_read, portMAX_DELAY);
        webSocket.sendBIN((uint8_t*)buffer, bytes_read);
        
        if (!isRecording) {
            isRecording = true;
            webSocket.sendTXT("{\"type\":\"record_start\"}");
        }
    } else {
        if (isRecording) {
            isRecording = false;
            webSocket.sendTXT("{\"type\":\"record_end\"}");
        }
    }
}
```

## 四、香橙派5 服务端实现

### 4.1 环境准备

```bash
# 系统：Ubuntu 22.04 / Debian 12 for Orange Pi 5

# 安装依赖
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg portaudio19-dev

# 创建项目目录
mkdir -p ~/jarvis-server && cd ~/jarvis-server
python3 -m venv venv
source venv/bin/activate

# Python 依赖
pip install websockets openai-whisper edge-tts aiohttp numpy
```

### 4.2 核心服务端代码

```python
#!/usr/bin/env python3
"""
Jarvis Server - 香橙派5语音助理服务端
接收ESP32音频 → STT识别 → LLM对话 → TTS合成 → 回传音频
"""

import asyncio
import json
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional

import websockets
import whisper
import edge_tts
from openai import OpenAI  # 或其他LLM客户端

# ============ 配置 ============

WS_HOST = "0.0.0.0"
WS_PORT = 8765

# STT 配置 (使用 OpenAI Whisper 本地模型)
WHISPER_MODEL = "base"  # tiny/base/small/medium 选一个

# LLM 配置
LLM_API_KEY = "你的API密钥"
LLM_BASE_URL = "https://api.openai.com/v1"  # 或其他兼容API
LLM_MODEL = "gpt-4o-mini"

# TTS 配置 (使用 Edge TTS，免费且质量不错)
TTS_VOICE = "zh-CN-YunxiNeural"  # 男声，也可选女声

# ============ 核心类 ============

class JarvisBrain:
    """贾维斯大脑 - 处理语音识别、对话和语音合成"""
    
    def __init__(self):
        print("[初始化] 加载 Whisper 模型...")
        self.stt_model = whisper.load_model(WHISPER_MODEL)
        
        print("[初始化] 连接 LLM API...")
        self.llm_client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_BASE_URL
        )
        
        # 对话历史
        self.conversation_history = [
            {
                "role": "system",
                "content": """你是贾维斯，一个智能AI助理。
                你的回答应该简洁、准确、有帮助。
                你可以帮助用户查询信息、控制智能家居、设置提醒等。
                说话风格像电影里的贾维斯：专业但带点幽默。"""
            }
        ]
        
        print("[初始化] 贾维斯大脑就绪 ✓")
    
    def speech_to_text(self, audio_data: bytes) -> str:
        """语音转文字"""
        # 将 bytes 转为 numpy 数组
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Whisper 需要 float32 格式，16kHz
        result = self.stt_model.transcribe(
            audio_np,
            language="zh",
            fp16=False  # 香橙派5 用 CPU，关闭 fp16
        )
        
        text = result["text"].strip()
        print(f"[STT] 识别结果: {text}")
        return text
    
    def chat(self, user_input: str) -> str:
        """LLM 对话"""
        self.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        response = self.llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=self.conversation_history,
            max_tokens=500,
            temperature=0.7
        )
        
        assistant_message = response.choices[0].message.content
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_message
        })
        
        # 保持对话历史不超过20轮
        if len(self.conversation_history) > 22:  # system + 20轮
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[-20:]
        
        print(f"[LLM] 回复: {assistant_message}")
        return assistant_message
    
    async def text_to_speech(self, text: str) -> bytes:
        """文字转语音 (Edge TTS)"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(temp_path)
        
        # 转换为 PCM 格式 (16kHz, 16bit, mono)
        import subprocess
        pcm_path = temp_path.replace(".mp3", ".pcm")
        subprocess.run([
            "ffmpeg", "-y", "-i", temp_path,
            "-ar", "16000", "-ac", "1", "-f", "s16le",
            pcm_path
        ], capture_output=True)
        
        audio_data = Path(pcm_path).read_bytes()
        
        # 清理临时文件
        Path(temp_path).unlink(missing_ok=True)
        Path(pcm_path).unlink(missing_ok=True)
        
        print(f"[TTS] 合成完成，{len(audio_data)} 字节")
        return audio_data

class JarvisServer:
    """WebSocket 服务器 - 处理 ESP32 通信"""
    
    def __init__(self):
        self.brain = JarvisBrain()
        self.clients = set()
    
    async def handle_client(self, websocket, path):
        """处理单个客户端连接"""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0]
        print(f"[连接] 新客户端: {client_ip}")
        
        audio_buffer = bytearray()
        is_recording = False
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    # 文本消息（控制指令）
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "register":
                        print(f"[设备] ESP32 注册: {data.get('device')}")
                    
                    elif msg_type == "record_start":
                        is_recording = True
                        audio_buffer.clear()
                        print("[录音] 开始接收音频")
                    
                    elif msg_type == "record_end":
                        is_recording = False
                        print(f"[录音] 结束，共 {len(audio_buffer)} 字节")
                        
                        if len(audio_buffer) > 1600:  # 至少0.05秒音频
                            # 处理语音
                            await self.process_voice(websocket, bytes(audio_buffer))
                        audio_buffer.clear()
                
                elif isinstance(message, bytes):
                    # 音频数据
                    if is_recording:
                        audio_buffer.extend(message)
        
        except websockets.exceptions.ConnectionClosed:
            print(f"[断开] 客户端: {client_ip}")
        finally:
            self.clients.discard(websocket)
    
    async def process_voice(self, websocket, audio_data: bytes):
        """处理语音：STT → LLM → TTS"""
        try:
            # 1. 语音识别
            await websocket.send(json.dumps({"type": "status", "content": "正在识别..."}))
            text = self.brain.speech_to_text(audio_data)
            
            if not text or len(text) < 2:
                await websocket.send(json.dumps({
                    "type": "text",
                    "content": "没听清，请再说一次"
                }))
                return
            
            # 2. LLM 对话
            await websocket.send(json.dumps({"type": "status", "content": "思考中..."}))
            reply = self.brain.chat(text)
            
            # 3. 发送文本回复
            await websocket.send(json.dumps({
                "type": "text",
                "content": reply
            }))
            
            # 4. TTS 合成并发送音频
            await websocket.send(json.dumps({"type": "tts_start"}))
            audio = await self.brain.text_to_speech(reply)
            
            # 分块发送音频（每块 1024 字节）
            chunk_size = 1024
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i + chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(0.01)  # 模拟实时播放
            
            await websocket.send(json.dumps({"type": "tts_end"}))
            
        except Exception as e:
            print(f"[错误] 处理失败: {e}")
            await websocket.send(json.dumps({
                "type": "error",
                "content": str(e)
            }))
    
    async def start(self):
        """启动服务器"""
        print(f"[服务器] 启动于 ws://{WS_HOST}:{WS_PORT}")
        async with websockets.serve(self.handle_client, WS_HOST, WS_PORT):
            await asyncio.Future()  # 永远运行

# ============ 启动 ============

if __name__ == "__main__":
    server = JarvisServer()
    asyncio.run(server.start())
```

### 4.3 系统服务配置

```bash
# /etc/systemd/system/jarvis.service
sudo tee /etc/systemd/system/jarvis.service << 'EOF'
[Unit]
Description=Jarvis Voice Assistant Server
After=network.target

[Service]
Type=simple
User=orangepi
WorkingDirectory=/home/orangepi/jarvis-server
ExecStart=/home/orangepi/jarvis-server/venv/bin/python server.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

# 启用并启动
sudo systemctl daemon-reload
sudo systemctl enable jarvis
sudo systemctl start jarvis
sudo systemctl status jarvis
```

## 五、OpenClaw 集成

### 5.1 作为 OpenClaw 工具调用

让贾维斯不仅能聊天，还能执行 OpenClaw 的能力：

```python
# 在 server.py 中添加 OpenClaw 集成

import subprocess

def execute_openclaw_command(command: str) -> str:
    """调用 OpenClaw 执行指令"""
    try:
        result = subprocess.run(
            ["openclaw", "run", "--message", command],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.stdout
    except Exception as e:
        return f"执行失败: {e}"

# 在 chat() 方法中添加工具调用判断
def chat(self, user_input: str) -> str:
    # 检查是否是指令
    if any(keyword in user_input for keyword in ["帮我", "执行", "运行", "查一下"]):
        # 调用 OpenClaw
        result = execute_openclaw_command(user_input)
        return result
    
    # 普通对话
    # ... 原有逻辑
```

### 5.2 智能家居控制扩展

```python
# 智能家居控制模块

class SmartHome:
    """智能家居控制"""
    
    def __init__(self):
        self.devices = {
            "客厅灯": {"state": False, "type": "light"},
            "卧室灯": {"state": False, "type": "light"},
            "空调": {"state": False, "type": "ac", "temp": 26},
        }
    
    def control(self, device_name: str, action: str) -> str:
        if device_name not in self.devices:
            return f"未找到设备: {device_name}"
        
        device = self.devices[device_name]
        
        if action == "开":
            device["state"] = True
            return f"已打开{device_name}"
        elif action == "关":
            device["state"] = False
            return f"已关闭{device_name}"
        
        return "未知操作"
    
    def parse_command(self, text: str) -> Optional[tuple]:
        """解析语音指令"""
        for device_name in self.devices:
            if device_name in text:
                if "开" in text or "打开" in text:
                    return (device_name, "开")
                elif "关" in text or "关闭" in text:
                    return (device_name, "关")
        return None
```

## 六、优化建议

### 6.1 降低延迟

- **STT 优化**：用 `whisper.cpp` 替代 Python Whisper，速度快 4 倍
- **流式识别**：边录边识别，不用等说完
- **本地 LLM**：用 llama.cpp 跑小模型，省去网络延迟

### 6.2 唤醒词

```python
# 使用 Porcupine 做唤醒词检测
# pip install pvporcupine

import pvporcupine
import pyaudio

porcupine = pvporcupine.create(
    access_key='你的KEY',
    keywords=['jarvis']  # 唤醒词："贾维斯"
)

pa = pyaudio.PyAudio()
stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

while True:
    pcm = stream.read(porcupine.frame_length)
    keyword_index = porcupine.process(pcm)
    if keyword_index >= 0:
        print("贾维斯在听...")
        # 开始录音处理
```

### 6.3 多房间支持

- 多个 ESP32 分布在不同房间
- 通过设备 ID 区分位置
- 根据位置调整回复（"客厅灯已打开" vs "卧室灯已打开"）

## 七、调试技巧

```bash
# 查看服务日志
sudo journalctl -u jarvis -f

# 测试 WebSocket 连接
pip install websocat
websocat ws://192.168.1.100:8765/ws

# 测试音频采集
ffmpeg -f alsa -i default -t 5 test.wav
```

## 八、后续计划

- [ ] 唤醒词支持（"贾维斯"唤醒）
- [ ] 多房间 ESP32 分布式部署
- [ ] 接入 Home Assistant 控制智能家居
- [ ] 情感识别（根据语气调整回复）
- [ ] 本地 LLM 部署（隐私+低延迟）
- [ ] 3D 打印贾维斯风格外壳

---

_钢铁侠花了几十亿研发贾维斯，我们花了六百块。虽然没有全息投影，但至少能关灯。_
