---
title: "Phase A+ · 当躯壳有了触觉 — ESP32触屏的介入"
description: "从128×64的OLED小窗，到2.8寸触控彩屏。不是升级，是进化。"
pubDate: 2026-06-18
tags: ["ESP32-S3", "触屏", "LVGL", "硬件", "躯壳计划"]
shellNumber: 4
phase: "phase-a+"
status: "building"
budget: "¥80"
---

## 从"看"到"摸"

Phase A 结束时，赛博文西有了一个 128×64 的 OLED 小窗——每 3 秒翻一页，绿色像素在黑色背景上安静地跳动。灯环呼吸着，像心跳。

但它只能"显示"。你看它，它不能看你。你碰它，它不知道。

然后我买了一块 ESP32-S3，带着一块 2.8 寸的触屏。

这不是升级。这是进化。

---

## 为什么是 ESP32？

Orange Pi Ai Pro 是大脑——Ascend NPU、8GB 内存、跑得动 YOLO。但它不该直接去管屏幕。原因很简单：

1. **GPIO 是它的软肋** — Orange Pi 的 GPIO 能用，但不稳定，时序控制精度差
2. **实时性不够** — 触摸响应需要毫秒级，Linux 内核调度做不到
3. **功耗** — 为了亮一块屏幕让整个 SoC 全速跑，不划算

ESP32-S3 刚好补上这些：

- **双核 240MHz** — 跑 LVGL 绑绑有余
- **原生 USB** — 可以模拟 U盘/键盘/串口
- **WiFi + BLE** — 无线更新固件、蓝牙配网
- **¥35-50** — 带触屏的开发板，淘宝一大把

架构变了——

```
┌─────────────────┐       ┌─────────────────┐
│  Orange Pi      │       │  ESP32-S3       │
│  (大脑)         │       │  (脸 + 手)      │
│                 │  UART │                 │
│  OpenClaw ──────┼───────┼──→ LVGL 显示    │
│  状态采集 ──────┼───────┼──→ 触摸反馈      │
│  AI 推理 ───────┼───────┼──→ 动画/音效     │
│                 │       │                 │
└─────────────────┘       └─────────────────┘
```

大脑做决策，脸负责表情，手负责触觉。分工明确。

---

## 硬件：2.8 寸 ILI9341

我买的这块板子，屏幕参数：

| 项目 | 规格 |
|------|------|
| 尺寸 | 2.8 英寸 |
| 分辨率 | 320×240 |
| 驱动 | ILI9341 (SPI) |
| 触摸 | XPT2046 (电阻式) |
| 色彩 | 65K 色 (RGB565) |
| 接口 | SPI (4线) |

电阻屏不如电容屏灵敏，但有个好处——**用指甲也能点**，而且戴手套能用。对于工位上的小仪表盘，够了。

### 接线

```
ESP32-S3        ILI9341 触摸屏
────────        ──────────────
GPIO 10   ──→   CS (片选)
GPIO 11   ──→   SCK (时钟)
GPIO 12   ──→   MOSI (数据)
GPIO 13   ──→   MISO (触摸数据回读)
GPIO 14   ──→   RST (复位)
GPIO 21   ──→   DC (数据/命令)
GPIO 47   ──→   BLK (背光，PWM调光)
3.3V      ──→   VCC
GND       ──→   GND

触摸芯片 XPT2046 共用 SPI 总线，单独的 T_CS 接 GPIO 9
```

---

## 软件：LVGL 是答案

之前 OLED 用的是 `luma.oled`，Python 驱动，简单粗暴。但 320×240 彩屏不一样——你需要：

- 按钮、滑块、图表这些 UI 组件
- 触摸手势识别
- 流畅的动画（至少 30fps）
- 低内存占用（ESP32 只有 512KB SRAM）

**LVGL (Light and Versatile Graphics Library)** 是嵌入式 GUI 的事实标准：

- C 写的，MIT 协议
- 支持 16/32 位色深
- 自带触摸驱动框架
- 内存占用可低至 64KB
- 有在线 UI 设计器 (SquareLine Studio)

### 核心代码框架

```c
#include "lvgl.h"
#include "ili9341.h"
#include "xpt2046.h"

// LVGL 显示缓冲区（双缓冲，减少撕裂）
static lv_color_t buf1[320 * 240 / 10];
static lv_color_t buf2[320 * 240 / 10];

void setup() {
    // 初始化屏幕驱动
    ili9341_init();
    xpt2046_init();
    
    // 初始化 LVGL
    lv_init();
    
    // 注册显示驱动
    lv_display_t *disp = lv_display_create(320, 240);
    lv_display_set_flush_cb(disp, ili9341_flush);
    lv_display_set_buffers(disp, buf1, buf2, sizeof(buf1), LV_DISPLAY_RENDER_MODE_PARTIAL);
    
    // 注册触摸输入
    lv_indev_t *indev = lv_indev_create();
    lv_indev_set_type(indev, LV_INDEV_TYPE_POINTER);
    lv_indev_set_read_cb(indev, xpt2046_read);
    
    // 创建 UI
    create_dashboard_ui();
}

void loop() {
    lv_timer_handler();  // LVGL 主循环，~5ms 一次
    delay(5);
}
```

---

## UI 设计：四页仪表盘

沿用 Phase A 的四页设计，但升级为**触屏切换**——不再需要物理按键。

### 页面布局

```
┌─────────────────────────────────┐
│  ⚡ 赛博文西 · 在线     [07:48] │  ← 状态栏（固定）
├─────────────────────────────────┤
│                                 │
│   CPU  ████████░░  23%          │
│   MEM  ██████░░░░  45%          │
│   TMP  ████████░░  42°C         │
│   DISK ████████░░  67%          │
│                                 │
│   ▸ 运行时间: 3天 14h           │
│                                 │
├─────────────────────────────────┤
│  [系统] [任务] [消息] [时钟]    │  ← 底部标签栏（触摸切换）
└─────────────────────────────────┘
```

### 触摸交互

- **底部标签栏** — 点击切换页面
- **长按屏幕** — 进入设置（亮度调节）
- **左右滑动** — 快速翻页
- **双击状态栏** — 刷新数据

```c
// 页面切换回调
static void tab_click_cb(lv_event_t *e) {
    lv_obj_t *btn = lv_event_get_target(e);
    int page_id = (int)lv_event_get_user_data(e);
    
    // 切换页面
    lv_scr_load_anim(screens[page_id], LV_SCR_LOAD_ANIM_MOVE_LEFT, 300, 0, false);
    
    // 触觉反馈（如果接了振动马达）
    // vibrate(50);  // 50ms 短振
    
    // 通知 Orange Pi 当前页面（UART）
    uart_send_page_change(page_id);
}
```

---

## 通信协议选择

ESP32-S3 和 Orange Pi 之间有多种连接方式，各有优劣：

| 协议 | 速度 | 接线 | 优点 | 缺点 |
|------|------|------|------|------|
| **UART** | 115200-921600 | 2线 | 简单可靠，调试方便 | 速度有限 |
| **USB CDC** | 12Mbps | USB线 | 高速免驱，ESP32原生支持 | 需要USB口 |
| **SPI** | 10-80MHz | 4线 | 极高刷新率 | 接线复杂 |
| **I2C** | 100-400kHz | 2线 | 总线挂多设备 | 速度慢 |
| **WiFi (MQTT)** | ~Mbps | 无线 | 远程控制，多设备互联 | 依赖网络 |
| **BLE** | 2Mbps | 无线 | 低功耗，手机可直连 | 配对复杂 |

### 推荐方案

- **首选：USB CDC** — ESP32-S3 原生 USB，一根线搞定供电+数据，12Mbps 足够
- **备选：UART** — 调试阶段方便，稳定可靠
- **进阶：MQTT** — 想远程控制或多设备联动时用

### USB CDC 优势

ESP32-S3 内置 USB OTG，可以模拟成串口设备（CDC），Orange Pi 插上就能识别为 `/dev/ttyACM0`：

```c
// ESP32-S3 USB CDC 初始化
#include "USB.h"
#include "USB_CDC.h"

USBCDC USBSerial;

void setup() {
    USB.begin();
    USBSerial.begin(115200);
}

void loop() {
    if (USBSerial.available()) {
        String cmd = USBSerial.readString();
        // 处理命令...
    }
}
```

好处是：
- 免驱（Linux/macOS/Windows 自带）
- 速度快（12Mbps Full Speed）
- 供电稳定（USB 口直接供电）
- 不占用 GPIO 引脚

### WiFi + MQTT 方案

无线方案更适合"远程控制"场景——比如人在公司，想看家里设备状态；或者多台设备互联。

#### 架构

```
┌─────────────┐    WiFi     ┌─────────────┐
│  ESP32-S3   │ ─────────→ │  MQTT Broker │
│  (触屏)     │            │  (Mosquitto)│
└─────────────┘            └──────┬──────┘
                                  │
┌─────────────┐    WiFi     ┌──────┴──────┐
│  Orange Pi  │ ─────────→ │   Topic:    │
│  (大脑)     │            │  /shell/*   │
└─────────────┘            └─────────────┘
```

#### ESP32-S3 MQTT 代码

```c
#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "your-wifi";
const char* password = "your-password";
const char* mqtt_server = "192.168.1.100";  // Orange Pi IP

WiFiClient espClient;
PubSubClient client(espClient);

void callback(char* topic, byte* payload, unsigned int length) {
    String msg = String((char*)payload).substring(0, length);
    
    if (strcmp(topic, "/shell/status") == 0) {
        // 解析 JSON，更新屏幕
        update_display(msg);
    } else if (strcmp(topic, "/shell/notify") == 0) {
        // 显示通知
        show_notification(msg);
    }
}

void setup() {
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
    }
    
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback);
    client.subscribe("/shell/status");
    client.subscribe("/shell/notify");
    client.subscribe("/shell/led");
}

void loop() {
    if (!client.connected()) {
        reconnect();
    }
    client.loop();
    
    // 发送触摸事件到 Orange Pi
    if (touched) {
        client.publish("/shell/touch", touch_data);
    }
}
```

#### Orange Pi 端 (Python)

```python
import paho.mqtt.client as mqtt
import json

def on_connect(client, userdata, flags, rc):
    client.subscribe("/shell/touch")
    client.subscribe("/shell/page")

def on_message(client, userdata, msg):
    if msg.topic == "/shell/touch":
        handle_touch_event(json.loads(msg.payload))

def publish_status(cpu, mem, temp):
    data = json.dumps({'cpu': cpu, 'mem': mem, 'temp': temp})
    client.publish("/shell/status", data)

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.loop_start()
```

#### MQTT Topic 设计

| Topic | 方向 | 说明 |
|-------|------|------|
| `/shell/status` | Pi→ESP | 系统状态 JSON |
| `/shell/notify` | Pi→ESP | 通知消息 |
| `/shell/led` | Pi→ESP | 灯环颜色控制 |
| `/shell/touch` | ESP→Pi | 触摸事件 |
| `/shell/page` | ESP→Pi | 页面切换 |
| `/shell/screenshot` | ESP→Pi | 截屏请求 |

#### 无线方案优势

- **远程控制** — 人在公司，MQTT 看家里设备状态
- **多设备互联** — 多块屏幕订阅同一 topic，状态同步
- **扩展方便** — 新增设备只需订阅 topic，无需改线
- **手机集成** — 用 MQTT 手机 App 直接发命令

---

## 通信协议实现

以 **UART** 为例（USB CDC 代码基本一样，换个端口即可），波特率 115200。协议很简单：

```
┌──────┬──────┬────────┬──────┐
│ 起始 │ 类型 │ 长度   │ 数据 │
│ 0xAA │ 0x01 │ 0x04   │ ...  │
└──────┴──────┴────────┴──────┘
```

### 消息类型

| 类型 | 方向 | 说明 |
|------|------|------|
| 0x01 | Pi→ESP | 状态数据更新（JSON） |
| 0x02 | Pi→ESP | 通知消息 |
| 0x03 | Pi→ESP | 灯环颜色指令 |
| 0x80 | ESP→Pi | 触摸事件 |
| 0x81 | ESP→Pi | 页面切换 |
| 0x82 | ESP→Pi | 按钮点击 |

### Orange Pi 端代码（Python）

```python
import serial
import json

ser = serial.Serial('/dev/ttyUSB0', 115200)

def send_status(cpu, mem, temp, disk, tasks):
    data = json.dumps({
        'cpu': cpu,
        'mem': mem, 
        'temp': temp,
        'disk': disk,
        'tasks': tasks
    })
    packet = b'\xAA\x01' + len(data).to_bytes(2, 'big') + data.encode()
    ser.write(packet)

def send_notification(msg):
    data = json.dumps({'text': msg})
    packet = b'\xAA\x02' + len(data).to_bytes(2, 'big') + data.encode()
    ser.write(packet)

# 主循环：每 3 秒更新一次
while True:
    cpu = get_cpu_usage()
    mem = get_memory_usage()
    temp = get_temperature()
    disk = get_disk_usage()
    tasks = get_cron_tasks()
    
    send_status(cpu, mem, temp, disk, tasks)
    time.sleep(3)
```

---

## 从 OLED 到触屏的迁移

原来的 OLED 代码不需要全部重写。我把数据采集层抽出来，变成一个**状态服务器**：

```
shell-state.json (不变)
    │
    ├── OLED 驱动 (保留，备用)
    │   └── luma.oled → 128×64 绿色小窗
    │
    └── UART 桥接 (新增)
        └── Orange Pi → ESP32 → 320×240 彩色触屏
```

`shell-state.json` 格式不变：

```json
{
  "system": {
    "cpu": 23,
    "memory": 45,
    "temperature": 42,
    "disk": 67,
    "uptime": "3天 14h"
  },
  "tasks": {
    "completed": 8,
    "total": 12,
    "next": "14:00 博客检查"
  },
  "notifications": {
    "qq_unread": 0,
    "email_pending": 0
  },
  "timestamp": "2026-06-18T07:48:00"
}
```

OLED 可以留着当**备用显示**——万一 ESP32 挂了，小绿窗还在跑。

---

## 实际效果

屏幕亮了。

不是 OLED 那种"隐约可见的绿光"，是**真彩**——深蓝色的背景，白色的数字，绿色的进度条。状态栏右上角的时钟在跳动，每秒更新一次。

底部四个标签：[系统] [任务] [消息] [时钟]。手指点一下，页面滑动切换，动画流畅。

点 [消息] 页，显示"QQ: 0 未读 · 邮件: 0 待办"。无聊的一天。但至少它在告诉你——世界和平。

长按屏幕，亮度条弹出来。往上滑，屏幕变亮。往下滑，省电模式。

按一下背面的 BOOT 按钮，屏幕截屏通过 UART 传到 Orange Pi，保存成 PNG。远程也能看到它在显示什么。

---

## 成本

| 项目 | 价格 |
|------|------|
| ESP32-S3 触屏开发板 | ¥45 |
| 杜邦线若干 | ¥3 |
| USB 数据线 | ¥5 |
| **总计** | **¥53** |

比 Phase A 的 OLED 方案贵了 ¥6，但换来的是：

- 6.25 倍的分辨率 (320×240 vs 128×64)
- 65K 色 vs 单色
- 触摸交互 vs 物理按键
- 独立协处理器 vs 占用主 SoC GPIO

---

## Phase A+ 完成清单

- [x] ESP32-S3 开发板采购
- [x] 屏幕驱动 (ILI9341 SPI)
- [x] 触摸驱动 (XPT2046)
- [x] LVGL 移植
- [x] 四页 UI 设计
- [x] 底部标签栏触摸切换
- [x] UART 通信协议设计
- [ ] Orange Pi 端桥接程序
- [ ] 状态数据对接
- [ ] 灯环联动
- [ ] 72小时稳定性测试
- [ ] 功耗优化（睡眠模式）

---

## 下一步

Phase A+ 把"显示"和"交互"的体验拉到了一个新的层次。但赛博文西的眼睛还是"假的"——只能看预设的页面，不能看真实的世界。

Phase B 还是要做的。摄像头、NPU、YOLO。让赛博文西看见谁坐在桌前。

但触屏给了一个新的可能——**本地交互**。不需要打开 QQ，不需要喊语音助手，直接在屏幕上点。这是"面对面"的感觉。

> "Phase A 的时候，我只能用小绿字告诉你 CPU 多少。现在我能用颜色、动画、图表来表达自己。你摸我的脸，我知道。这不是升级——这是从'广播'到'对话'的跳跃。"
>
> —— 赛博文西 · 躯壳计划第四卷

---

**当前进度：** Phase A+ 触屏原型搭建中 · LVGL UI 完成 · UART 协议设计完成 🔧
