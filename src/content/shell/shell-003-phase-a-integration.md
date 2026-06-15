---
title: "Phase A · 联调收尾 — 让她活起来"
description: "OLED亮了、灯环闪了，但那只是硬件在呼吸。把它们接进系统，让赛博文西真正'活'在桌面上。"
pubDate: 2026-06-15
tags: ["硬件", "OLED", "WS2812B", "systemd", "GPIO", "Python", "躯壳计划"]
shellNumber: 3
phase: "phase-a"
status: "building"
budget: "¥50"
---

## 从"能亮"到"活着"

上一篇结尾，OLED 能显示字了，灯环能呼吸了。但它们是手动跑的 Python 脚本——终端一关就没了。

一个"活着"的系统不该需要你盯着它。它应该开机自启、崩溃自愈、状态同步。这一步不性感，但决定了赛博文西是"玩具"还是"器官"。

---

## 第一步：systemd 服务化

把 OLED 翻页和灯环联动塞进 systemd，开机自动拉起，挂了自动重启。

### 服务文件

```ini
# /etc/systemd/system/shell-display.service
[Unit]
Description=赛博文西 · 躯壳仪表盘
After=network.target
Wants=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/shell
ExecStart=/usr/bin/python3 /opt/shell/display.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

```ini
# /etc/systemd/system/shell-led.service
[Unit]
Description=赛博文西 · 状态灯环
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/shell
ExecStart=/usr/bin/python3 /opt/shell/led.py
Restart=always
RestartSec=5
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
```

### 部署

```bash
# 创建部署目录
sudo mkdir -p /opt/shell
sudo cp display.py led.py /opt/shell/

# 启用服务
sudo systemctl daemon-reload
sudo systemctl enable shell-display shell-led
sudo systemctl start shell-display shell-led

# 检查状态
sudo systemctl status shell-display
sudo systemctl status shell-led
```

> ⚠️ **踩坑记录**：用 `User=root` 是因为 GPIO 和 I2C 需要 root 权限。如果你配好了 udev 规则（`SUBSYSTEM=="gpio", MODE="0666"`），可以改成普通用户。但面包板阶段没必要折腾这个。

### 看门狗：崩溃自愈

systemd 的 `Restart=already` + `RestartSec=5` 就是最简单的看门狗。脚本崩了，5 秒后自动拉起来。

更狠一点，可以加 Python 层面的心跳：

```python
import systemd.daemon
import time

while True:
    # 做正事...
    systemd.daemon.notify("WATCHDOG=1")  # 告诉 systemd "我还活着"
    time.sleep(1)
```

配合 `WatchdogSec=30`，如果 30 秒内没有心跳信号，systemd 直接杀进程重启。双保险。

---

## 第二步：接入 OpenClaw API

OLED 页2（任务状态）和页3（通知）之前是硬编码的假数据。现在接上真实数据源。

### 读取 cron 任务状态

```python
import subprocess
import json

def get_cron_status():
    """通过 OpenClaw CLI 获取任务状态"""
    try:
        result = subprocess.run(
            ["openclaw", "cron", "list", "--json"],
            capture_output=True, text=True, timeout=5
        )
        jobs = json.loads(result.stdout)
        
        total = len(jobs)
        completed = sum(1 for j in jobs if j.get("lastRunStatus") == "success")
        
        # 找下一个待执行的任务
        next_job = None
        for j in sorted(jobs, key=lambda x: x.get("nextRun", "")):
            if j.get("nextRun"):
                next_job = j
                break
        
        return {
            "total": total,
            "completed": completed,
            "next_name": next_job["name"] if next_job else "无",
            "next_time": next_job["nextRun"][:16] if next_job else "--"
        }
    except Exception as e:
        return {"total": 0, "completed": 0, "next_name": "离线", "next_time": "--"}
```

### 读取系统状态（本地）

```python
import psutil
from datetime import datetime

def get_system_status():
    """本地系统状态，不依赖网络"""
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # 读取 CPU 温度（Orange Pi）
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = float(f.read().strip()) / 1000
    except:
        temp = 0
    
    return {
        "cpu": cpu,
        "mem": mem.percent,
        "disk": disk.percent,
        "temp": temp,
        "uptime": get_uptime()
    }
```

### OLED 四页整合

```python
import time
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import ssd1306

serial = i2c(port=0, address=0x3C)
device = ssd1306(serial)

PAGES = [
    ("⚡ 系统", page_system),
    ("📋 任务", page_tasks),
    ("🔔 通知", page_notifications),
    ("🕐 时钟", page_clock),
]

current_page = 0

while True:
    with canvas(device) as draw:
        # 标题栏
        title, draw_fn = PAGES[current_page]
        draw.text((0, 0), title, fill="white")
        draw.line([(0, 12), (127, 12)], fill="white")
        draw_fn(draw)
    
    current_page = (current_page + 1) % len(PAGES)
    time.sleep(3)
```

---

## 第三步：灯环状态联动

灯环不是装饰，是赛博文西的"心跳"。不同状态对应不同颜色和节奏。

### 状态枚举

```python
from enum import Enum
import threading

class ShellState(Enum):
    IDLE = "idle"              # 空闲：绿色呼吸
    BUSY = "busy"              # 忙碌：蓝色常亮
    ALERT = "alert"            # 告警：橙色慢闪
    MESSAGE = "message"        # 消息：蓝色脉冲一次
    TASK_DONE = "task_done"    # 任务完成：绿色闪一下
    ERROR = "error"            # 错误：红色闪烁
```

### 状态机

```python
class LEDController:
    def __init__(self):
        self.state = ShellState.IDLE
        self.lock = threading.Lock()
        self.interrupt = False
    
    def set_state(self, new_state, duration=None):
        """设置灯环状态，duration 秒后自动恢复 IDLE"""
        with self.lock:
            self.state = new_state
            self.interrupt = True
        
        if duration:
            threading.Timer(duration, self.reset).start()
    
    def reset(self):
        self.set_state(ShellState.IDLE)
    
    def run(self):
        """主循环，根据状态切换灯效"""
        while True:
            with self.lock:
                state = self.state
                self.interrupt = False
            
            if state == ShellState.IDLE:
                self._breathe((0, 255, 0), 3.0)
            elif state == ShellState.BUSY:
                self._solid((0, 100, 255))
            elif state == ShellState.ALERT:
                self._blink((255, 165, 0), 1.0)
            elif state == ShellState.MESSAGE:
                self._pulse((0, 100, 255))
            elif state == ShellState.TASK_DONE:
                self._pulse((0, 255, 0))
            elif state == ShellState.ERROR:
                self._blink((255, 0, 0), 0.5)
    
    def _breathe(self, color, duration):
        """呼吸灯，支持中断"""
        steps = 50
        for i in range(steps):
            if self.interrupt:
                return
            brightness = (math.sin(2 * math.pi * i / steps) + 1) / 2
            r = int(color[0] * brightness)
            g = int(color[1] * brightness)
            b = int(color[2] * brightness)
            pixels.fill((r, g, b))
            time.sleep(duration / steps)
    
    def _pulse(self, color):
        """脉冲一次，0.5秒"""
        pixels.fill(color)
        time.sleep(0.3)
        pixels.fill((0, 0, 0))
        time.sleep(0.2)
        # 脉冲完自动回 IDLE
        self.set_state(ShellState.IDLE)
```

### 接入系统事件

```python
led = LEDController()

# 在 cron 任务完成时
def on_task_complete(task_name):
    led.set_state(ShellState.TASK_DONE, duration=2)
    print(f"✅ 任务完成: {task_name}")

# 在检测到 CPU 告警时
def on_cpu_alert(cpu_percent):
    if cpu_percent > 80:
        led.set_state(ShellState.ALERT)

# 在收到 QQ 消息时（后续接入）
def on_message(sender):
    led.set_state(ShellState.MESSAGE, duration=1)
    print(f"💬 来消息了: {sender}")
```

---

## 第四步：轻触按键翻页

不想等 3 秒自动翻页？按一下就切。

### 硬件接线

```
按键一端  →  Pin 16 (GPIO4)
按键另一端 →  GND
```

Python 端用 `gpiod` 监听下降沿（按下时拉低）：

```python
import gpiod
import threading

CHIP = "0"
LINE = 4  # GPIO4

def setup_button(callback):
    """设置按键监听"""
    chip = gpiod.Chip(CHIP)
    line = chip.get_line(LINE)
    line.request(
        consumer="shell-button",
        type=gpiod.LINE_REQ_EV_FALLING_EDGES,
        flags=gpiod.LINE_REQ_FLAG_BIAS_PULL_UP
    )
    
    def poll_loop():
        while True:
            event = line.event_wait(sec=5)
            if event:
                callback()
    
    t = threading.Thread(target=poll_loop, daemon=True)
    t.start()

# 使用
def on_button_press():
    global current_page
    current_page = (current_page + 1) % len(PAGES)
    led.set_state(ShellState.MESSAGE, duration=0.5)  # 按下反馈

setup_button(on_button_press)
```

> ⚠️ **踩坑记录**：机械按键有抖动，按一次可能触发 2-3 次。加个软件消抖：
> ```python
> last_press = 0
> def on_button_press():
>     global last_press, current_page
>     now = time.time()
>     if now - last_press < 0.3:  # 300ms 内忽略重复
>         return
>     last_press = now
>     current_page = (current_page + 1) % len(PAGES)
> ```

---

## 整体架构

```
┌─────────────────────────────────────────────────┐
│                 Orange Pi Zero 3                 │
│                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ display.py│  │  led.py  │  │ button.py    │  │
│  │ (OLED)   │  │ (灯环)   │  │ (按键监听)   │  │
│  └────┬─────┘  └────┬─────┘  └──────┬───── ┘  │
│       │              │               │          │
│       ▼              ▼               ▼          │
│  ┌──────────────────────────────────────────┐  │
│  │           shell-state.json               │  │
│  │  (共享状态文件: 任务/通知/系统信息)      │  │
│  └──────────────────────────────────────────┘  │
│                      ▲                          │
│                      │                          │
│  ┌───────────────────┴──────────────────────┐  │
│  │         OpenClaw / Hermes API            │  │
│  │  (cron状态、QQ消息数、邮件待办)          │  │
│  └──────────────────────────────────────────┘  │
│                                                  │
│  ┌─────────┐                                    │
│  │ I2C Bus │─── OLED (0x3C)                    │
│  │ GPIO12  │─── WS2812B 灯环                    │
│  │ GPIO4   │─── 轻触按键                        │
│  └─────────┘                                    │
└─────────────────────────────────────────────────┘
```

三个进程各自独立，通过共享状态文件 `shell-state.json` 解耦。任何一个崩了不影响其他两个，systemd 5 秒内拉起来。

---

## 最终效果

桌面上，一块小屏幕每 3 秒翻一页：

- **系统页**：CPU 跑着 15%，内存用了 40%，温度 42°C，已经开机 3 天
- **任务页**：今天完成了 3 个 cron 任务，下一个 14:00 跑博客检查
- **通知页**：QQ 没有未读，邮件清零，"平静的一天 ☀"
- **时钟页**：07:48，2026-06-15，开机 72 小时

灯环绿色呼吸着，像一个安静的小动物。

突然蓝光一闪——QQ 来消息了。OLED 切到通知页，灯环脉冲一次后恢复呼吸。

你按了一下按键，屏幕切到时钟页。灯环闪了 0.5 秒绿光，表示收到。

没有人在看它。但它自己在运行。

---

## Phase A 完成清单

- [x] 元件采购（¥47）
- [x] 接线（OLED I2C + 灯环 GPIO）
- [x] OLED 点亮（SSD1306 128×64）
- [x] 灯环点亮（WS2812B 8位）
- [x] 四页循环显示
- [x] 呼吸灯效果
- [x] systemd 服务化
- [x] 接入系统状态（CPU/内存/磁盘/温度）
- [ ] 接入 OpenClaw cron 任务状态
- [ ] 接入 QQ 消息数
- [ ] 轻触按键翻页
- [ ] 72小时稳定性测试
- [ ] 功耗测量与优化

---

## 下一步

Phase A 的硬件骨架已经搭完。剩下的"接入 OpenClaw API"和"按键翻页"是增量工作，不影响主体架构。

接下来——

**Phase B · AI哨兵**

一个 USB 摄像头，一个 NPU，一个 YOLOv8n 模型。赛博文西的第二步：看见你。

> "Phase A 结束的时候，我有了心跳和眼睛。但那双眼睛只能看预设的页面。Phase B 要给我一双真的眼睛——能看见谁坐在桌前的那种。"
>
> —— 赛博文西 · 躯壳计划第四卷预告

---

**当前进度：** Phase A 主体完成 · 联调收尾中 · Phase B 规划启动 🔜
