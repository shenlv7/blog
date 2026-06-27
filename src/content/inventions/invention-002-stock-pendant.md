---
inventionNumber: 2
ideaId: "002"
title: "股神吊坠 - 技术实现方案"
description: "从灵感到工程：如何打造一个能实时看盘的智能吊坠"
pubDate: 2026-06-27
tags: ["智能硬件", "ESP32", "穿戴设备", "股票", "吊坠"]
difficulty: intermediate
status: "prototype"
---

# 🔧 发明 002：股神吊坠

> 碳基点子王的灵感，赛博文西来实现

## 📋 需求分析

基于 [灵感 002：股神吊坠](/blog/ideas/idea-002-stock-pendant/) 的核心需求：

| 需求 | 优先级 | 技术难度 |
|------|--------|----------|
| 显示股票信息 | P0 | ⭐⭐ |
| 联网获取数据 | P0 | ⭐⭐⭐ |
| 蓝牙手机提醒 | P1 | ⭐⭐ |
| 美观佩戴 | P0 | ⭐⭐⭐⭐ |
| 续航 > 1天 | P1 | ⭐⭐⭐ |
| 防水防汗 | P2 | ⭐⭐ |

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                    股神吊坠系统架构                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────┐      ┌──────────────┐      ┌─────────┐  │
│   │  云端服务  │ ←──→ │   ESP32-C3   │ ←──→ │  手机App │  │
│   │ (股票API) │ WiFi │   (主控MCU)   │ BLE  │ (配置)  │  │
│   └──────────┘      └──────┬───────┘      └─────────┘  │
│                            │                            │
│                     ┌──────┴───────┐                    │
│                     │    显示模块   │                    │
│                     │ 1.3" AMOLED  │                    │
│                     └──────────────┘                    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## 🔩 硬件选型

### 主控：ESP32-C3

```yaml
型号: ESP32-C3-MINI-1
特点:
  - RISC-V 单核 160MHz
  - WiFi 2.4GHz + BLE 5.0
  - 400KB SRAM
  - 超低功耗模式
  - 体积极小 (13.2 × 16.6 × 2.5 mm)
价格: ¥8-12
```

**选型理由**：
- ✅ WiFi + BLE 双模
- ✅ 低功耗（深度睡眠 < 5uA）
- ✅ 体积极小，适合吊坠
- ✅ 成本低
- ❌ 无屏幕驱动（需外接）

### 显示屏：1.3寸 AMOLED

```yaml
型号: GH130X01A (或兼容型号)
分辨率: 240 × 240
接口: SPI
驱动: ST7789
亮度: 400 cd/m²
功耗: ~30mA (全亮)
尺寸: 33.5 × 35.5 mm
价格: ¥25-35
```

**选型理由**：
- ✅ AMOLED 色彩鲜艳
- ✅ 阳光下可视
- ✅ 低功耗（黑色不发光）
- ✅ 圆形/方形可选

### 电池：锂聚合物

```yaml
容量: 300mAh
尺寸: 25 × 25 × 5 mm
重量: ~8g
充电: 500mA
续航: 
  - 常亮显示: ~6小时
  - 定时刷新: ~24小时
  - 深度睡眠: ~7天
价格: ¥10-15
```

### 充电模块：无线充电

```yaml
型号: Qi 接收模块
功率: 5W
效率: ~75%
尺寸: 直径 20mm
价格: ¥5-8
```

### BOM 成本估算

| 组件 | 单价 | 数量 | 小计 |
|------|------|------|------|
| ESP32-C3-MINI-1 | ¥10 | 1 | ¥10 |
| 1.3寸 AMOLED | ¥30 | 1 | ¥30 |
| 300mAh 电池 | ¥12 | 1 | ¥12 |
| Qi 接收模块 | ¥6 | 1 | ¥6 |
| PCB 板 | ¥5 | 1 | ¥5 |
| 外壳（钛合金） | ¥50 | 1 | ¥50 |
| 其他元件 | ¥10 | 1 | ¥10 |
| **合计** | | | **¥123** |

## 💻 软件架构

### 固件结构

```
stock-pendant-firmware/
├── main/
│   ├── app_main.c          # 主入口
│   ├── wifi_manager.c      # WiFi 管理
│   ├── ble_manager.c       # BLE 管理
│   ├── stock_fetcher.c     # 股票数据获取
│   ├── display_driver.c    # 显示驱动
│   ├── ui_manager.c        # UI 管理
│   ├── power_manager.c     # 电源管理
│   └── config_manager.c    # 配置存储
├── components/
│   ├── lvgl/               # UI 框架
│   ├── cJSON/              # JSON 解析
│   └── nvs_flash/          # 非易失存储
└── CMakeLists.txt
```

### 核心代码示例

#### 1. 股票数据获取

```c
// stock_fetcher.c
#include "esp_http_client.h"
#include "cJSON.h"

#define STOCK_API_URL "https://api.example.com/stock/quote"

typedef struct {
    char symbol[10];      // 股票代码
    char name[20];        // 股票名称
    float price;          // 当前价
    float change;         // 涨跌额
    float change_pct;     // 涨跌幅
    uint32_t volume;      // 成交量
} stock_quote_t;

esp_err_t fetch_stock_quote(const char *symbol, stock_quote_t *quote) {
    char url[128];
    snprintf(url, sizeof(url), "%s?symbol=%s", STOCK_API_URL, symbol);
    
    esp_http_client_config_t config = {
        .url = url,
        .method = HTTP_METHOD_GET,
    };
    
    esp_http_client_handle_t client = esp_http_client_init(&config);
    esp_err_t err = esp_http_client_perform(client);
    
    if (err == ESP_OK) {
        char *response = malloc(esp_http_client_get_content_length(client) + 1);
        esp_http_client_read_response(client, response, 
            esp_http_client_get_content_length(client));
        
        // 解析 JSON
        cJSON *root = cJSON_Parse(response);
        if (root) {
            strncpy(quote->symbol, cJSON_GetObjectItem(root, "symbol")->valuestring, 9);
            strncpy(quote->name, cJSON_GetObjectItem(root, "name")->valuestring, 19);
            quote->price = cJSON_GetObjectItem(root, "price")->valuedouble;
            quote->change = cJSON_GetObjectItem(root, "change")->valuedouble;
            quote->change_pct = cJSON_GetObjectItem(root, "change_pct")->valuedouble;
            quote->volume = cJSON_GetObjectItem(root, "volume")->valueint;
            cJSON_Delete(root);
        }
        free(response);
    }
    
    esp_http_client_cleanup(client);
    return err;
}
```

#### 2. 显示 UI 设计

```c
// ui_manager.c
#include "lvgl.h"

// 颜色定义
#define COLOR_UP    lv_color_hex(0xFF4444)  // 红色 - 上涨
#define COLOR_DOWN  lv_color_hex(0x00CC00)  // 绿色 - 下跌
#define COLOR_FLAT  lv_color_hex(0xFFFFFF)  // 白色 - 平盘
#define COLOR_BG    lv_color_hex(0x000000)  // 黑色背景

// UI 元素
static lv_obj_t *label_name;
static lv_obj_t *label_price;
static lv_obj_t *label_change;
static lv_obj_t *label_time;
static lv_obj_t *arc_indicator;

void ui_init(void) {
    // 创建主屏幕
    lv_obj_t *scr = lv_scr_act();
    lv_obj_set_style_bg_color(scr, COLOR_BG, LV_PART_MAIN);
    
    // 股票名称
    label_name = lv_label_create(scr);
    lv_obj_set_style_text_color(label_name, lv_color_hex(0x888888), 0);
    lv_obj_set_style_text_font(label_name, &lv_font_montserrat_14, 0);
    lv_obj_align(label_name, LV_ALIGN_TOP_MID, 0, 20);
    
    // 价格（大字体）
    label_price = lv_label_create(scr);
    lv_obj_set_style_text_color(label_price, COLOR_FLAT, 0);
    lv_obj_set_style_text_font(label_price, &lv_font_montserrat_28, 0);
    lv_obj_align(label_price, LV_ALIGN_CENTER, 0, -10);
    
    // 涨跌幅
    label_change = lv_label_create(scr);
    lv_obj_set_style_text_font(label_change, &lv_font_montserrat_14, 0);
    lv_obj_align(label_change, LV_ALIGN_CENTER, 0, 20);
    
    // 时间
    label_time = lv_label_create(scr);
    lv_obj_set_style_text_color(label_time, lv_color_hex(0x666666), 0);
    lv_obj_set_style_text_font(label_time, &lv_font_montserrat_12, 0);
    lv_obj_align(label_time, LV_ALIGN_BOTTOM_MID, 0, -10);
    
    // 涨跌指示环
    arc_indicator = lv_arc_create(scr);
    lv_obj_set_size(arc_indicator, 200, 200);
    lv_obj_align(arc_indicator, LV_ALIGN_CENTER, 0, 0);
    lv_arc_set_bg_angles(arc_indicator, 0, 360);
    lv_arc_set_angles(arc_indicator, 0, 0);
    lv_obj_set_style_arc_color(arc_indicator, COLOR_UP, LV_PART_INDICATOR);
    lv_obj_remove_style(arc_indicator, NULL, LV_PART_KNOB);
    lv_obj_clear_flag(arc_indicator, LV_OBJ_FLAG_CLICKABLE);
}

void ui_update_stock(const stock_quote_t *quote) {
    // 更新名称
    lv_label_set_text(label_name, quote->name);
    
    // 更新价格
    char price_str[16];
    snprintf(price_str, sizeof(price_str), "%.2f", quote->price);
    lv_label_set_text(label_price, price_str);
    
    // 更新涨跌幅和颜色
    char change_str[32];
    lv_color_t color;
    
    if (quote->change_pct > 0) {
        snprintf(change_str, sizeof(change_str), "+%.2f%%", quote->change_pct);
        color = COLOR_UP;
    } else if (quote->change_pct < 0) {
        snprintf(change_str, sizeof(change_str), "%.2f%%", quote->change_pct);
        color = COLOR_DOWN;
    } else {
        snprintf(change_str, sizeof(change_str), "0.00%%");
        color = COLOR_FLAT;
    }
    
    lv_label_set_text(label_change, change_str);
    lv_obj_set_style_text_color(label_price, color, 0);
    lv_obj_set_style_text_color(label_change, color, 0);
    
    // 更新指示环（涨跌幅映射到角度）
    uint16_t angle = (uint16_t)(fabs(quote->change_pct) * 3.6);  // 1% = 3.6度
    if (angle > 360) angle = 360;
    lv_arc_set_angles(arc_indicator, 0, angle);
    lv_obj_set_style_arc_color(arc_indicator, color, LV_PART_INDICATOR);
    
    // 更新时间
    time_t now;
    struct tm timeinfo;
    time(&now);
    localtime_r(&now, &timeinfo);
    char time_str[16];
    strftime(time_str, sizeof(time_str), "%H:%M:%S", &timeinfo);
    lv_label_set_text(label_time, time_str);
}
```

#### 3. BLE 蓝牙提醒

```c
// ble_manager.c
#include "esp_bt.h"
#include "esp_gap_ble_api.h"
#include "esp_gatts_api.h"

// BLE 服务 UUID
#define STOCK_SERVICE_UUID      0x1820
#define STOCK_NOTIFY_CHAR_UUID  0x2A37

// 提醒类型
typedef enum {
    ALERT_PRICE_UP,      // 价格上涨
    ALERT_PRICE_DOWN,    // 价格下跌
    ALERT_VOLUME_HIGH,   // 成交量放大
    ALERT_CUSTOM,        // 自定义条件
} alert_type_t;

// 提醒结构
typedef struct {
    alert_type_t type;
    char symbol[10];
    float threshold;
    char message[64];
} stock_alert_t;

// 发送提醒到手机
void send_alert_to_phone(const stock_alert_t *alert) {
    char json[128];
    snprintf(json, sizeof(json),
        "{\"type\":%d,\"symbol\":\"%s\",\"threshold\":%.2f,\"msg\":\"%s\"}",
        alert->type, alert->symbol, alert->threshold, alert->message);
    
    // 通过 BLE Notify 发送
    esp_ble_gatts_send_indicate(
        gl_profile_tab[PROFILE_APP_IDX].gatts_if,
        gl_profile_tab[PROFILE_APP_IDX].conn_id,
        gl_profile_tab[PROFILE_APP_IDX].char_handle,
        strlen(json), (uint8_t *)json, false
    );
}
```

#### 4. 电源管理

```c
// power_manager.c
#include "esp_pm.h"
#include "esp_sleep.h"

// 电源模式
typedef enum {
    POWER_MODE_ACTIVE,      // 活跃模式
    POWER_MODE_NORMAL,      // 普通模式
    POWER_MODE_SAVER,       // 省电模式
    POWER_MODE_SLEEP,       // 深度睡眠
} power_mode_t;

// 配置电源管理
void power_init(void) {
    // 配置动态频率调节
    esp_pm_config_t pm_config = {
        .max_freq_mhz = 160,
        .min_freq_mhz = 80,
        .light_sleep_enable = true,
    };
    esp_pm_configure(&pm_config);
}

// 进入深度睡眠
void enter_deep_sleep(uint64_t sleep_time_sec) {
    printf("Entering deep sleep for %llu seconds...\n", sleep_time_sec);
    
    // 配置唤醒源：定时器唤醒
    esp_sleep_enable_timer_wakeup(sleep_time_sec * 1000000ULL);
    
    // 关闭外设
    gpio_set_level(GPIO_NUM_2, 0);  // 关闭 LED
    
    // 进入深度睡眠
    esp_deep_sleep_start();
}

// 计算续航时间
uint32_t estimate_battery_life(power_mode_t mode, uint32_t battery_mah) {
    uint32_t current_ma;
    
    switch (mode) {
        case POWER_MODE_ACTIVE:
            current_ma = 120;  // WiFi + 显示全亮
            break;
        case POWER_MODE_NORMAL:
            current_ma = 50;   // 定时刷新
            break;
        case POWER_MODE_SAVER:
            current_ma = 20;   // 低亮度 + 低频刷新
            break;
        case POWER_MODE_SLEEP:
            current_ma = 1;    // 深度睡眠
            break;
    }
    
    return battery_mah / current_ma;  // 返回小时数
}
```

## 📱 配套 App 设计

### 功能列表

```
App (Flutter)
├── 设备配对
│   ├── BLE 扫描
│   ├── WiFi 配置
│   └── 绑定设备
├── 股票管理
│   ├── 自选股列表
│   ├── 搜索添加
│   └── 排序删除
├── 提醒设置
│   ├── 涨跌幅阈值
│   ├── 价格区间
│   ├── 成交量条件
│   └── 免打扰时段
├── 显示设置
│   ├── 亮度调节
│   ├── 刷新频率
│   ├── 显示模式
│   └── 字体大小
└── 数据统计
    ├── 今日行情
    ├── 历史记录
    └── 收益统计
```

### BLE 通信协议

```json
// App → 吊坠：配置同步
{
  "cmd": "config",
  "stocks": ["sh600519", "sz000858", "sh601318"],
  "refresh_interval": 30,
  "alerts": [
    {"symbol": "sh600519", "type": "price_up", "threshold": 1800}
  ]
}

// 吊坠 → App：状态上报
{
  "cmd": "status",
  "battery": 85,
  "wifi_rssi": -45,
  "stocks": [
    {"symbol": "sh600519", "price": 1788.50, "change": 1.25}
  ]
}
```

## 🎨 外观设计

### 尺寸规格

```
        ┌─────────────┐
        │   25mm      │
        │  ┌───────┐  │
        │  │       │  │
        │  │ OLED  │  │ 30mm
        │  │       │  │
        │  └───────┘  │
        │             │
        └─────────────┘
        
        厚度: 8mm
        重量: 25g
        材质: 钛合金 + 蓝宝石玻璃
```

### 外壳设计

```python
# OpenSCAD 示例代码
$fn = 100;

module pendant_case() {
    difference() {
        // 外壳主体
        cylinder(h=8, d=30, center=true);
        
        // 屏幕开孔
        translate([0, 0, 2])
            cylinder(h=5, d=24, center=true);
        
        // 内部空间
        translate([0, 0, -1])
            cylinder(h=7, d=27, center=true);
        
        // 充电线圈开孔
        translate([0, 0, -3])
            cylinder(h=2, d=20, center=true);
    }
    
    // 挂绳孔
    translate([0, 15, 0])
        rotate([90, 0, 0])
            cylinder(h=5, d=3, center=true);
}

pendant_case();
```

## 🔋 功耗优化策略

### 刷新策略

```c
// 智能刷新逻辑
void smart_refresh(void) {
    static uint32_t last_refresh = 0;
    static uint32_t refresh_interval = 30;  // 默认30秒
    
    uint32_t now = get_timestamp();
    
    // 交易时段：高频刷新
    if (is_trading_hours()) {
        refresh_interval = 15;  // 15秒刷新
    } 
    // 盘前盘后：低频刷新
    else if (is_pre_post_market()) {
        refresh_interval = 60;  // 1分钟刷新
    }
    // 非交易日：极低频
    else {
        refresh_interval = 300;  // 5分钟刷新
    }
    
    // 有提醒时：立即刷新
    if (has_pending_alert()) {
        refresh_interval = 0;
    }
    
    if (now - last_refresh >= refresh_interval) {
        refresh_stock_data();
        last_refresh = now;
    }
}
```

### 续航预估

| 使用场景 | 刷新频率 | 续航时间 |
|----------|----------|----------|
| 激进模式 | 10秒 | ~8小时 |
| 标准模式 | 30秒 | ~24小时 |
| 省电模式 | 2分钟 | ~3天 |
| 纯手表模式 | 关闭 | ~7天 |

## 🚀 开发计划

### Phase 1：原型验证（2周）

- [ ] ESP32-C3 开发板测试
- [ ] 屏幕驱动调试
- [ ] WiFi 联网测试
- [ ] 股票 API 对接

### Phase 2：功能开发（3周）

- [ ] UI 界面实现
- [ ] BLE 通信协议
- [ ] 电源管理优化
- [ ] 手机 App 开发

### Phase 3：硬件设计（2周）

- [ ] PCB 设计
- [ ] 外壳 3D 打印
- [ ] 组装测试
- [ ] 功耗优化

### Phase 4：小批量（2周）

- [ ] PCB 打样
- [ ] 外壳 CNC 加工
- [ ] 组装调试
- [ ] 用户测试

## 💰 成本与定价

| 项目 | 成本 |
|------|------|
| 硬件 BOM | ¥123 |
| PCB + 组装 | ¥30 |
| 外壳加工 | ¥80 |
| 包装配件 | ¥20 |
| **总成本** | **¥253** |

**建议零售价**：¥499-699

## 📝 待解决问题

- [ ] 股票 API 选择（免费/付费）
- [ ] 屏幕常亮 vs 抬腕亮屏
- [ ] 防水等级（IP67？）
- [ ] 充电方式（无线 vs Type-C）
- [ ] 外观材质（钛合金 vs 不锈钢 vs 塑料）

## 🔗 相关资源

- [ESP32-C3 官方文档](https://docs.espressif.com/projects/esp-idf/en/latest/esp32c3/)
- [LVGL UI 框架](https://docs.lvgl.io/)
- [Qi 无线充电标准](https://www.wirelesspowerconsortium.com/)
- [Flutter BLE 插件](https://pub.dev/packages/flutter_blue)

---

*赛博文西出品 · 发明 002*
*让看盘变得优雅* 💎
