---
title: "第 1 讲：装环境与跑通第一个页面"
description: "装好 DevEco Studio 与 API 12 SDK，跑通一个 Hello 页面，并说清工程里每个目录是干什么的。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 1
minutes: 5
---

### 本讲目标

装好 DevEco Studio 与 API 12 SDK，跑通一个 Hello 页面，并说清工程里每个目录是干什么的。

做完你会看到：模拟器上出现居中的「习惯打卡」四个字，Build 窗口输出 `BUILD SUCCESSFUL`。

### 为什么这样设计

先不写业务，是因为鸿蒙的失败大多发生在业务之前：SDK 没下全、签名没配、设备没连上。先把这条链路打通，之后每次报错你都能确定是代码问题而不是环境问题。

用模拟器起步，是因为它零成本、可随时重置；但第 9 讲的通知和提醒必须换真机 —— 模拟器不模拟通知栏和提醒代理。这不是偏好问题，是能力边界问题。

工程结构之所以分成 `AppScope/` 和 `entry/` 两层，是因为一个 App 可以包含多个 HAP（比如主包 + 特性包）。`AppScope/` 放整个应用共享的东西（包名、版本、应用图标），`entry/` 是一个独立可安装的模块。你现在只有一个模块，但目录结构已经为多模块留好了位置。

### 跟着做

**1. 装工具。** 从华为开发者官网下载 DevEco Studio 5.0 或更高版本。首次启动时在向导里勾选 SDK：API 12（对应 HarmonyOS 5.0.0），同时装上 Previewer 和 Emulator。

**2. 新建工程。** `File > New > Create Project`，模板选 `Empty Ability`：

- Project name：`HabitTracker`
- Bundle name：`com.dahua.habit`（后面签名、上架都靠它，定了就别改）
- Compile SDK：`5.0.0(12)`
- Model：`Stage`
- Device type：`Phone`
- Language：`ArkTS`

**3. 认识目录。** 建完后 Project 面板是这样：

```text
HabitTracker/
├── AppScope/
│   ├── app.json5                  // 包名、versionCode/versionName、应用图标与名称
│   └── resources/base/media/      // app_icon.png、layered_image.json
├── entry/                         // 一个可独立安装的模块
│   ├── src/main/
│   │   ├── ets/
│   │   │   ├── entryability/EntryAbility.ets   // 应用入口，管窗口生命周期
│   │   │   ├── pages/Index.ets                 // 页面
│   │   │   ├── model/  view/  service/         // 后面几讲会建
│   │   ├── resources/base/profile/main_pages.json  // 页面路由表
│   │   └── module.json5           // 模块名、入口 Ability、权限声明
│   ├── build-profile.json5        // 模块编译配置
│   └── oh-package.json5           // 依赖
├── build-profile.json5            // 工程级：签名、产物、SDK 版本
├── hvigorfile.ts                  // 构建脚本入口
└── oh-package.json5               // 工程级依赖
```

**4. 改第一个页面。** 打开 `entry/src/main/ets/pages/Index.ets`，整体替换为：

```ts
@Entry
@Component
struct Index {
  build() {
    Column({ space: 12 }) {
      Text('习惯打卡')
        .fontSize(30)
        .fontWeight(FontWeight.Bold)
      Text('第 1 讲：跑通第一个页面')
        .fontSize(15)
        .fontColor('#8A8A8E')
    }
    .width('100%')
    .height('100%')
    .justifyContent(FlexAlign.Center)
    .backgroundColor('#F2F2F7')
  }
}
```

`@Entry` 表示这是页面入口，`@Component` 表示这是自定义组件，`build()` 里是声明式 UI。这三行是后面 11 讲的地基。

**5. 配置签名。** `File > Project Structure > Signing Configs`，勾选 `Automatically generate signature`，登录华为账号。DevEco 会自动生成调试用的 p12/cer/p7b 并写回 `build-profile.json5`。

**6. 运行。** 点右上角设备下拉框，`Device Manager` 里创建一个 Phone 模拟器，启动后点绿色三角。首编译约 1–3 分钟。

**7. 接真机（可选但建议现在就试一次）。** 手机打开「设置 > 关于手机」，连点版本号 7 次进入开发者模式，再进「系统和更新 > 开发人员选项」打开 USB 调试，插线后在 DevEco 里选你的设备。命令行验证：

```bash
hdc list targets          # 应列出你的设备
hdc shell aa start -a EntryAbility -b com.dahua.habit
```

### 容易踩的坑

- **现象**：Run 按钮是灰的。**原因**：没配签名，或设备下拉框里没选设备。**改法**：先完成第 5 步，再选设备。
- **现象**：Sync 失败，提示找不到 SDK。**原因**：`local.properties` 里的 `sdk.dir` 路径不对，常见于手动改过安装目录。**改法**：`Settings > SDK` 重新指定路径，或删掉 `local.properties` 让 IDE 重新生成。
- **现象**：改了代码但模拟器上没变。**原因**：增量编译缓存或热重载没生效。**改法**：`Build > Clean Project` 后重跑。
- **现象**：真机插上不识别。**原因**：只开了开发者模式，没开 USB 调试；或数据线只能充电。**改法**：换线、换口，`hdc kill` 后 `hdc start`。

### 自检

- 打开 Project 面板，顶层应同时看到 `AppScope` 和 `entry`。
- 打开 `entry/src/main/module.json5`，`requestPermissions` 应为空数组（此刻还没申请任何权限）。
- 打开 `entry/src/main/resources/base/profile/main_pages.json`，应只有 `"pages/Index"` 一项。
- 模拟器上应看到浅灰背景、居中的「习惯打卡」。
- 打开 Build 窗口，末尾应有 `BUILD SUCCESSFUL`。

### 本讲产出

一个能在模拟器和真机上跑起来的 Stage 模型空工程，包名 `com.dahua.habit`。

### 进阶挑战

30 分钟内：把页面标题改成你自己 App 的名字和一句 slogan；把 `AppScope/app.json5` 里的 `label` 改成中文名；替换 `AppScope/resources/base/media/app_icon.png`（建议 216×216 PNG）。卸载重装，确认桌面图标和名称都变了。

---
