---
title: "工程结构：每个文件改什么会导致什么"
description: "新建 Stage 工程后，先认识这棵树："
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 3
minutes: 5
---

新建 Stage 工程后，先认识这棵树：

```text
MyTodo/
├── AppScope/
│   ├── app.json5
│   └── resources/base/{element,media}
├── entry/
│   ├── src/main/
│   │   ├── ets/
│   │   │   ├── entryability/EntryAbility.ets
│   │   │   ├── pages/Index.ets
│   │   │   ├── components/
│   │   │   └── model/
│   │   ├── resources/
│   │   │   ├── base/{element,media,profile}
│   │   │   ├── dark/element/
│   │   │   └── rawfile/
│   │   └── module.json5
│   ├── build-profile.json5
│   └── oh-package.json5
├── build-profile.json5
├── oh-package.json5
└── hvigorfile.ts
```

**AppScope/app.json5**：全局应用配置。`bundleName` 是应用唯一身份，上架后不能改；改了就是另一个应用，签名和 AGC 记录全部对不上。`versionCode` 必须是递增整数，每次发版都要加；`versionName` 是给人看的。`icon`、`label` 指向 AppScope 的资源，删了图标或改错资源名会编译失败。

**entry/**：默认的入口模块。`src/main/ets` 放代码，`src/main/resources` 放资源，`src/main/module.json5` 描述模块。一个工程可以有多个 Module，但入口模块通常只有一个。

**module.json5**：模块级清单。`name` 是模块名，`type` 是 `entry` 或 `feature`，`mainElement` 指向入口 UIAbility，`deviceTypes` 决定能装到哪些设备，`pages` 指向页面清单，`abilities` 里是 UIAbility 配置，`requestPermissions` 声明权限。改错 `srcEntry` 会找不到 Ability；`deviceTypes` 少了目标设备，设备上装不上；`pages` 指错，首屏白屏或路由 404。

**根 build-profile.json5**：应用级构建配置。`signingConfigs` 放签名材料，`products` 定义产物（`compatibleSdkVersion`、`runtimeOS`），`buildModeSet` 定义 debug/release，`modules` 把模块挂到产物上。把 `compatibleSdkVersion` 调低，可能装到老设备但运行到新 API 时崩溃；调高，老设备直接装不上。

**entry/build-profile.json5**：模块级构建配置，常见的是 `apiType: "stageMode"`、`targets`、`buildOption`、混淆规则。改 `targets` 会影响产物名和测试目标；改混淆规则会影响 Release 包，调试包一般不受影响。

**oh-package.json5**：包管理与依赖声明。根目录和模块目录都可能有这个文件，根依赖全工程可用，模块依赖只在该模块可用；第三方包写进 `dependencies`，执行 `ohpm install`，`oh-package-lock.json5` 锁版本，不要手改。`@kit.*`、`@ohos.*` 是系统 Kit，不要写进 dependencies。把依赖从根挪到模块、或反过来，可能造成“编译时能找到、运行时找不到”的模块解析问题。

**resources/**：`base/element/string.json` 放字符串，`color.json` 放颜色，`media` 放图片，`profile/main_pages.json` 放页面清单，`rawfile` 放不编译的原始文件；`dark/element/color.json` 是深色模式资源，`en_US`、`zh_CN` 等是语言限定词目录。系统按设备语言、深浅色、屏幕密度等限定词自动选最匹配的资源，找不到就回退到 `base`。`rawfile` 不参与资源编译，要用 `$rawfile('config.json')` 或文件接口读。资源名改了，所有 `$r('app.type.name')` 引用都要跟着改，否则编译报“资源不存在”。

**main_pages.json**：页面路由表。路径相对 `src/main/ets`，不带 `.ets`，例如 `pages/Index`。用 `router` 时每个页面都必须在这里注册；用 `Navigation` 的 `NavDestination` 时，子页面不在这里注册，只有首页在。用系统路由表时，还要在 `module.json5` 配 `routerMap` 并添加 `route_map.json`（API 12 起可用，具体以官方文档为准）。
