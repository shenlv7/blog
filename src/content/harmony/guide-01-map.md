---
title: "先建立三个坐标：DevEco Studio、ArkTS/ArkUI、Stage 模型"
description: "你写的每个 .ets 文件，最终由 DevEco Studio 调用 hvigor 编译成 HAP，签名后装到设备；运行时由 Stage 模型管理 UIAbility 和窗口，ArkUI 把 …"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 1
minutes: 5
---

你写的每个 `.ets` 文件，最终由 DevEco Studio 调用 hvigor 编译成 HAP，签名后装到设备；运行时由 Stage 模型管理 UIAbility 和窗口，ArkUI 把 `build()` 里的声明渲染成界面。这三件事分别对应工具链、语言与 UI、运行模型。

**DevEco Studio** 不只是编辑器。它集成 SDK Manager、模拟器、hvigor 构建、ohpm 包管理、签名配置、日志、断点和 Profiler。命令行能做的事它基本都能做，但新手先把 IDE 的 Build、Run、Signing Configs、Device Manager 四条链路跑通，比背命令有用。

**ArkTS 与 ArkUI** 是两层：ArkTS 是语言，ArkUI 是声明式 UI 框架。一句话说清 ArkTS 与 TypeScript 的关系和差异：ArkTS 以 TypeScript 为基础，但它是为 ArkUI 和方舟运行时裁剪过的静态子集，保留类、接口、泛型、async/await 等，禁用 `any`/`unknown`、`var`、`Symbol`、结构化类型、索引签名、tuple、`delete`/`in`、正则字面量、条件类型、`as const`、构造函数参数属性等动态特性，并要求对象字面量和泛型调用有明确类型；不要把任意 TS 代码直接搬进来。

**Stage 模型** 是 HarmonyOS NEXT 的唯一应用模型。一个应用由多个 Module 组成，每个 Module 编译成一个 HAP；每个 HAP 里可以有 UIAbility、ExtensionAbility、页面和资源。UIAbility 是带界面的应用组件，`onWindowStageCreate` 拿到 `WindowStage`，再 `loadContent('pages/Index')` 加载首个页面。一个 entry 模块通常只有一个入口 UIAbility，启动链路是 `onCreate` → `onWindowStageCreate` → `onForeground`，退到后台走 `onBackground`，真正销毁才走 `onWindowStageDestroy` 和 `onDestroy`。页面生命周期和 Ability 前后台不是一回事：Ability 还在前台，页面也可能因为 Navigation 入栈而被盖住；别把 `onForeground` 当成页面显示回调。排错时先分清是 Ability、窗口还是页面没走。
