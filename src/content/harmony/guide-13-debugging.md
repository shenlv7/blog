---
title: "常用调试手段：什么时候用哪个"
description: ""
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 13
minutes: 5
---

**hilog 日志**：最适合看业务流程、网络返回、生命周期和真机问题。`import { hilog } from '@kit.PerformanceAnalysisKit';`，`hilog.info(0x0000, 'Todo', 'value=%{public}s', value)`。`%{public}s` 才会在日志里显示明文，隐私数据不要打。命令行 `hdc shell hilog | grep Todo`，IDE 的 Log 窗口按 tag/级别过滤。日志不要刷屏，关键分支打一条就够。

**断点**：适合状态为什么不对、回调有没有进、循环边界。DevEco 用 Debug 启动，行号点红点；可加条件断点，比如 `item.id === 'xxx'`；配合 Watches 看 `this.todos`、`popInfo.result`。UI 不刷新时，先断点确认状态变量到底有没有变，再怀疑框架。

**DevEco Testing**：适合回归、稳定性、性能、兼容性这类“跑很多遍”的测试。写完 Todo 后可以用它做启动、页面遍历、重复操作；上架前跑一轮比手点可靠。它不替代断点，定位到具体行还是要回 IDE。

**Profiler**：适合卡顿、掉帧、内存涨、耗电。用真机跑，先录一段典型操作（列表滚动、频繁增删），看 CPU、内存、帧率、能耗；不要只看平均值，找尖峰和 GC。优化前先录基线，改完再录一次对比，否则很容易“感觉快了”。
