---
title: "四周学习路径"
description: ""
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 11
minutes: 5
---

**第 1 周：环境与声明式 UI。** 装 DevEco、API 12 SDK，跑通模板；把 `StateDemo` 敲一遍；理解 `@State/@Prop/@Link/@Builder`。产出物：一个三页静态应用，能跳转、能返回。自检：能解释 `build()` 为什么不是命令式；能独立排出“页面没注册”“资源名写错”两类错。

**第 2 周：状态、路由、本地数据。** 做 Todo 的列表和详情，用 Preferences 持久化；把 Navigation 的传参和返回传值跑通。产出物：可增删改查、杀进程不丢数据的 Todo。自检：不看清屏代码能说出 `pushPathByName` 和 `pop` 的数据流；能处理空列表、空标题、非法 id。

**第 3 周：网络、权限、V2、深色模式。** 给 Todo 加一个“从服务端拉取待办”的按钮，用 `http` 请求并处理超时/非 200；加相机权限申请；把 V1 示例迁一个组件到 V2；补 `resources/dark`。产出物：带网络错误提示、权限拒绝提示、深色适配的版本。自检：飞行模式下不崩；拒绝权限后再点有提示；V1/V2 不混在同一组件。

**第 4 周：签名与上架。** 在 AGC 创建应用、证书、Profile；配置 Release 签名；打 `.app`；准备隐私政策、截图、更新说明；提交审核或至少走到“待提交”。产出物：可安装的 Release 包 + AGC 应用草稿。自检：Release 包能真机安装；`versionCode` 已递增；证书指纹与 AGC 一致；隐私政策 URL 可公开访问。
