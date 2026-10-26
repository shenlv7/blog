---
title: "环境搭建：版本、SDK、模拟器、真机、签名"
description: "从华为开发者官网下载 DevEco Studio，不要用来路不明的安装包。Windows 10/11 64 位或 macOS，内存建议 16 GB 以上，磁盘预留 20 GB 以上；模拟器需要…"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 2
minutes: 5
---

### 2.1 下载与版本选择

从华为开发者官网下载 DevEco Studio，不要用来路不明的安装包。Windows 10/11 64 位或 macOS，内存建议 16 GB 以上，磁盘预留 20 GB 以上；模拟器需要开启硬件虚拟化。安装时勾选 HarmonyOS SDK，登录华为开发者账号。

本指南基线是 DevEco Studio 5.0 及以上、API 12 及以上。新项目创建时选 **Empty Ability**、**Stage**、**ArkTS**、`compatibleSdkVersion` 12。用 5.1 或 6.x 的 IDE 也可以，但要在 SDK Manager 里装 API 12，否则 `compatibleSdkVersion: "5.0.0(12)"` 编译不过；如果新版 IDE 已经不再提供 API 12 SDK，就换回仍支持 API 12 的 5.0.x，别硬改配置。不要为了用新 API 把最低版本抬到 18 或 20，除非你确认目标用户设备都满足。

### 2.2 SDK 与 API 版本怎么选

三个概念别混：

- `compileSdkVersion`：编译时用的 SDK，决定你能调用哪些 API。
- `compatibleSdkVersion`：应用能安装的最低系统版本，写进 `build-profile.json5` 的 `products` 里，最常用。
- `targetSdkVersion`：目标行为版本，影响系统对权限、后台、隐私等行为的兼容策略。

新项目把 `compatibleSdkVersion` 设为 12，`compileSdkVersion` 用 IDE 已安装的 API 12 或更高；`targetSdkVersion` 设为你实际测过的版本。常见组合是“高编译、低兼容”：用 API 18 的 SDK 编译，但 `compatibleSdkVersion` 仍是 12，这样能装到 5.0 设备；代价是代码里不能随便调用 API 18 才有的接口，否则低版本设备运行时会报错。模板通常只显式写 `compatibleSdkVersion`，`compileSdkVersion` 由 IDE 和已安装 SDK 决定；需要显式指定时，值必须与已安装 SDK 匹配。字段的可用性和写法随 IDE 版本变化，具体以官方 build-profile 文档为准。

### 2.3 模拟器与真机调试的差别

模拟器适合跑 UI、路由、状态管理和大部分本地数据逻辑，启动快、不占真机。它不适合验证相机、蓝牙、NFC、推送、支付、传感器、真实性能和最终签名安装。真机调试要先在设备上开启“开发者模式”和“USB 调试”，连接后 `hdc list targets` 能看到设备；`hdc shell hilog` 看日志，`hdc install xxx.hap` 手动安装。如果 `hdc list targets` 为空，先换数据线、确认系统驱动、在设备上重新授权 USB 调试；IDE 看不到但 hdc 能看到，重启 hdc 服务或 IDE。真机还要注意锁屏、授权弹窗，以及 USB 模式不能是“仅充电”。

真机的第一个门槛不是代码，是签名。HAP 在安装时校验签名，调试签名和发布签名不是一回事。

### 2.4 签名为什么是新手第一个坑

调试签名由 DevEco 自动生成，通常需要登录华为账号；调试 Profile 会绑定设备 UDID，换一台真机就可能要重新生成。发布签名要自己在 AppGallery Connect（AGC）创建证书和 Profile。常见报错不是“代码错了”，而是 bundleName、证书、Profile、UDID 四者有一个对不上。

| 文件 | 是什么 | 谁生成 | 关键点 |
| --- | --- | --- | --- |
| `.p12` | PKCS#12 密钥库，含私钥和证书链 | DevEco 的 Generate Key and CSR，或 keytool | 绝不要提交到 Git；密码丢了只能重签 |
| `.cer` | X.509 公钥证书 | 上传 CSR 后由 AGC 签发 | 是公开文件，但必须与 `.p12` 配对 |
| `.p7b` / Profile | 描述文件，绑定 bundleName、证书、设备 UDID、权限 | AGC 创建并下载 | 调试 Profile 有设备列表，发布 Profile 没有 |
| `.app` | APP Pack，包含一个或多个 HAP 和 pack.info | DevEco Build APP(s) | 上架 AGC 用 `.app`，不是单个 `.hap` |

Debug 与 Release 的差别：Debug 用调试证书和调试 Profile，方便安装到有限设备；Release 用发布证书和发布 Profile，用于 AGC 上架。**Profile 不是证书**，它是“允许谁、在哪些设备、用哪个证书运行哪个应用”的授权文件。签名配置在 DevEco 的 Project Structure > Signing Configs 里填：Store File 选 `.p12`，填 Store Password、Key Alias、Key Password，Profile 选 `.p7b`，Certpath 选 `.cer`。

签名排错按“四件套同源”查：`.p12` 里的私钥、`.cer` 的公钥、`.p7b` 绑定的证书指纹、AGC 应用记录的指纹，必须是同一套；`bundleName` 必须与 AGC 应用一致；真机调试还要确认 UDID 在调试 Profile 里。换电脑、换证书、换包名、Profile 过期，都会让原本能跑的工程突然装不上。发布证书的私钥丢了不能找回，只能新建证书重签，所以 `.p12` 和密码要放进密码管理器，不要进 Git。
