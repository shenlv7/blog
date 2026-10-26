---
title: "打包发布：从证书到 AGC"
description: "发布前先做六件事：把 bundleName 改成自己的域名反写；versionCode 加一；versionName 改成发布版本；检查应用图标、名称、启动图；删掉测试地址和调试日志；确认隐私…"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 9
minutes: 5
---

发布前先做六件事：把 `bundleName` 改成自己的域名反写；`versionCode` 加一；`versionName` 改成发布版本；检查应用图标、名称、启动图；删掉测试地址和调试日志；确认隐私政策、权限说明和收集个人信息清单。Release 构建前，最好在真机上用 Release 签名装一遍，不要只跑 Debug。

签名流程：

1. 在 AGC 完成开发者实名认证，创建应用，包名与 `bundleName` 完全一致。
2. 在 DevEco 选 Build > Generate Key and CSR，生成 `.p12` 和 `.csr`；记下 Key Alias 和两个密码。
3. 在 AGC 的证书管理里上传 `.csr`，下载 `.cer`。
4. 在 AGC 创建 Profile：选应用、选发布证书，发布 Profile 不绑定设备；下载 `.p7b`。
5. 在 DevEco Project Structure > Signing Configs 取消自动签名，填入 `.p12`、`.cer`、`.p7b` 和密码。
6. Build > Build Hap(s)/APP(s) > Build APP(s)，在 Build Output 面板找 `.app`。常见输出在工程 `build/default/outputs/` 下，具体路径和文件名以 IDE 显示为准。

AGC 上架的关键材料：应用名称与包名、应用图标、简介、截图、隐私政策 URL、用户隐私保护指引、内容分级、版权信息、版本号和更新说明。图标、截图要按 AGC 页面要求的尺寸和格式准备，截图必须反映真实界面。涉及登录的提供测试账号；涉及支付、推送、地图、健康等能力的，按品类补充资质。国内分发通常还需要 APP 备案，特殊品类可能要求软件著作权或行业资质。**这些材料随地区、品类和 AGC 页面调整，以 AGC 当前必填项和官方发布文档为准。**

审核时间不由你控制，能提前做的是：在 AGC 先建应用草稿、把签名和包名一次配对、用内部测试轨道装一遍、把隐私政策放到公网可访问的 HTTPS 地址、写清楚每个敏感权限的用途。证书和 Profile 都有有效期，换证书后所有历史版本仍用旧证书，新版本必须用新 Profile 重签；不要把“上次能传”当成“这次也能传”。

常见被拒：`versionCode` 没递增；证书指纹和 AGC 配置不一致；Profile 的 bundleName 对不上；用了 Debug 包；隐私政策打不开；申请了与功能无关的权限；截图和实际界面不符。打包成功后，Release HAP 先用 `hdc install` 或 DevEco 在真机装一遍；`.app` 留给 AGC 上传。Debug 包和 Release 包可以用同一个 bundleName，但证书不同；同一台设备上换签名安装前先卸载旧包，避免签名冲突。
