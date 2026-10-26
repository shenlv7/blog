---
title: "新手高频坑清单"
description: "1. 现象： 编译报 Cannot find module '@kit.XXX'。原因： SDK 没装或 compatibleSdkVersion 低于该 Kit 的起始 API。解决： SD…"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 12
minutes: 5
---

1. **现象：** 编译报 `Cannot find module '@kit.XXX'`。**原因：** SDK 没装或 `compatibleSdkVersion` 低于该 Kit 的起始 API。**解决：** SDK Manager 装对应 API，按报错把最低版本调到 API 12/更高，别只改 import。
2. **现象：** `@Link` 子组件不更新。**原因：** 父组件传了 `this.count` 而不是 `$count`。**解决：** V1 的 `@Link` 用 `$count` 传引用；V2 改用 `@Param + @Event`。
3. **现象：** 子组件改了 `@Prop`，父组件没变。**原因：** `@Prop` 是单向的。**解决：** 需要回传就换 `@Link`，或让子组件发 `@Event` 由父组件改。
4. **现象：** `@Consume` 运行时报找不到 Provider。**原因：** 祖先没有同 key 的 `@Provide`，或 key 拼错。**解决：** 在最近公共祖先加 `@Provide('key')`，子组件 `@Consume('key')` 完全一致。
5. **现象：** `router.pushUrl` 后白屏/404。**原因：** 页面没写进 `main_pages.json`，或路径带了 `.ets`。**解决：** 注册 `pages/Detail` 这种无扩展名路径；Navigation 子页不要往 `main_pages` 里塞。
6. **现象：** Navigation 详情页白屏。**原因：** builder 根节点不是 `NavDestination`，或外面包了带属性/事件的自定义组件。**解决：** 让 `NavDestination` 做根；自定义组件只通过参数传数据。
7. **现象：** 真机安装报 SDK 版本不匹配。**原因：** `compatibleSdkVersion` 高于设备 API。**解决：** 降到设备支持的版本，或换设备；不要靠改 `targetSdkVersion` 糊过去。
8. **现象：** 安装报签名/Profile 错误。**原因：** bundleName、证书、Profile、UDID 有一个对不上。**解决：** 用 AGC 重新生成对应 Profile；确认 DevEco 里 `.p12/.cer/.p7b` 是同一套。
9. **现象：** Preferences 写成功，重启丢了。**原因：** 只 `put` 没 `flush`，或进程被强杀前没落盘。**解决：** 每次写关键数据后 `await flush()`；不要用它存大列表。
10. **现象：** RDB 查询越来越卡/内存涨。**原因：** `ResultSet` 没 `close()`，或一次查太多。**解决：** `finally` 里关闭；分页查询，单次别超 5000 条。
11. **现象：** 模拟器 HTTP 正常，真机失败。**原因：** 没声明 `INTERNET` 权限、域名证书问题、明文 HTTP 被限制。**解决：** 加权限；优先 HTTPS；检查网络安全配置和服务端证书。
12. **现象：** 权限弹窗只出一次，之后没反应。**原因：** 用户已拒绝，系统不再重复弹。**解决：** 首次申请后处理拒绝；API 12 起用 `requestPermissionOnSetting` 引导去设置页。
13. **现象：** 深色模式没变化。**原因：** 颜色写死 `#FFFFFF`，没走资源。**解决：** 用 `$r('app.color.xxx')`，补 `resources/dark/element/color.json`。
14. **现象：** 同一个组件加 V2 装饰器编译报错。**原因：** V1/V2 装饰器混用。**解决：** 整个组件迁到 V2，或整个留在 V1；不要一半一半。
15. **现象：** Release 包上架被拒。**原因：** `versionCode` 没加、用了 Debug 包、证书指纹不一致、隐私政策缺失。**解决：** 逐项对照 AGC 检查清单；Release 包先在真机装一遍。
16. **现象：** 写 `any` 或 `var` 编译不过。**原因：** ArkTS 静态限制。**解决：** 用具体类/接口/联合类型，`let`/`const` 替代 `var`；`JSON.parse` 的结果用 `as 具体类型`。
17. **现象：** 模拟器测不了相机/蓝牙/推送。**原因：** 模拟器能力与真机不同。**解决：** 这些能力必须上真机；模拟器只做 UI 和逻辑回归。
