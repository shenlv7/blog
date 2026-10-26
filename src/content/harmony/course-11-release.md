---
title: "第 11 讲：打包与真机/上架"
description: "搞清楚调试签名和发布签名的区别，理解 p12/cer/p7b 的关系，打出可上架的 .app，拿到一份 AGC 材料清单。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 11
minutes: 5
---

### 本讲目标

搞清楚调试签名和发布签名的区别，理解 p12/cer/p7b 的关系，打出可上架的 `.app`，拿到一份 AGC 材料清单。

做完你会看到：`build/outputs/` 下出现签名后的 `.app`，真机可正常安装启动。

### 为什么这样设计

签名解决的是「这个包是谁发的、有没有被改过」。系统在安装时会验证：证书是否可信、Profile 是否匹配当前 bundleName、调试包的设备 UDID 是否在列表里。任何一项不匹配就拒绝安装。

四个文件的关系，用一句话串起来：

- **`.p12`（密钥库）**：装着你的私钥和证书链。它**只留在你本地**，泄露等于别人能冒充你发版。
- **`.csr`（证书请求）**：由 p12 导出，上传给 AGC 换取证书。
- **`.cer`（数字证书）**：华为签发的「身份证」，证明那个公钥属于你。
- **`.p7b`（Profile）**：把 bundleName、证书、设备 UDID（调试用）、权限 ACL 绑在一起的文件，由 AGC 签发。

打包时 hvigor 用 p12 里的私钥签名，把 cer 和 p7b 一起塞进包里。所以「换一台电脑就签不了名」是正常的 —— 缺的是 p12。

调试签名与发布签名的区别不在算法，在**用途约束**：调试 Profile 绑定了设备列表，只能装到列表里的机器；发布 Profile 不限设备，但只能通过 AGC 分发。两者不能混用，也不该共用一套证书。

### 跟着做

**1. 调试签名（真机调试）。** `File > Project Structure > Signing Configs` 勾选自动签名，登录华为账号。DevEco 会自动：生成 p12 → 生成 CSR → 向 AGC 申请 cer → 创建调试 Profile → 把设备 UDID 注册进去（需要手机已登录同一账号）。

**2. 发布签名。** 手动走一遍，理解每步在做什么：

1. AGC 控制台 → 我的应用 → 创建应用，bundleName 必须与 `AppScope/app.json5` 完全一致（创建后不可改）。
2. 「证书、APP ID 和 Profile」→ 创建发布证书，上传第 1 步生成的 CSR，下载 `.cer`。
3. 创建发布 Profile（类型选 release），绑定 bundleName 与上一步的证书，下载 `.p7b`。
4. DevEco 里新建一个 Signing Config，填入 p12 路径、密码、alias、cer、p7b。

`build-profile.json5` 里最终应是这样（密码用 IDE 加密存储，不要明文提交到仓库）：

```json5
"signingConfigs": [
  {
    "name": "release",
    "type": "HarmonyOS",
    "material": {
      "certpath": "sign/release.cer",
      "storePassword": "******",
      "keyAlias": "habit",
      "keyPassword": "******",
      "profile": "sign/release.p7b",
      "signAlg": "SHA256withECDSA",
      "storeFile": "sign/habit.p12"
    }
  }
]
```

**3. 打 `.app`。** `Build > Build App(s)` 或 `Build > Generate Signed App Package`。产物位置以你本地实际目录为准，通常在：

- HAP：`entry/build/default/outputs/default/entry-default-signed.hap`
- App Pack：工程根目录 `build/outputs/default/*.app`

**4. 装到真机验证。**

```bash
hdc install -r entry/build/default/outputs/default/entry-default-signed.hap
hdc shell aa start -a EntryAbility -b com.dahua.habit
```

**5. AGC 上架材料清单。** 提审前把这些准备齐（具体要求以 AGC 提交页面的实时提示为准）：

- 软件著作权证书（或电子版权认证）
- APP 备案号（中国大陆上架必须，工信部备案）
- 隐私政策链接（必须是可公网访问的页面）
- 应用图标（AGC 要求 216×216 PNG，以页面提示为准）
- 各机型尺寸的截图（首页、核心功能页）
- 应用简介与详细描述（说清解决什么问题，别写「最好用的 App」）
- 每个敏感权限的用途说明（对应 `module.json5` 里的每一条）
- 测试账号（若需登录）
- 年龄分级、分类、联系方式

**6. 提审前 Checklist。** 改 `versionCode`（每次 +1）和 `versionName`；删掉所有 `console.log` 里的敏感信息；确认 `buildMode` 是 release；确认 `module.json5` 里没有多余的权限（多申请一条就是多一条驳回理由）。

### 容易踩的坑

- **现象**：真机安装报权限相关错误（如 `grant request permissions failed`）。**原因**：Profile 里没包含该权限，尤其是 ACL 权限。**改法**：AGC 里编辑 Profile 加权限，重新下载 p7b 再打包。
- **现象**：安装提示「签名不一致」。**原因**：手机上已装过同 bundleName 但不同签名的包。**改法**：先卸载旧包再装。
- **现象**：`.app` 上传 AGC 报「包名不匹配」。**原因**：AGC 创建应用时的 bundleName 与 `app.json5` 不一致。**改法**：必须一致，且 AGC 侧创建后不可改，只能重建应用。
- **现象**：发布版白屏或功能缺失，调试版正常。**原因**：release 构建压缩/混淆后，某些依赖反射或动态属性的代码失效。**改法**：用 release 包在真机上做完整回归，别只测 debug。
- **现象**：新版本覆盖安装失败。**原因**：`versionCode` 没递增。**改法**：每次提审前 +1。

### 自检

- 打开 `build-profile.json5`，`material` 里 p12、cer、p7b、alias、两个密码六项齐全。
- `Build App(s)` 后，能在输出目录找到 `.app` 文件，文件大小合理（通常几 MB）。
- `hdc install -r` 能装到真机并正常启动，通知与提醒功能可用。
- AGC 控制台里应用版本号与 `app.json5` 的 `versionName` 一致。
- 把 `.p12` 移到别的目录再编译，应报签名失败 —— 确认它确实是必需品。

### 本讲产出

一个签名完整、可装真机、可上传 AGC 的 `.app`，以及一份上架材料清单。

### 进阶挑战

写一个 `release-checklist.md`，把上面六项发布前检查做成可勾选清单，然后**真的按它走一遍**：改版本号 → 清理日志 → 打 release 包 → 装真机 → 逐条回归（新增、打卡、跳转、持久化、每日一句、提醒）。把遇到的问题记在清单下面。

---
