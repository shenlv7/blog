---
title: "权限：声明只是“报名”，user_grant 还要运行时申请"
description: "权限分两类：system_grant 安装时自动授予，如 ohos.permission.INTERNET；user_grant 必须运行时弹窗让用户同意，如相机、麦克风、位置。module.…"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 7
minutes: 5
---

权限分两类：`system_grant` 安装时自动授予，如 `ohos.permission.INTERNET`；`user_grant` 必须运行时弹窗让用户同意，如相机、麦克风、位置。`module.json5` 里的 `requestPermissions` 只告诉系统“我要用”，不会替用户点同意。

下面只是 `module` 里的 `requestPermissions` 片段，合并进你现有的 `module.json5`，不要把整个文件替换成这段：

```json5
{
  "module": {
    "requestPermissions": [
      {
        "name": "ohos.permission.CAMERA",
        "reason": "$string:camera_reason",
        "usedScene": {
          "abilities": ["EntryAbility"],
          "when": "inuse"
        }
      }
    ]
  }
}
```

`user_grant` 权限必须写 `reason` 和 `usedScene`，`reason` 要指向字符串资源，不能写裸字符串；`when` 常用 `inuse` 或 `always`。漏了这两个字段，安装或上架检查会出问题。上面的 `camera_reason` 要在 `entry/src/main/resources/base/element/string.json` 里补一条 `{ "name": "camera_reason", "value": "用于拍摄待办照片" }`。申请代码作为 `pages/PermissionDemo.ets` 运行，记得注册到 `main_pages.json`，并先声明上面的相机权限：

```typescript
import { abilityAccessCtrl, common, Permissions, PermissionRequestResult } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { promptAction } from '@kit.ArkUI';

@Entry
@Component
struct PermissionDemo {
  async requestCamera(): Promise<boolean> {
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    const permissions: Array<Permissions> = ['ohos.permission.CAMERA'];
    const atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
    try {
      const result: PermissionRequestResult =
        await atManager.requestPermissionsFromUser(context, permissions);
      const granted: boolean = result.authResults.length > 0 &&
        result.authResults[0] === abilityAccessCtrl.GrantStatus.PERMISSION_GRANTED;
      if (!granted) {
        promptAction.showToast({ message: '未授权相机，无法拍照' });
      }
      return granted;
    } catch (error) {
      const err: BusinessError = error as BusinessError;
      console.error(`申请权限失败：${err.code} ${err.message}`);
      return false;
    }
  }

  build() {
    Button('申请相机权限').onClick(() => {
      this.requestCamera();
    })
  }
}
```

用户第一次拒绝后，再次调用 `requestPermissionsFromUser` 通常不会弹窗。API 12 起可以用 `requestPermissionOnSetting(context, permissions)` 拉起权限设置页，但不要一上来就调，先走首次申请。权限只在真正用到时申请，不要在启动页一口气要相机、麦克风、位置，既容易被拒，也过不了隐私审查。

还有两类容易误解的权限：同一权限组里的权限可能被一起授权，申请前先看权限组说明；ACL（访问控制列表）权限不是写进 `module.json5` 就一定有，通常要在 AGC 申请并说明用途，普通应用声明了也可能被拒。别抄别人的权限清单，只声明你的功能真正需要的。
