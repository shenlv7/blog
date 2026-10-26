---
title: "第 9 讲：通知与后台提醒"
description: "申请通知开关、发一条本地通知、用提醒代理设置每天 21:00 的打卡提醒，并讲清 user_grant 权限为什么必须运行时申请。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 9
minutes: 5
---

### 本讲目标

申请通知开关、发一条本地通知、用提醒代理设置每天 21:00 的打卡提醒，并讲清 `user_grant` 权限为什么必须运行时申请。

做完你会看到：下拉通知栏出现「习惯打卡」通知；系统里能看到一条每日提醒任务。

### 为什么这样设计

先纠正一个常见误解：**发本地通知不需要在 `module.json5` 里声明任何权限**，需要的是用户把「允许通知」这个开关打开。开关是用户资产，不是安装期授予的权限，所以只能用 `requestEnableNotification` 引导，不能假设它有。

「明天 21:00 提醒我」则是另一回事。你的进程活不到明天 —— 应用退到后台会被挂起甚至回收。所以定时提醒必须交给系统代理：`reminderAgentManager` 把提醒注册进系统，到点由系统拉起通知。它需要声明 `ohos.permission.PUBLISH_AGENT_REMINDER`，这个权限是 `system_grant`（安装即授予），你在 `module.json5` 里写了就生效，不需要弹窗。

那 `user_grant` 是什么？是**用户可随时撤销**的敏感权限，比如通讯录、位置、相机、麦克风。它的特点是：安装时给一次，用户之后能在设置里关掉，所以每次使用前都得检查，用 `abilityAccessCtrl.createAtManager().requestPermissionsFromUser()` 运行时申请。本项目的打卡功能用不到，但你做「定位打卡」「拍照打卡」时一定会遇到 —— 所以本讲把这个模板一次写对，后面直接抄。

### 跟着做

**1. 补两行权限。** `entry/src/main/module.json5` 的 `requestPermissions`：

```json5
"requestPermissions": [
  { "name": "ohos.permission.INTERNET" },
  { "name": "ohos.permission.PUBLISH_AGENT_REMINDER" }
]
```

**2. 新建 `entry/src/main/ets/service/ReminderService.ets`：**

```ts
import { notificationManager } from '@kit.NotificationKit';
import { reminderAgentManager } from '@kit.BackgroundTasksKit';
import { abilityAccessCtrl, common, Permissions } from '@kit.AbilityKit';
import { BusinessError } from '@kit.BasicServicesKit';

// 1) 确保通知开关已打开
export async function ensureNotificationEnabled(
  context: common.UIAbilityContext
): Promise<boolean> {
  if (await notificationManager.isNotificationEnabled()) {
    return true;
  }
  try {
    await notificationManager.requestEnableNotification(context);
    return await notificationManager.isNotificationEnabled();
  } catch (e) {
    const err = e as BusinessError;
    console.error(`用户未开启通知: ${err.code}`);
    return false;
  }
}

// 2) 立刻发一条，用来验证链路通不通
export async function publishNow(title: string, text: string): Promise<void> {
  const request: notificationManager.NotificationRequest = {
    id: 1001,
    content: {
      notificationContentType: notificationManager.ContentType.NOTIFICATION_CONTENT_BASIC_TEXT,
      normal: { title: title, text: text }
    }
  };
  await notificationManager.publish(request);
}

// 3) 每天固定时间提醒，交给系统代理
export async function scheduleDailyReminder(hour: number, minute: number): Promise<number> {
  const request: reminderAgentManager.ReminderRequestAlarm = {
    reminderType: reminderAgentManager.ReminderType.REMINDER_TYPE_ALARM,
    hour: hour,
    minute: minute,
    daysOfWeek: [1, 2, 3, 4, 5, 6, 7],   // 1=周一 … 7=周日，不是 0 基
    title: '习惯打卡',
    content: '今天还有习惯没打卡，花 30 秒搞定它',
    wantAgent: { pkgName: 'com.dahua.habit', abilityName: 'EntryAbility' },
    ringDuration: 3,
    snoozeTimes: 0,
    notificationId: 1001
  };
  return await reminderAgentManager.publishReminder(request);   // 返回 reminderId
}

export async function cancelReminder(reminderId: number): Promise<void> {
  await reminderAgentManager.cancelReminder(reminderId);
}

// 4) user_grant 权限的通用模板：本讲用不到，做定位/相机/通讯录时直接抄
export async function requestUserGrant(
  context: common.UIAbilityContext,
  permissions: Array<Permissions>
): Promise<boolean> {
  const atManager: abilityAccessCtrl.AtManager = abilityAccessCtrl.createAtManager();
  const result = await atManager.requestPermissionsFromUser(context, permissions);
  // authResults 里 0 表示已授权，-1 表示拒绝
  return result.authResults.every((code: number) => code === 0);
}
```

**3. 在页面上加一个开关。** 用一个 `Toggle` 控制，`reminderId` 存进 `Preferences` 才能真正「取消」：

```ts
@State reminderOn: boolean = false;
private reminderId: number = -1;

private async toggleReminder(on: boolean): Promise<void> {
  const ctx = getContext(this) as common.UIAbilityContext;
  if (on) {
    const ok: boolean = await ensureNotificationEnabled(ctx);
    if (!ok) {
      this.reminderOn = false;
      return;
    }
    this.reminderId = await scheduleDailyReminder(21, 0);
    this.reminderOn = true;
  } else {
    if (this.reminderId >= 0) {
      await cancelReminder(this.reminderId);
      this.reminderId = -1;
    }
    this.reminderOn = false;
  }
}

// build() 里加一行：
Row() {
  Text('每天 21:00 提醒我').layoutWeight(1)
  Toggle({ type: ToggleType.Switch, isOn: this.reminderOn })
    .onChange((isOn: boolean) => { this.toggleReminder(isOn); })
}
.width('100%')
.padding(16)
.backgroundColor(Color.White)
.borderRadius(14)
```

`reminderId` 要持久化（存进第 7 讲的 `HabitStore`），否则下次启动取消不掉旧提醒，用户会被重复打扰。

### 容易踩的坑

- **现象**：`publish` 成功但通知栏什么都没有。**原因**：系统通知开关关着；或模拟器不支持。**改法**：先 `isNotificationEnabled` 判断并引导，最终用真机验收。
- **现象**：`publishReminder` 报权限错误。**原因**：`module.json5` 漏了 `ohos.permission.PUBLISH_AGENT_REMINDER`。**改法**：补上并重新安装。
- **现象**：到点不响。**原因**：`daysOfWeek` 写成 0 基（0 表示周日），或手机处于超级省电模式。**改法**：用 1–7 表示周一到周日；测试时把 `hour/minute` 设成 2 分钟后。
- **现象**：关掉开关后到点还响。**原因**：`publishReminder` 每次都会生成新的 reminderId，旧的没取消。**改法**：把 id 存下来，先 `cancelReminder` 再重新 `publish`。
- **现象**：`requestPermissionsFromUser` 不弹窗。**原因**：传的 context 不是 `UIAbilityContext`，或权限没在 `module.json5` 声明。**改法**：两处都要对；且 `user_grant` 权限必须在 `requestPermissions` 里声明过，运行时申请才有效。

### 自检

- 打开 App 点「发一条测试通知」，下拉通知栏应看到「习惯打卡」。
- 在系统设置里关掉本应用通知开关，再点发布，应弹出授权引导而不是静默失败。
- 打开开关设置 21:00 提醒，在系统「设置 > 应用 > 应用管理 > 习惯打卡 > 通知」或提醒代理列表里应能看到该任务。
- 把时间改成 2 分钟后，退到桌面等待，应收到通知。
- 关掉开关后重复上一步，到点不应再响。

### 本讲产出

`ReminderService.ets`：通知开关引导、本地通知、每日定时提醒、以及一份 `user_grant` 权限模板。

### 进阶挑战

用 `TimePickerDialog` 让用户自己选提醒时间，选完自动取消旧提醒、注册新提醒，并把 `reminderId` 和 `hour/minute` 一起存进 `Preferences`。重启 App 后开关状态应保持正确。

---
