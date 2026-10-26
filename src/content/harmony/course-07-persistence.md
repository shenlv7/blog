---
title: "第 7 讲：接入持久化"
description: "用 Preferences 把习惯数据存到本地，讲清什么时候落盘、为什么刷新后还在。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 7
minutes: 5
---

### 本讲目标

用 `Preferences` 把习惯数据存到本地，讲清什么时候落盘、为什么刷新后还在。

做完你会看到：添加习惯、打卡后杀掉进程重进，数据完整保留。

### 为什么这样设计

内存数据在进程被杀时消失，磁盘不会 —— 这就是「刷新后还在」的全部原因。持久化不是魔法，只是把内存里的 `Habit[]` 序列化成字符串写进文件，启动时再读回来。

选 `Preferences` 而不是别家：它是 key-value 存储，适合配置和小对象（官方建议单条不超过 8KB、总量有上限，具体以《@ohos.data.preferences》文档为准）。习惯打卡数据量小、结构简单，`Preferences` 一步到位。如果以后要做「按天统计、按月查询」，再换 `relationalStore`（SQLite）—— 那时才是关系型的场景。

落盘时机是关键区分：`put()` 只是写内存，`flush()` 才落盘。只 `put` 不 `flush`，杀进程就丢；每次改动都 `flush`，又会有写盘开销。本项目的取舍是**每次用户操作后立刻 `flush`** —— 操作频率低（一天几次到几十次），数据量小，可靠性优先于那点性能。

不用 `PersistentStorage` / `@StorageLink` 自动持久化，是因为它只适合简单类型，且观察粒度粗、调试困难；显式读写虽然多几行，但数据流向一目了然。

### 跟着做

**1. 新建 `entry/src/main/ets/store/HabitStore.ets`：**

```ts
import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';
import { Habit } from '../model/Habit';

const STORE_NAME: string = 'habit_store';
const KEY_HABITS: string = 'habits';
const KEY_VERSION: string = 'schema_version';

export class HabitStore {
  private static instance: HabitStore | undefined = undefined;
  private store: preferences.Preferences | undefined = undefined;

  static getInstance(): HabitStore {
    if (HabitStore.instance === undefined) {
      HabitStore.instance = new HabitStore();
    }
    return HabitStore.instance;
  }

  async init(context: common.UIAbilityContext): Promise<void> {
    this.store = await preferences.getPreferences(context, STORE_NAME);
  }

  async load(): Promise<Habit[]> {
    if (this.store === undefined) {
      return [];
    }
    const raw: preferences.ValueType = await this.store.get(KEY_HABITS, '[]');
    const text: string = raw as string;
    if (text.length === 0) {
      return [];
    }
    try {
      return JSON.parse(text) as Habit[];
    } catch (e) {
      console.error('习惯数据解析失败，按空数据处理');
      return [];
    }
  }

  async save(habits: Habit[]): Promise<void> {
    if (this.store === undefined) {
      return;
    }
    await this.store.put(KEY_HABITS, JSON.stringify(habits));
    await this.store.put(KEY_VERSION, 1);
    await this.store.flush();          // 关键：不 flush 只在内存里
  }
}
```

**2. 在 `Index.ets` 里接上读写。** 加三个方法，`addHabit` / `checkIn` / `openDetail` 末尾都调 `persist()`：

```ts
import { BusinessError } from '@kit.BasicServicesKit';
import { HabitStore } from '../store/HabitStore';

aboutToAppear(): void {
  this.bootstrap();      // 不阻塞首帧渲染
}

private async bootstrap(): Promise<void> {
  const ctx = getContext(this) as common.UIAbilityContext;
  const store = HabitStore.getInstance();
  await store.init(ctx);
  const saved: Habit[] = await store.load();
  if (saved.length > 0) {
    this.habits = saved;
  }
}

private persist(): void {
  HabitStore.getInstance().save(this.habits).catch((err: BusinessError) => {
    console.error(`保存失败: ${err.code} ${err.message}`);
  });
}
```

`aboutToAppear` 里不直接 `await`，是因为页面渲染不该等 IO；先渲染预置数据，读到再覆盖，用户几乎看不到闪烁。

**3. 想清楚迁移。** `KEY_VERSION` 现在没用上，但它是你未来改结构时的唯一退路。下次给 `Habit` 加字段时，读出来先判断版本号，做一次字段补全再写回，老用户数据才不会炸。

### 容易踩的坑

- **现象**：杀进程重进数据没了。**原因**：只 `put` 没 `flush`，或者 `save` 根本没被调用。**改法**：每个写操作后都 `persist()`，并在 `save` 里确认 `await this.store.flush()`。
- **现象**：`getPreferences` 报 context 无效。**原因**：`getContext()` 拿到的不是 UIAbilityContext。**改法**：显式 `as common.UIAbilityContext`；API 12 起也可以试 `this.getUIContext().getHostContext()`。
- **现象**：第一次启动读到空串，`JSON.parse('')` 抛异常。**原因**：默认值传了 `''`。**改法**：默认值传 `'[]'`，并保留 `try/catch` 兜底。
- **现象**：`JSON.parse` 后的对象调用方法报 `is not a function`。**原因**：JSON 只还原数据不还原原型（第 2 讲提过）。**改法**：模型保持纯数据，逻辑放函数。
- **现象**：数据变大了以后启动卡顿。**原因**：一次性读全量并同步解析。**改法**：分页存储，或换 `relationalStore`。

### 自检

- 添加 2 个习惯，杀掉进程（模拟器多任务上滑），重进，数据还在。
- 打卡一次后杀进程重进，按钮仍是「已打卡」。
- 用 DevEco 的 Device File Explorer 打开 `/data/app/el2/100/base/com.dahua.habit/haps/entry/preferences/`，应能看到 `habit_store` 文件。
- 在系统设置里清空 App 数据后重进，应回到第 5 讲的预置 2 条。
- 故意把 `flush()` 注释掉再测一次，数据丢失 —— 确认你理解了落盘时机。

### 本讲产出

数据可持久化的完整习惯列表，重启不丢。

### 进阶挑战

加一个「导出 JSON」按钮，把 `JSON.stringify(this.habits, null, 2)` 写进剪贴板（`@kit.BasicServicesKit` 的 `pasteboard`），再找一个文本框粘贴出来看。目的是让你亲眼看到磁盘里到底存了什么结构。

---
