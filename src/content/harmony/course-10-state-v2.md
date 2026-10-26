---
title: "第 10 讲：状态管理 V2"
description: "把第 5、6 讲的行组件和列表页迁到 @ComponentV2 / @Local / @Param / @Once / @Event，并判断什么情况下才值得迁。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 10
minutes: 5
---

### 本讲目标

把第 5、6 讲的行组件和列表页迁到 `@ComponentV2 / @Local / @Param / @Once / @Event`，并判断什么情况下才值得迁。

做完你会看到：V1 和 V2 两套实现并存，界面行为一致，但 V2 少了一层 `@ObjectLink`。

### 为什么这样设计

V1 的三个硬伤：观察不到嵌套（必须一层层 `@Observed` + `@ObjectLink`）、`@Prop` 深拷贝成本高、装饰器语义分散（`@State`/`@Link`/`@Provide` 各管一摊）。

V2 的解法是把「观察」和「传值」拆开：`@ObservedV2` + `@Trace` 只负责标记「这个字段变了要通知」，深度不限、层级不限，不需要在每一层都写 `@ObjectLink`；`@Local` / `@Param` / `@Event` 只负责「拥有 / 接收 / 回调」，职责单一。`@Once` 补上「只初始化一次、后续父组件更新不再同步」的场景，正好对应 V1 里 `@Prop` 被误用来做「初始化快照」的用法。

但**不要为了迁而迁**。判断标准：新页面新组件直接用 V2；老页面能稳定运行就不动；只有当出现「三层以上嵌套对象要同步」「`@ObjectLink` 写了五六个」「`@Prop` 拷贝大对象卡顿」时才迁。V1 与 V2 可以混用，但边界要放在纯展示子组件上。

### 跟着做

**1. 加 V2 数据模型。** 在 `model/Habit.ets` 末尾追加（V1 的 `Habit` 保留，两套并存便于对比）：

```ts
@ObservedV2
export class HabitV2 {
  id: string = '';
  name: string = '';
  icon: string = '';
  createdAt: number = 0;
  @Trace records: string[] = [];   // 只有加了 @Trace 的字段才可被观察

  constructor(id: string, name: string, icon: string) {
    this.id = id;
    this.name = name;
    this.icon = icon;
    this.createdAt = Date.now();
  }
}
```

**2. 新建 V2 版行组件 `view/HabitRowV2.ets`：**

```ts
import { HabitV2, calcStreak, todayKey } from '../model/Habit';

@ComponentV2
export struct HabitRowV2 {
  @Param habit: HabitV2 = new HabitV2('', '', '');     // 只读，本地初值兜底
  @Param dense: boolean = false;
  @Once initialName: string = '';                      // 只初始化一次，后续不同步
  @Event onCheck: (id: string) => void = () => {};

  build() {
    Row({ space: 12 }) {
      Text(this.habit.icon).fontSize(26)
      Column({ space: 4 }) {
        Text(this.habit.name).fontSize(17).fontWeight(FontWeight.Medium)
        Text(`连续 ${calcStreak(this.habit.records)} 天`)
          .fontSize(this.dense ? 11 : 13)
          .fontColor('#8A8A8E')
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)

      Button(this.habit.records.includes(todayKey()) ? '已打卡' : '打卡')
        .height(32)
        .fontSize(14)
        .onClick(() => { this.onCheck(this.habit.id); })
    }
    .width('100%')
    .padding(16)
    .backgroundColor(Color.White)
    .borderRadius(16)
  }
}
```

**3. 迁列表页。** 只需改这几行，其余结构不动：

```ts
import { HabitV2 } from '../model/Habit';
import { HabitRowV2 } from '../view/HabitRowV2';

@Entry
@ComponentV2
struct Index {
  @Local habits: HabitV2[] = [];       // 原 @State
  @Local draft: string = '';           // 原 @State

  private checkIn(id: string): void {
    const target: HabitV2 | undefined =
      this.habits.find((h: HabitV2) => h.id === id);
    if (target === undefined || target.records.includes(todayKey())) {
      return;
    }
    target.records.push(todayKey());    // V2 下可直接改，@Trace 负责通知
    this.persist();
  }

  // build 里 ForEach 换成：
  // ForEach(this.habits, (item: HabitV2) => {
  //   ListItem() {
  //     HabitRowV2({ habit: item, dense: false, onCheck: (id: string) => this.checkIn(id) })
  //   }
  // }, (item: HabitV2) => item.id)
}
```

**4. V1 → V2 迁移对照表。** 迁移时只改这几处，别的不用动：

| V1 | V2 | 说明 |
| --- | --- | --- |
| `@State` | `@Local` | 组件自己拥有的状态 |
| `@Prop` | `@Param` | 父传子只读；需要改就配 `@Event` 回调 |
| `@Link` | `@Param` + `@Event` | V2 没有双向绑定，用回调显式回写 |
| `@Observed` / `@ObjectLink` | `@ObservedV2` / `@Trace` | 删掉 `@ObjectLink`，深度观察由 `@Trace` 负责 |
| `@Watch` | `@Monitor` | 监听字段变化 |
| `@Provide` / `@Consume` | `@Provider` / `@Consumer` | 跨层传递 |
| `@StorageLink` | `AppStorageV2` | 应用级共享状态 |

**注意**：`@Entry` 与 `@ComponentV2` 的搭配能力随 SDK 版本有差异。如果你的本地 SDK 编译报错，就保留 `@Entry @Component` 的外壳页，只把子组件迁到 V2 —— V1 与 V2 可以混用，这是官方支持的用法。

### 容易踩的坑

- **现象**：`@Local` 改了但界面不刷新。**原因**：改的是 `@ObservedV2` 类里没加 `@Trace` 的字段。**改法**：给该字段加 `@Trace`。
- **现象**：`@Param` 编译报 `must be initialized`。**原因**：既没有本地初值，父组件也没传。**改法**：给默认值，或用 `@Require` 强制父组件传。
- **现象**：V1 页面里嵌 V2 组件后状态不同步。**原因**：V1 的 `@State` 观察不到 `@ObservedV2` 的 `@Trace`，反之亦然。**改法**：边界放在纯展示子组件上，不要跨代共享可变状态。
- **现象**：迁完 `@Link` 的地方编译不过。**原因**：V2 没有 `@Link`。**改法**：`@Param` 读 + `@Event` 回调写。
- **现象**：`@Once` 用成了 `@Param`。**原因**：`@Once` 只在首次传值时同步，后续父组件更新不会跟进。**改法**：需要持续同步就用 `@Param`，`@Once` 只用于「初始快照」。

### 自检

- 列表页、详情页功能与第 9 讲完全一致（新增、打卡、跳转、返回、持久化）。
- 全工程搜索 `@ObjectLink`，应为 0 处。
- 给 `records` 加 `@Monitor`，点打卡后控制台应打印出变化前后的值。
- 把 `@Trace records` 的 `@Trace` 删掉，打卡后连续天数不再刷新 —— 亲眼看到 V2 的观察边界。
- 编译窗口无 ArkTS 报错，无「装饰器不匹配」告警。

### 本讲产出

一套 V2 版的模型、行组件、列表页，与 V1 版本并存可对比。

### 进阶挑战

用 `PersistenceV2` 把 `habits` 做成自动持久化，删掉第 7 讲里手写的 `persist()` 调用；再用 `@Monitor` 在连续天数变化时打一行日志。完成后对比：代码少了多少行，可控性有没有下降。

---
