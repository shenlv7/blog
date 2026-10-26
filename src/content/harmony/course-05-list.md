---
title: "第 5 讲：列表与数据模型"
description: "用 List + ForEach + 自定义行组件做出真正的习惯列表，支持新增和打卡（数据只在内存里）。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 5
minutes: 5
---

### 本讲目标

用 `List` + `ForEach` + 自定义行组件做出真正的习惯列表，支持新增和打卡（数据只在内存里）。

做完你会看到：能输入新习惯并添加，能点按钮打卡，连续天数实时变化；重启 App 数据清空。

### 为什么这样设计

用 `List` 而不是 `Scroll` + `Column`，因为 `List` 自带子组件复用和按需渲染；条目多了之后 `Scroll + Column` 会一次性把全部节点建出来，滚动直接卡死。数据量进一步变大时再换 `LazyForEach`，那是另一个量级的事，本项目用 `ForEach` 足够。

`ForEach` 的第三个参数 `keyGenerator` 必须返回**稳定且唯一**的值。框架靠 key 判断「哪个节点可以复用、哪个要重建」。用 index 当 key，删除中间项后所有后续节点的 key 都会前移，界面就会错位；用 `item.name` 当 key，改名字会导致整行重建、输入态丢失。用 `item.id` 是唯一正确解。

把行拆成 `@Component` 而不是 `@Builder`，是因为行里要接回调和 `@Prop` 数据，有自己的一小块状态边界；`@Builder` 没有这个能力。

### 跟着做

**1. 新建行组件 `entry/src/main/ets/view/HabitRow.ets`：**

```ts
import { Habit, calcStreak, todayKey } from '../model/Habit';

@Component
export struct HabitRow {
  @Prop habit: Habit;                                  // 父传子，单向快照
  onCheck: (id: string) => void = () => {};            // 普通成员，回调不需要响应式

  build() {
    Row({ space: 12 }) {
      Text(this.habit.icon).fontSize(26)

      Column({ space: 4 }) {
        Text(this.habit.name).fontSize(17).fontWeight(FontWeight.Medium)
        Text(`连续 ${calcStreak(this.habit.records)} 天`)
          .fontSize(13)
          .fontColor('#8A8A8E')
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)

      Button(this.habit.records.includes(todayKey()) ? '已打卡' : '打卡')
        .height(32)
        .fontSize(14)
        .enabled(!this.habit.records.includes(todayKey()))
        .onClick(() => { this.onCheck(this.habit.id); })
    }
    .width('100%')
    .padding(16)
    .backgroundColor(Color.White)
    .borderRadius(16)
  }
}
```

**2. 重写 `pages/Index.ets`：**

```ts
import { Habit, todayKey } from '../model/Habit';
import { HabitRow } from '../view/HabitRow';

@Entry
@Component
struct Index {
  @State habits: Habit[] = [
    { id: '1', name: '晨跑 3 公里', icon: '🏃', createdAt: 1717000000000, records: [] },
    { id: '2', name: '读 20 页书', icon: '📖', createdAt: 1717000000001, records: [] }
  ];
  @State draft: string = '';

  private addHabit(): void {
    const name: string = this.draft.trim();
    if (name.length === 0) {
      return;
    }
    const habit: Habit = {
      id: Date.now().toString(),
      name: name,
      icon: '✅',
      createdAt: Date.now(),
      records: []
    };
    this.habits = [habit].concat(this.habits);   // 整体替换，触发刷新
    this.draft = '';
  }

  private checkIn(id: string): void {
    const today: string = todayKey();
    this.habits = this.habits.map((h: Habit) => {
      if (h.id !== id || h.records.includes(today)) {
        return h;
      }
      const next: Habit = {
        id: h.id, name: h.name, icon: h.icon,
        createdAt: h.createdAt,
        records: h.records.concat(today)
      };
      return next;
    });
  }

  build() {
    Column({ space: 12 }) {
      Text('我的习惯').fontSize(30).fontWeight(FontWeight.Bold).width('100%')

      Row({ space: 8 }) {
        TextInput({ placeholder: '新习惯，比如：喝 8 杯水', text: this.draft })
          .layoutWeight(1)
          .height(40)
          .onChange((v: string) => { this.draft = v; })
        Button('添加')
          .height(40)
          .onClick(() => { this.addHabit(); })
      }
      .width('100%')

      List({ space: 12 }) {
        ForEach(this.habits, (item: Habit) => {
          ListItem() {
            HabitRow({ habit: item, onCheck: (id: string) => this.checkIn(id) })
          }
        }, (item: Habit) => item.id)          // key 必须唯一且稳定
      }
      .layoutWeight(1)
      .width('100%')
    }
    .width('100%')
    .height('100%')
    .padding(20)
    .backgroundColor('#F2F2F7')
    .alignItems(HorizontalAlign.Start)
  }
}
```

**3. 理解为什么用 `map` / `concat` 而不是 `push`。** `map` 返回一个新数组，引用变了，`@State` 一定能观察到；同时对被修改的那个 `Habit` 也创建了新对象，`@Prop` 会重新拷贝，行组件必然刷新。这条「不变式更新」的写法在 V2 里依然有效，是最不容易出错的路径。

### 容易踩的坑

- **现象**：新增后列表不刷新。**原因**：写成了 `this.habits[0].name = 'x'` 或 `this.habits.push()` 后没换引用。**改法**：用 `map` / `concat` 生成新数组再赋值。
- **现象**：删掉中间一项后，后面所有行的内容都错位了。**原因**：`keyGenerator` 用了 index。**改法**：用 `item.id`。
- **现象**：`List` 高度为 0，什么都看不见。**原因**：`List` 在 `Column` 里没有确定高度。**改法**：`Column` 加 `.height('100%')`，`List` 加 `.layoutWeight(1)`。
- **现象**：`TextInput` 输入一个字就丢焦点。**原因**：`text` 绑定的是 `@State`，每次 `onChange` 都重建了输入态。**改法**：保持受控写法不变，确认 `onChange` 里只改 `draft`，不要在 `build` 里做数据转换。

### 自检

- 打开 App，应看到预置的 2 条习惯。
- 输入「喝 8 杯水」点添加，应立刻出现在列表顶部，输入框清空。
- 输入空字符串点添加，列表不变。
- 点某行「打卡」，按钮变灰显示「已打卡」，连续天数变成 1。
- 再次点击已打卡的按钮（此时已 disabled），无任何变化。
- 杀掉进程重进，新增的习惯消失 —— 这是第 7 讲要解决的问题。

### 本讲产出

可新增、可打卡、带连续天数统计的习惯列表，数据暂存在内存。

### 进阶挑战

给 `HabitRow` 加左滑删除：`ListItem` 上加 `.swipeAction({ end: this.DeleteButton(item.id) })`，`DeleteButton` 用 `@Builder` 画一个红色「删除」。删除后列表立即更新，并确保 key 仍然唯一。

---
