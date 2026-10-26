---
title: "第 6 讲：生命周期与页面跳转"
description: "用 Navigation + NavPathStack 做列表到详情页的跳转，掌握参数怎么传、返回值怎么带回来。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 6
minutes: 5
---

### 本讲目标

用 `Navigation` + `NavPathStack` 做列表到详情页的跳转，掌握参数怎么传、返回值怎么带回来。

做完你会看到：点任意一行进入详情页，在详情页打卡或改名，返回后列表同步更新。

### 为什么这样设计

`router` 模块从 API 9 起不再演进，官方推荐 `Navigation`。原因在于 `router` 是「页面栈 + URL 字符串」，参数只能序列化塞进 URL，类型全丢；`Navigation` 是「导航容器 + `NavPathStack` 对象栈」，参数是内存里的对象引用，返回时还能带任意结果，并且天然支持一多设备的分栏布局。

`NavPathStack` 的本质是一个栈：`pushPathByName` 入栈，`pop` 出栈。它不是路由表 —— 你不需要为每个页面注册一条路由，而是在 `navDestination` 的 `@Builder` 里按名字决定渲染哪个组件。这个设计让「页面」变成普通组件，可以带状态、可以嵌套。

参数传递有三条路，按场景选：**构造参数**适合父子直接传；**栈参数**（`pushPathByName` 第二个参数）适合跨页面；**AppStorage** 适合全局单例。本项目用第二条，因为它和返回栈绑定，页面销毁时参数自动释放。

列表页和详情页的数据同步，最稳的做法不是共享引用，而是**详情页返回时把整条数据 pop 回来，列表页整体替换**。这样每次同步都是一次明确的数据流，不会出现「两个页面同时改一个对象」的竞态。

### 跟着做

**1. 新建详情页 `entry/src/main/ets/pages/HabitDetail.ets`：**

```ts
import { Habit, calcStreak, todayKey } from '../model/Habit';

@Component
export struct HabitDetail {
  @State habit: Habit = { id: '', name: '', icon: '', createdAt: 0, records: [] };
  private stack: NavPathStack = new NavPathStack();

  build() {
    NavDestination() {
      Column({ space: 16 }) {
        Text(this.habit.icon).fontSize(64)
        Text(this.habit.name).fontSize(24).fontWeight(FontWeight.Bold)
        Text(`连续 ${calcStreak(this.habit.records)} 天 · 累计 ${this.habit.records.length} 次`)
          .fontSize(15)
          .fontColor('#8A8A8E')

        Button('打卡')
          .width('80%')
          .enabled(!this.habit.records.includes(todayKey()))
          .onClick(() => {
            const today: string = todayKey();
            if (this.habit.records.includes(today)) {
              return;
            }
            this.habit = {
              id: this.habit.id, name: this.habit.name, icon: this.habit.icon,
              createdAt: this.habit.createdAt,
              records: this.habit.records.concat(today)
            };
          })

        Button('保存并返回')
          .width('80%')
          .backgroundColor('#34C759')
          .onClick(() => { this.stack.pop(this.habit); })   // 出栈并带回结果
      }
      .width('100%')
      .height('100%')
      .justifyContent(FlexAlign.Center)
    }
    .title(this.habit.name)
    .onReady((ctx: NavDestinationContext) => {
      this.stack = ctx.pathStack;                 // 拿到当前页所在的栈
      const param: Object | undefined = ctx.pathInfo.param;
      if (param !== undefined) {
        this.habit = param as Habit;              // 栈参数在 pathInfo 里
      }
    })
  }
}
```

**2. 改造 `pages/Index.ets`。** 把整页包进 `Navigation`，并挂上 `navDestination`：

```ts
import { Habit, todayKey } from '../model/Habit';
import { HabitRow } from '../view/HabitRow';
import { HabitDetail } from './HabitDetail';

@Entry
@Component
struct Index {
  @State habits: Habit[] = [ /* 同第 5 讲 */ ];
  @State draft: string = '';
  private pageStack: NavPathStack = new NavPathStack();

  // 导航目标：名字 -> 组件
  @Builder
  PageMap(name: string) {
    if (name === 'HabitDetail') {
      HabitDetail()
    }
  }

  private openDetail(habit: Habit): void {
    this.pageStack.pushPathByName('HabitDetail', habit, (info: PopInfo) => {
      const updated: Habit = info.result as Habit;
      if (updated === undefined || updated === null) {
        return;                                   // 手势返回，无结果
      }
      this.habits = this.habits.map((h: Habit) => h.id === updated.id ? updated : h);
    });
  }

  // addHabit / checkIn 同第 5 讲，省略

  build() {
    Navigation(this.pageStack) {
      Column({ space: 12 }) {
        // 标题、输入行、List 同第 5 讲，区别是 ListItem 上加 onClick
        List({ space: 12 }) {
          ForEach(this.habits, (item: Habit) => {
            ListItem() {
              HabitRow({ habit: item, onCheck: (id: string) => this.checkIn(id) })
            }
            .onClick(() => { this.openDetail(item); })
          }, (item: Habit) => item.id)
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
    .navDestination(this.PageMap)      // 必须挂在 Navigation 上
    .title('我的习惯')
    .titleMode(NavigationTitleMode.Mini)
    .mode(NavigationMode.Stack)
  }
}
```

注意：`HabitDetail` 不是独立路由，**不要**把它加进 `main_pages.json`，否则会出现两套导航系统打架。

**3. 讲清生命周期。** `EntryAbility` 的 `onCreate/onWindowStageCreate/onForeground/onBackground/onDestroy` 管的是应用级；页面的 `aboutToAppear`（创建时）、`aboutToDisappear`（销毁时）、`onPageShow` / `onPageHide`（前后台切换）管的是页面级。第 7 讲的读取数据放 `aboutToAppear`，第 8 讲的拉取放 `onPageShow` 更合适 —— 从后台回前台会重新触发。

### 容易踩的坑

- **现象**：点列表没反应。**原因**：`navDestination` 没挂，或 `Navigation` 内放了多个根子组件。**改法**：`Navigation` 内只能有一个根容器；`.navDestination(this.PageMap)` 写在 `Navigation` 的链式调用上。
- **现象**：详情页白屏。**原因**：`PageMap` 里的 `name` 和 `pushPathByName` 的 `name` 大小写不一致。**改法**：抽成常量 `const ROUTE_DETAIL: string = 'HabitDetail'`。
- **现象**：`ctx.pathInfo.param` 是 undefined。**原因**：`pushPathByName` 只传了名字没传参数。**改法**：第二个参数传 `habit`。
- **现象**：返回后列表没更新。**原因**：`pop()` 没带结果，或 `pushPathByName` 没接第三个参数。**改法**：`pop(this.habit)` + 回调里接 `info.result`。
- **现象**：详情页打卡了但界面不动。**原因**：写成了 `this.habit.records.push(today)`，观察不到嵌套层。**改法**：整体替换 `this.habit`（上面的写法）。

### 自检

- 点任意一行，应进入详情页，顶部标题是习惯名。
- 详情页点打卡，连续天数 +1，按钮变灰。
- 点「保存并返回」，列表页对应行同步显示新的连续天数。
- 用系统返回手势退出详情页，不应崩溃，且不修改原数据。
- 打开 `main_pages.json`，应仍然只有 `"pages/Index"` 一项。

### 本讲产出

可跳转、可传参、可回传的详情页，以及一套基于 `NavPathStack` 的导航骨架。

### 进阶挑战

在详情页加「删除这个习惯」按钮，`pop` 时返回 `{ deletedId: string }`；列表页收到后从数组中移除该项。注意定义返回结果的类型并区分「改名返回」和「删除返回」两种情况。

---
