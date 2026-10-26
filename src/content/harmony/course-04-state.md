---
title: "第 4 讲：状态驱动界面"
description: "用一个计数器把 @State → @Prop → @Link → @Observed/@ObjectLink 四种观察方式一次讲透，并建立「改了不刷新」的排查直觉。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 4
minutes: 5
---

### 本讲目标

用一个计数器把 `@State → @Prop → @Link → @Observed/@ObjectLink` 四种观察方式一次讲透，并建立「改了不刷新」的排查直觉。

做完你会看到：同一页上三个子组件，点父组件按钮时全部同步；点子组件按钮时只有对应那个按预期变化。

### 为什么这样设计

「改了不刷新」是鸿蒙新手第一大坑，原因永远落在这三类里：

1. **变量没被装饰器修饰** —— 普通成员变量不参与观察，改了框架根本不知道。
2. **装饰器选错** —— `@Prop` 是单向深拷贝，子改父不会变；`@Link` 是双向引用，父必须用 `$` 传。
3. **观察层级不够** —— `@State` 只观察第一层：给对象整体赋值能看到，改对象里的字段看不到。要观察嵌套，必须 `@Observed` + `@ObjectLink`。

`@Prop` 之所以是深拷贝，是为了让子组件「拿到一份快照」，父组件后续怎么变都不影响它已经渲染的内容 —— 适合纯展示。代价是拷贝开销，对象大时别用。

`@Link` 是引用，父子共享同一份数据，改哪边都同步 —— 代价是数据流向不清晰，层级一深就难追踪。所以规则是：**能单向就单向，确实要回写再用双向**。

### 跟着做

**1. 新建 `pages/StateDemo.ets`。** 这是练习页，第 5 讲会删掉：

```ts
// @Observed 让类的实例可被 @ObjectLink 观察
@Observed
class Counter {
  value: number = 0;
}

@Entry
@Component
struct StateDemo {
  @State count: number = 0;                    // 自己拥有，自己改
  @State counter: Counter = new Counter();     // 嵌套对象的入口

  build() {
    Column({ space: 16 }) {
      Text(`父组件 count = ${this.count}`).fontSize(20)
      Button('父 +1').onClick(() => { this.count++; })

      Divider()

      // @Prop：父传子，单向
      ChildProp({ value: this.count })

      // @Link：父子共享，注意 $ 前缀
      ChildLink({ value: $count })

      Divider()

      Text(`嵌套对象 value = ${this.counter.value}`).fontSize(18)
      Button('改嵌套属性').onClick(() => { this.counter.value++; })
      ChildObjectLink({ counter: this.counter })
    }
    .width('100%')
    .padding(20)
    .alignItems(HorizontalAlign.Start)
  }
}

@Component
struct ChildProp {
  @Prop value: number = 0;   // 允许本地初值，父组件传值会覆盖

  build() {
    Row({ space: 8 }) {
      Text(`@Prop 收到 ${this.value}`)
      Button('子 +1（不回传）').onClick(() => { this.value++; })
    }
  }
}

@Component
struct ChildLink {
  @Link value: number;       // 不能本地初始化，必须由父组件用 $ 传

  build() {
    Row({ space: 8 }) {
      Text(`@Link 收到 ${this.value}`)
      Button('子 +1（双向）').onClick(() => { this.value++; })
    }
  }
}

@Component
struct ChildObjectLink {
  @ObjectLink counter: Counter;   // 不能本地初始化，必须接 @Observed 的实例

  build() {
    Text(`@ObjectLink 看到 ${this.counter.value}`)
  }
}
```

**2. 让它能跑起来。** 打开 `entry/src/main/resources/base/profile/main_pages.json`，加一行 `"pages/StateDemo"`；再把 `EntryAbility.ets` 里 `windowStage.loadContent('pages/Index')` 临时改成 `'pages/StateDemo'`。练完记得改回来 —— 从第 5 讲起一切在 `Index` 上迭代。

**3. 验证四条规则。** 依次点：父 +1 三次 → `@Prop`、`@Link` 两个子组件都应显示 3；点 `@Prop` 子组件的按钮 → 只有它自己变；点 `@Link` 子组件的按钮 → 父组件跟着变；点「改嵌套属性」→ `@ObjectLink` 子组件实时变。

### 容易踩的坑

- **现象**：点按钮数字在日志里变了，界面纹丝不动。**原因**：变量没加装饰器，或是 `let count` 这种局部变量。**改法**：改成 `@State count`，并确保用 `this.count` 修改。
- **现象**：子组件改了 `@Prop`，父组件没反应。**原因**：`@Prop` 是单向深拷贝，设计如此。**改法**：要回写就换 `@Link`，或保留 `@Prop` + 一个 `onChange` 回调向上通知。
- **现象**：编译报 `The '@Link' property 'value' cannot be initialized locally`。**原因**：给 `@Link` 写了初值。**改法**：删掉初值，父组件传 `$count`。
- **现象**：数组 `push` 后 `ForEach` 不刷新。**原因**：`@State` 能观察到数组本身的增删（API 11 起），但观察不到**元素内部属性**的变化。**改法**：元素类型加 `@Observed`，行组件用 `@ObjectLink`；或干脆整体替换数组。
- **现象**：`@ObjectLink` 编译报类型不匹配。**原因**：父组件传的对象不是 `@Observed` 类的实例。**改法**：给类加 `@Observed`，或用普通 `@Prop`。

### 自检

- 点「父 +1」三次，`@Prop` 与 `@Link` 两个子组件都显示 3。
- 点 `@Prop` 子组件的按钮，父组件仍显示 3，只有子组件变成 4。
- 点 `@Link` 子组件的按钮，父组件同步变成 4。
- 点「改嵌套属性」，`@ObjectLink` 子组件的数字实时 +1。
- 把 `Counter` 上的 `@Observed` 删掉，最后一项不再刷新 —— 这就是观察边界。

### 本讲产出

一个可运行的五种状态装饰器对照页 `pages/StateDemo.ets`，作为后续排查问题的参照物。

### 进阶挑战

把 `Counter` 换成 `Habit`，实现「点打卡 → `records` 加一条 → 连续天数跟着变」。写两遍：第一遍用「整体替换对象」，第二遍用「`@Observed` + `@ObjectLink`」。比较两者的代码量和刷新范围，记下你更愿意维护哪一种。

---
