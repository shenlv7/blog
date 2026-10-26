---
title: "第 3 讲：声明式 UI 与布局"
description: "用 Column / Row / Stack / Flex 把「习惯卡片」的版式搭出来，理解 vp 与 fp 的区别，并把重复结构抽成 @Builder。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 3
minutes: 5
---

### 本讲目标

用 Column / Row / Stack / Flex 把「习惯卡片」的版式搭出来，理解 vp 与 fp 的区别，并把重复结构抽成 `@Builder`。

做完你会看到：浅灰背景上有两张白底圆角的习惯卡片，左图标、中文字、右按钮。

### 为什么这样设计

声明式 UI 的本质是 `UI = f(state)`：你描述「长什么样」，框架决定「怎么画、什么时候重画」。所以布局代码里不该出现「先创建、再设置、再添加」这种命令式思路。

布局的规则是**约束向下传、尺寸向上报**。父容器给子组件一个最大尺寸，子组件算出自己想要多大再报回去。这就是为什么 `width('100%')` 能撑满、而什么都不写就只包裹内容。

单位必须分清：`vp` 是虚拟像素，随屏幕密度缩放，布局尺寸都用它；`fp` 是字体像素，额外跟随系统字体大小设置，`fontSize` 默认就是 fp。用 `px` 写死尺寸，在折叠屏和不同 DPI 的机器上一定翻车。`lpx` 已废弃，不要再用。

`@Builder` 是轻量复用：它不产生组件实例、没有自己的状态，只是一段可被调用的 UI 描述。适合「同一套版式换个数据」。有状态、有生命周期、要复用的才用 `@Component`。

### 跟着做

**1. 重写 `pages/Index.ets`。** 这一讲只做静态版式，不加任何状态：

```ts
@Entry
@Component
struct Index {
  // 组件内 @Builder：抽出的卡片版式，参数即“换个数据”
  @Builder
  HabitCard(name: string, streak: number, checked: boolean) {
    Row({ space: 12 }) {
      Text('🏃').fontSize(28)

      Column({ space: 4 }) {
        Text(name).fontSize(17).fontWeight(FontWeight.Medium)
        Text(`连续 ${streak} 天`).fontSize(13).fontColor('#8A8A8E')
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)          // 吃掉 Row 的剩余宽度，把按钮推到最右

      Button(checked ? '已打卡' : '打卡')
        .height(32)
        .fontSize(14)
        .backgroundColor(checked ? '#D1D1D6' : '#007DFF')
    }
    .width('100%')
    .padding(16)
    .backgroundColor(Color.White)
    .borderRadius(16)
  }

  build() {
    Column({ space: 12 }) {
      Text('我的习惯')
        .fontSize(30)
        .fontWeight(FontWeight.Bold)
        .width('100%')

      this.HabitCard('晨跑 3 公里', 12, false)
      this.HabitCard('读 20 页书', 3, true)
    }
    .width('100%')
    .height('100%')
    .padding(20)
    .backgroundColor('#F2F2F7')
    .alignItems(HorizontalAlign.Start)
  }
}
```

**2. 理解四种容器怎么选。** 记住这张表就够了：

| 容器 | 主轴方向 | 典型用途 |
| --- | --- | --- |
| `Column` | 纵向 | 页面从上到下的区块、卡片内文字堆叠 |
| `Row` | 横向 | 图标 + 文字 + 按钮的一行 |
| `Stack` | 叠放（Z 轴） | 背景图上盖文字、角标、遮罩 |
| `Flex` | 可换行/可配对齐 | 标签云、九宫格；一般布局优先用前三个 |

间距优先用容器的 `space`，而不是给每个子组件加 `margin` —— 后者在最后一个元素上会多出一段空白。

### 容易踩的坑

- **现象**：`Column` 里的 `Text` 不撑满宽度，`.width('100%')` 也不生效。**原因**：`Column` 默认 `alignItems(HorizontalAlign.Center)`，子组件按自身内容居中。**改法**：父容器加 `.alignItems(HorizontalAlign.Start)`，子组件加 `.width('100%')`。
- **现象**：加了 `.layoutWeight(1)` 元素反而消失了。**原因**：父容器主轴方向不对，比如在 `Column` 里想分水平空间。**改法**：`layoutWeight` 只在父容器的主轴方向分配剩余空间，横向要用 `Row`。
- **现象**：系统调大字体后文字被裁掉。**原因**：用 `px` 固定了行高，或手动把 fontSize 写成了 vp。**改法**：`fontSize` 用默认 fp，行高用 `constraintSize` 而不是死写 `height`。
- **现象**：圆角背景里子组件溢出圆角。**原因**：`borderRadius` 只裁自身背景，不裁子组件。**改法**：加 `.clip(true)`。

### 自检

- 打开模拟器，应按 20vp 内边距、12vp 卡片间距排列两张白卡。
- 把 `Column({ space: 12 })` 改成 `space: 0`，两张卡片应贴在一起 —— 说明 `space` 生效。
- 把 `HabitCard` 里的 `.layoutWeight(1)` 删掉，按钮应紧贴文字右侧而不是靠右。
- 把第二张卡片的 `checked` 改成 `false`，按钮应从灰色变蓝。

### 本讲产出

一个版式完整、纯静态的习惯列表页，以及一个可复用的卡片 `@Builder`。

### 进阶挑战

把 `HabitCard(name, streak, checked)` 改成 `HabitCard(habit: Habit)`，从第 2 讲的 `Habit` 接口取数据；再手写 5 条假数据数组，用 `ForEach` 渲染出来（提前预习第 5 讲，编译不过也没关系）。

---
