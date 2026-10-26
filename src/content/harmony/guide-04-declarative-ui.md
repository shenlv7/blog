---
title: "ArkUI 声明式 UI：装饰器各自解决什么问题"
description: "ArkUI 的心智模型：build() 只描述“当前状态下的 UI 长什么样”，状态变量变了，框架重新执行相关部分。不要命令式地找控件、改属性。自定义组件用 struct，@Entry 标记页…"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 4
minutes: 5
---

ArkUI 的心智模型：`build()` 只描述“当前状态下的 UI 长什么样”，状态变量变了，框架重新执行相关部分。不要命令式地找控件、改属性。自定义组件用 `struct`，`@Entry` 标记页面入口，`@Component` 标记普通组件。

| 装饰器 | 作用 | 数据方向 | 典型场景 |
| --- | --- | --- | --- |
| `@Entry` | 页面入口 | 无 | 每个页面文件一个，配合 `main_pages.json` |
| `@Component` | 自定义组件 | 无 | 可复用的 UI 单元 |
| `@State` | 组件内部状态 | 本地 | 计数、输入框、加载中 |
| `@Prop` | 父传子 | 单向 | 只读展示，子改不影响父 |
| `@Link` | 父子双向 | 双向 | 子组件修改父状态 |
| `@Provide/@Consume` | 跨层共享 | 双向 | 主题、当前用户、跨多层传值 |
| `@Watch` | 监听状态变化 | 无 | 状态变化后做校验、请求、写日志 |
| `@Builder` | 可复用 UI 片段 | 参数值传递 | 列表项、标题、空状态 |

下面这个文件可以直接作为 `pages/StateDemo.ets` 运行，记得在 `main_pages.json` 注册：

```typescript
@Entry
@Component
struct StateDemo {
  @State count: number = 0;
  @State @Watch('onWatchedChange') watched: number = 0;
  @Provide('themeColor') themeColor: string = '#0A59F7';

  onWatchedChange(propName: string): void {
    console.info(`${propName} 变成了 ${this.watched}`);
  }

  @Builder
  tag(text: string) {
    Text(text)
      .fontSize(12)
      .padding({ left: 8, right: 8, top: 4, bottom: 4 })
      .borderRadius(8)
      .backgroundColor('#EEF3FF')
  }

  build() {
    Column({ space: 12 }) {
      Text(`父组件 count = ${this.count}`)
      Button('父组件 +1').onClick(() => {
        this.count++;
      })
      Button('触发 @Watch').onClick(() => {
        this.watched++;
      })
      this.tag('@Builder 生成的标签')
      PropChild({ value: this.count })
      LinkChild({ count: $count })
      ProvideChild()
    }
    .padding(16)
    .width('100%')
  }
}

@Component
struct PropChild {
  @Prop value: number = 0;

  build() {
    Text(`@Prop 单向：${this.value}`)
  }
}

@Component
struct LinkChild {
  @Link count: number;

  build() {
    Button(`@Link 双向：${this.count}`).onClick(() => {
      this.count++;
    })
  }
}

@Component
struct ProvideChild {
  @Consume('themeColor') themeColor: string;

  build() {
    Button('@Consume 改主题色')
      .backgroundColor(this.themeColor)
      .onClick(() => {
        this.themeColor = this.themeColor === '#0A59F7' ? '#E84026' : '#0A59F7';
      })
  }
}
```

几个容易错的点：`@State` 必须本地初始化；`@Link` 不能写 `= 0`，父组件用 `$count` 传引用；`@Prop` 子组件改了只改本地副本，不会回传父组件；`@Consume` 往上找不到同 key 的 `@Provide` 会运行时报错；`@Watch` 只在初始化之后变化时触发，第一次渲染不触发；`@Builder` 的参数是值传递，不要把“子 Builder 改参数能回写父状态”当机制，状态要放在组件变量里。

V1 的 `@State` 对对象只稳定观察第一层，嵌套对象的字段变化经常不刷新；这时用 `@Observed` 类配合 `@ObjectLink`，或者直接换 V2 的 `@ObservedV2` + `@Trace`。数组整体赋值、`push`、`splice` 这类操作通常能触发刷新，但“只改数组里某个对象的深层字段”仍可能漏掉。布局上优先 `Column`/`Row`/`Stack`/`List`，列表用 `ForEach` 并给稳定 key，不要用绝对定位硬堆页面。
