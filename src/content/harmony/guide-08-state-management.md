---
title: "状态管理 V2：新项目怎么选、怎么迁"
description: "V2 从 API 12 起提供，核心变化是更明确的输入输出和更强的观测能力。V1 和 V2 不是简单的“新旧替换”，同一个组件内不能混用两套装饰器。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 8
minutes: 5
---

V2 从 API 12 起提供，核心变化是更明确的输入输出和更强的观测能力。V1 和 V2 不是简单的“新旧替换”，同一个组件内不能混用两套装饰器。

| 维度 | V1 | V2 |
| --- | --- | --- |
| 组件 | `@Component` | `@ComponentV2` |
| 内部状态 | `@State` | `@Local` |
| 父传子 | `@Prop` | `@Param` |
| 只接收一次 | 无直接对应 | `@Param @Once` |
| 子传父 | `@Link` | `@Param + @Event` |
| 跨层 | `@Provide/@Consume` | `@Provider/@Consumer` |
| 监听 | `@Watch` | `@Monitor` |
| 深度观测 | `@Observed/@ObjectLink` | `@ObservedV2/@Trace` |
| 计算属性 | 无原生对应 | `@Computed` |
| 适用 | 旧代码、V1 三方组件 | 新项目、API 12+ |

最小 V2 示例，作为 `pages/V2Demo.ets` 运行，并注册到 `main_pages.json`：

```typescript
@Entry
@ComponentV2
struct V2Demo {
  @Local count: number = 0;
  @Local items: string[] = ['A'];

  @Computed
  get total(): number {
    return this.items.length + this.count;
  }

  build() {
    Column({ space: 12 }) {
      Text(`count=${this.count}, total=${this.total}`)
      Button('+1').onClick(() => {
        this.count++;
      })
      V2Child({
        title: `父组件的 count=${this.count}`,
        onceTitle: '只接收第一次',
        onAdd: (delta: number) => {
          this.count += delta;
        }
      })
    }
    .padding(16)
  }
}

@ComponentV2
struct V2Child {
  @Param title: string = '';
  @Param @Once onceTitle: string = '';
  @Event onAdd: (delta: number) => void = (delta: number) => {
    console.info(`未处理的增量 ${delta}`);
  };

  build() {
    Column({ space: 8 }) {
      Text(this.title)
      Text(this.onceTitle)
      Button('子组件 +1').onClick(() => {
        this.onAdd(1);
      })
      Button('本地改 @Once').onClick(() => {
        this.onceTitle = '本地已改';
      })
    }
  }
}
```

`@Param` 变量本身只读，不能整体重新赋值；`@Once` 只接收父组件第一次给的值，之后允许本地改，父组件再更新也不会覆盖；`@Event` 是子组件对外的回调，父组件传函数进来，子组件调用它。`@Computed` 只根据依赖重新计算，不要在 getter 里做副作用。`@Param` 接收类对象时，内部属性可以改并同步回源，但默认只观测整体替换；要观测类属性深层变化，给类加 `@ObservedV2`、给字段加 `@Trace`。

新项目怎么选：如果是 API 12+、没有历史包袱，优先 V2，数据流更清楚，类对象深层观测用 `@ObservedV2` + `@Trace`，跨层用 `@Provider/@Consumer`，全局状态再看 `AppStorageV2` / `PersistenceV2`（API 12 起，具体用法以官方文档为准）。如果团队已有大量 V1 代码、依赖 V1 三方组件，继续 V1 更稳，不要为了“新”把半个工程改成 V2。

迁移注意事项：

- `@Component` 换 `@ComponentV2`，`@State` 换 `@Local`，`@Prop` 换 `@Param`，`@Provide/@Consume` 换 `@Provider/@Consumer`，`@Watch` 换 `@Monitor`，`@Observed/@ObjectLink` 换 `@ObservedV2/@Trace`。
- `@Link` 没有直接对应：改成 `@Param` + `@Event`，子组件发事件、父组件改状态；跨层双向用 `@Provider/@Consumer`。
- `@Param` 默认只读；子组件确实要本地改，才加 `@Once`，或者复制到 `@Local` 再通过 `@Event` 回传。
- 去掉 `$变量` 的 `@Link` 写法；V2 不靠 `$` 建双向绑定。
- 同一个组件不要一半 V1、一半 V2；迁移按组件整块替换，先改叶子组件，再改父组件。
- 迁移顺序建议：先做无状态展示组件，再做有本地状态的组件，最后做跨层共享和页面级状态；每改一个组件就编译、跑一次父子传值和列表刷新。
- V1 的 `@StorageLink`、`@StorageProp`、`@LocalStorageLink` 在 V2 里没有同名替代，要看 `AppStorageV2`、`PersistenceV2` 或 `@Provider/@Consumer` 是否满足场景，不要硬套。
- V1/V2 组件嵌套、混用时的数据传递限制，以官方《状态管理 V1V2 混用》文档为准，别凭感觉写。
