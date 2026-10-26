---
title: "页面路由与导航：新项目用 Navigation"
description: "router 是页面级路由，靠 main_pages.json 注册，router.pushUrl / router.back 传参。Navigation 是组件级导航，靠 NavPathSt…"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 5
minutes: 5
---

`router` 是页面级路由，靠 `main_pages.json` 注册，`router.pushUrl` / `router.back` 传参。`Navigation` 是组件级导航，靠 `NavPathStack` 管理页面栈，支持单栏/分栏、转场、状态保留，官方新范式推荐它。

| 维度 | router | Navigation |
| --- | --- | --- |
| 路由单位 | 页面 | NavDestination |
| 注册方式 | `main_pages.json` 必填 | 首页注册，子页用 builder/route_map |
| 传参 | `params` 对象 | `pushPathByName(name, param)` |
| 返回传值 | `router.back({ params })` | `pop(result)` + `onPop` |
| 分栏/大屏 | 支持有限 | 原生 Split/Auto |
| 新项目建议 | 旧项目、极简页面 | 默认选它 |

下面是一组完整可运行的最小示例。`NavIndex.ets` 要注册到 `main_pages.json`，`NavDetail.ets` 是 NavDestination，不用注册。`pages/NavIndex.ets`：

```typescript
import { NavDetail } from './NavDetail';

@Entry
@ComponentV2
struct NavIndex {
  private stack: NavPathStack = new NavPathStack();

  @Builder
  pageMap(name: string, param: Object) {
    if (name === 'detail') {
      NavDetail({ id: param as string })
    }
  }

  build() {
    Navigation(this.stack) {
      Button('带参数进入详情')
        .onClick(() => {
          this.stack.pushPathByName('detail', '42', (popInfo: PopInfo) => {
            const result: string = popInfo.result as string;
            console.info(`详情页返回：${result}`);
          });
        })
    }
    .title('首页')
    .navDestination(this.pageMap)
  }
}
```

`pages/NavDetail.ets`：

```typescript
@ComponentV2
export struct NavDetail {
  @Param id: string = '';
  private stack: NavPathStack = new NavPathStack();

  build() {
    NavDestination() {
      Column({ space: 12 }) {
        Text(`收到参数 id = ${this.id}`)
        Button('返回并传值').onClick(() => {
          this.stack.pop(`已处理 ${this.id}`);
        })
      }
      .padding(16)
    }
    .title('详情')
    .onReady((context: NavDestinationContext) => {
      this.stack = context.pathStack;
    })
  }
}
```

`NavDestination` 的根必须是 `NavDestination`，不要在外面再包一层带属性/事件的自定义组件，否则可能白屏。返回传值的关键是 `pop(result)`，上一页在 `pushPathByName` 的 `onPop` 回调里取 `popInfo.result`。子页面拿路由栈不要靠全局变量，用 `.onReady((context: NavDestinationContext) => { this.stack = context.pathStack; })`。

为什么新项目偏向 Navigation？因为 `router` 是全局页面栈，页面返回时容易重建，跨模块、分栏、状态保留和转场都弱一些；Navigation 把路由栈放在组件树里，`NavPathStack` 可以精确 push/pop、替换栈顶，还能直接支持平板分栏。代价是你要自己写 `NavDestination` builder 和返回结果处理，初看比 `router` 多几行。旧项目如果 router 跑得稳，不必为了“新”迁移；新项目直接按 Navigation 设计，后面加详情页、弹窗页、分栏都更顺。

如果坚持用 `router`，跳转调用 `router.pushUrl` 并传 `url` 与 `params`，返回调用 `router.back` 并传 `url` 与 `params`，上一页在 `onPageShow` 里用 `router.getParams()` 取值并做类型断言。`pages/Detail` 必须写进 `main_pages.json`，否则 404。新项目不建议同时维护两套路由。
