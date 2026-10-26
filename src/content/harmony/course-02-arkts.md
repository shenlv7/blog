---
title: "第 2 讲：ArkTS 速成，只学这个项目要用的部分"
description: "掌握能支撑本项目的 ArkTS 子集：类型注解、interface、async/await、模块导入，并知道哪些 TS 写法在 ArkTS 里编译不过。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 2
minutes: 5
---

### 本讲目标

掌握能支撑本项目的 ArkTS 子集：类型注解、interface、async/await、模块导入，并知道哪些 TS 写法在 ArkTS 里编译不过。

做完你会看到：`model/Habit.ets` 里有一个 `Habit` 接口和三个工具函数，编译零告警。

### 为什么这样设计

ArkTS 是 TypeScript 的**静态子集**，不是 TS 加几个装饰器。它砍掉了 `any`、结构化类型、运行时改对象布局这些能力，换来的是编译期可确定的类型和可追踪的状态依赖。这不是保守，是必要：声明式 UI 靠「谁依赖了谁」来决定重绘哪一块，一旦允许 `any`，依赖图就断了。

数据用 `interface` 而不是 `class`，是为了持久化和网络。`JSON.parse` 只能还原数据，还原不了原型和方法。把 `Habit` 定义成纯数据结构，方法写成独立函数，第 7 讲存盘和第 8 讲解析时才不会踩坑。

从 API 12 起，官方推荐用 Kit 聚合包导入（`@kit.NetworkKit`），旧的 `@ohos.net.http` 写法仍能用，但新代码统一用 Kit。

### 跟着做

**1. 建数据模型。** 新建 `entry/src/main/ets/model/Habit.ets`：

```ts
// 一条习惯。用 interface 而非 class：要能被 JSON 序列化/反序列化
export interface Habit {
  id: string;
  name: string;
  icon: string;       // emoji 简易图标，省掉图片资源
  createdAt: number;  // 毫秒时间戳
  records: string[];  // 已打卡日期，格式 'YYYY-MM-DD'，用 Set 语义去重
}

// 把日期格式化成 'YYYY-MM-DD'，本地时区
export function dateKey(d: Date): string {
  const m: string = (d.getMonth() + 1).toString().padStart(2, '0');
  const day: string = d.getDate().toString().padStart(2, '0');
  return `${d.getFullYear()}-${m}-${day}`;
}

export function todayKey(): string {
  return dateKey(new Date());
}

// 连续天数：今天没打卡时从昨天往前算，避免“今天还没打卡就断签”
export function calcStreak(records: string[]): number {
  if (records.length === 0) {
    return 0;
  }
  const marked: Set<string> = new Set<string>(records);
  const cursor: Date = new Date();
  if (!marked.has(dateKey(cursor))) {
    cursor.setDate(cursor.getDate() - 1);
  }
  let streak: number = 0;
  while (marked.has(dateKey(cursor))) {
    streak++;
    cursor.setDate(cursor.getDate() - 1);
  }
  return streak;
}

// async/await 的最小范式：永远配 try/catch
export async function delay(ms: number): Promise<void> {
  await new Promise<void>((resolve: (value: void) => void) => {
    setTimeout(() => resolve(), ms);
  });
}
```

**2. 验证模块导入。** 在 `pages/Index.ets` 顶部加一行，并在 `build()` 里用一次：

```ts
import { calcStreak, todayKey } from '../model/Habit';

// build() 里临时加：
Text(`今天 ${todayKey()}，连续 ${calcStreak([])} 天`)
```

路径是相对当前文件的，**不带 `.ets` 后缀**。Ctrl+点击 `calcStreak` 能跳转就说明导入成功。

**3. 记住这张禁用表。** 这不是全部，但覆盖你 90% 的报错：

| TypeScript 写法 | ArkTS 中的处理 |
| --- | --- |
| `any` / `unknown` | 编译报错。用具体类型、联合类型或泛型 |
| 结构化类型（鸭子类型） | 不支持。必须显式 `implements` 接口 |
| `obj.newField = 1`、`delete obj.x` | 不支持运行时改对象布局。字段必须在 interface/class 里声明 |
| 对象字面量不标注类型 | 报错。写 `const h: Habit = { ... }` |
| `#private` 私有字段 | 不支持。用 `private` |
| `Function` 类型、`fn.apply/call/bind` | 不支持。用显式函数类型 `(id: string) => void` 和箭头函数 |
| `eval` / `new Function` / `with` | 禁止 |
| 跨文件 `interface` 声明合并 | 不支持。一个接口只在一个文件里定义完整 |
| `namespace` | 不支持。用 ES module 的 `export` / `import` |

完整清单以官方文档《从 TypeScript 到 ArkTS 的适配规则》为准，遇到拿不准的写法先去那一页搜规则名。

### 容易踩的坑

- **现象**：`Object literal must correspond to some explicitly declared class or interface`。**原因**：`const h = { id: '1', name: 'x' }` 没有目标类型。**改法**：`const h: Habit = { ... }`。
- **现象**：`Argument of type 'X' is not assignable to parameter of type 'Y'`，但两个对象字段一模一样。**原因**：ArkTS 不做结构化匹配。**改法**：让类显式 `implements` 接口，或写一个转换函数。
- **现象**：`Cannot find module '../model/Habit'`。**原因**：路径写错、少了 `../`，或文件名大小写不一致。**改法**：从当前文件出发写相对路径，不带扩展名。
- **现象**：`JSON.parse` 出来的对象调用方法时报 `is not a function`。**原因**：JSON 只还原数据不还原原型。**改法**：模型只放数据，逻辑放独立函数（本讲已经这么做）。

### 自检

- 打开 `Habit.ets`，`calcStreak([])` 应返回 `0`。
- `calcStreak([todayKey()])` 应返回 `1`。
- 编译窗口里 `Habit.ets` 无红色波浪线、无告警。
- 在 `Index.ets` 里 Ctrl+点击 `Habit`，能跳到定义处。

### 本讲产出

`model/Habit.ets`：项目的数据结构、日期工具、连续天数算法。

### 进阶挑战

给 `Habit` 加 `targetPerWeek: number`，写 `isGoalMet(records: string[], target: number): boolean`（统计本周打卡数）。注意本周起点是周一，用 `getDay()` 时周日是 0，要特殊处理。

---
