---
title: "第 8 讲：网络与异步"
description: "用 @kit.NetworkKit 的 http 拉一次网络数据，做成「每日一句」彩蛋，并写全超时、失败、重试三条路径。"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "实战课程"]
track: "course"
order: 8
minutes: 5
---

### 本讲目标

用 `@kit.NetworkKit` 的 `http` 拉一次网络数据，做成「每日一句」彩蛋，并写全超时、失败、重试三条路径。

做完你会看到：列表页顶部出现一句随机短句和出处；断网时显示「加载失败，点击重试」且不崩溃。

### 为什么这样设计

网络请求必须异步，因为主线程要用来渲染。同步请求哪怕只卡 200ms，用户也能感知到掉帧；卡 5 秒，系统会直接判定应用无响应。

超时和重试不是技术洁癖，是产品需求。移动网络下首次请求失败率不低，用户看到的应该是「马上重试成功」，而不是「转圈到天荒地老」。所以要有连接超时、读取超时、有限次重试、指数退避 —— 退避是为了不给服务器雪崩式压力。

网络层单独抽 `service/`，是为了让 UI 只关心「成功给我什么、失败显示什么」。UI 里出现 `http.createHttp()` 的那天，就是这个项目开始难维护的那天。

**注意**：本讲用的 `https://v1.hitokoto.cn/` 是第三方公开示例接口，可用性不由本项目保证。真实项目请换成你自己的服务端，或至少准备好本地兜底文案（本讲代码的降级路径就是这个作用）。

### 跟着做

**1. 声明网络权限。** 打开 `entry/src/main/module.json5`，在 `module` 里加：

```json5
"requestPermissions": [
  { "name": "ohos.permission.INTERNET" }
]
```

**2. 新建 `entry/src/main/ets/service/QuoteService.ets`：**

```ts
import { http } from '@kit.NetworkKit';

export interface Quote {
  text: string;
  from: string;
}

interface QuoteResponse {
  hitokoto: string;   // 接口字段名
  from: string;
}

const ENDPOINT: string = 'https://v1.hitokoto.cn/?encode=json';
const TIMEOUT_MS: number = 5000;
const MAX_RETRY: number = 2;

function sleep(ms: number): Promise<void> {
  return new Promise<void>((resolve: (value: void) => void) => {
    setTimeout(() => resolve(), ms);
  });
}

// 单次请求：负责建连、校验、销毁
async function requestOnce(): Promise<Quote> {
  const client: http.HttpRequest = http.createHttp();
  try {
    const res: http.HttpResponse = await client.request(ENDPOINT, {
      method: http.RequestMethod.GET,
      connectTimeout: TIMEOUT_MS,
      readTimeout: TIMEOUT_MS,
      expectDataType: http.HttpDataType.STRING   // 不设会拿到 Object，别问为什么打印是 [object Object]
    });
    if (res.responseCode !== http.ResponseCode.OK) {
      throw new Error(`HTTP ${res.responseCode}`);
    }
    const body: QuoteResponse = JSON.parse(res.result as string) as QuoteResponse;
    return { text: body.hitokoto, from: body.from };
  } finally {
    client.destroy();     // 必须销毁，否则句柄泄漏
  }
}

// 对外只暴露这一个函数：带重试与指数退避
export async function fetchDailyQuote(): Promise<Quote> {
  let lastError: Error = new Error('未知错误');
  for (let i: number = 0; i <= MAX_RETRY; i++) {
    try {
      return await requestOnce();
    } catch (e) {
      lastError = e as Error;
      if (i < MAX_RETRY) {
        await sleep(500 * Math.pow(2, i));   // 0.5s、1s
      }
    }
  }
  throw lastError;
}
```

**3. 在 `Index.ets` 顶部加卡片。** 三个状态变量 + 一个加载方法：

```ts
import { Quote, fetchDailyQuote } from '../service/QuoteService';

@State quote: Quote = { text: '正在加载今日一句…', from: '' };
@State quoteLoading: boolean = false;
@State quoteFailed: boolean = false;

private async loadQuote(): Promise<void> {
  if (this.quoteLoading) {
    return;                      // 天然防抖：上一发没回来就别再发
  }
  this.quoteLoading = true;
  this.quoteFailed = false;
  try {
    this.quote = await fetchDailyQuote();
  } catch (e) {
    this.quoteFailed = true;
    this.quote = { text: '今日一句加载失败，检查网络后重试', from: '' };
  } finally {
    this.quoteLoading = false;
  }
}

// build() 里 List 之前插入：
Column({ space: 4 }) {
  Text(this.quote.text).fontSize(15).fontColor('#3A3A3C')
  if (this.quote.from.length > 0) {
    Text(`—— ${this.quote.from}`).fontSize(12).fontColor('#8A8A8E')
  }
}
.width('100%')
.padding(14)
.backgroundColor('#FFFFFF')
.borderRadius(14)
.onClick(() => {
  if (this.quoteFailed) {
    this.loadQuote();
  }
})
```

在 `aboutToAppear` 里调一次 `this.loadQuote()`。这里用 `onPageShow` 也合理 —— 从后台切回来会重新触发。

### 容易踩的坑

- **现象**：请求直接报错，错误码 201。**原因**：`module.json5` 没声明 `ohos.permission.INTERNET`。**改法**：加上并重新安装（权限变更必须重装，热重载不生效）。
- **现象**：`res.result` 打印出来是 `[object Object]`。**原因**：没设 `expectDataType: http.HttpDataType.STRING`。**改法**：显式声明返回类型。
- **现象**：模拟器上一直失败，真机正常。**原因**：模拟器的网络走宿主机代理，代理没配。**改法**：模拟器设置里配 HTTP 代理，或直接用真机验证。
- **现象**：界面在请求期间卡住。**原因**：忘了 `await`，或在同步回调里阻塞。**改法**：全链路 async，UI 用 loading 态表达进度。
- **现象**：失败后重复点击，日志里出现多个并发请求。**原因**：没有用 `quoteLoading` 挡住。**改法**：入口处判空返回（上面已写）。
- **现象**：超时时间设了但没生效。**原因**：只设了 `connectTimeout` 没设 `readTimeout`，连接快但响应慢。**改法**：两个都设，且 `readTimeout` 通常要大于 `connectTimeout`。

### 自检

- 打开 App，顶部卡片应在 1–2 秒内显示一句中文短句和出处。
- 打开模拟器飞行模式后重进，应显示加载失败文案，且 App 不崩溃、列表仍可操作。
- 点失败文案，应重新发起请求；恢复网络后能成功。
- 连续快速点 5 次失败文案，Log 里同一时刻只应有 1 个请求。
- 打开 Log，请求结束后应有 `destroy` 相关行为，再次请求没有句柄泄漏告警。

### 本讲产出

`service/QuoteService.ets` 网络层 + 一个带完整失败处理的「每日一句」彩蛋。

### 进阶挑战

把结果按日期缓存进 `Preferences`（key 用 `quote_YYYY-MM-DD`），同一天内不再请求；再加一个「换一句」按钮，点它时绕过缓存强制刷新。要求：缓存命中时首屏不应有任何 loading 闪烁。

---
