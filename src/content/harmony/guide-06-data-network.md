---
title: "数据与网络：Preferences、RDB、文件、HTTP"
description: "选型顺序可以简单记：配置、开关、上次打开的页面用 Preferences；需要按条件查、排序、分页、事务的用 RDB；导入导出、大文本、图片、日志用文件。网络拉回来的结构化数据，先落 RDB …"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 6
minutes: 5
---

| 方案 | 数据模型 | 适合 | 不适合 |
| --- | --- | --- | --- |
| Preferences | Key-Value | 配置、草稿、少量用户偏好 | 大列表、复杂查询、多进程并发 |
| RDB | 关系表 + SQL | 待办、记账、聊天记录等结构化数据 | 超大文件、图片二进制 |
| 文件 | 文本/JSON/二进制 | 导入导出、日志、缓存、图片 | 需要事务和条件查询的数据 |

选型顺序可以简单记：配置、开关、上次打开的页面用 Preferences；需要按条件查、排序、分页、事务的用 RDB；导入导出、大文本、图片、日志用文件。网络拉回来的结构化数据，先落 RDB 再渲染，别把整个列表塞进 Preferences；敏感凭据不要明文存 Preferences，用 HUKS 这类密钥能力。

**Preferences** 轻量、简单，值类型支持 number/string/boolean 及其数组。API 10 起可用 `getPreferences(context, options)` 形式。改完必须 `flush()` 才落盘；它不保证多进程并发安全。

```typescript
import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';

async function saveDraft(context: common.UIAbilityContext, text: string): Promise<void> {
  const store: preferences.Preferences = await preferences.getPreferences(context, { name: 'draft' });
  await store.put('text', text);
  await store.flush();
}

async function loadDraft(context: common.UIAbilityContext): Promise<string> {
  const store: preferences.Preferences = await preferences.getPreferences(context, { name: 'draft' });
  const value: preferences.ValueType = await store.get('text', '');
  return value as string;
}
```

**RDB** 基于 SQLite，适合列表、条件查询、事务。`securityLevel` 只能由低向高设置，不能降级。查询结果 `ResultSet` 必须 `close()`，否则会泄漏。

```typescript
import { relationalStore } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';

const CONFIG: relationalStore.StoreConfig = {
  name: 'demo.db',
  securityLevel: relationalStore.SecurityLevel.S1
};

async function openStore(context: common.UIAbilityContext): Promise<relationalStore.RdbStore> {
  const store: relationalStore.RdbStore = await relationalStore.getRdbStore(context, CONFIG);
  await store.executeSql(
    'CREATE TABLE IF NOT EXISTS note (id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL)'
  );
  return store;
}

async function addNote(store: relationalStore.RdbStore, title: string): Promise<number> {
  const values: relationalStore.ValuesBucket = { title: title };
  return await store.insert('note', values);
}

async function listNotes(store: relationalStore.RdbStore): Promise<string[]> {
  const predicates: relationalStore.RdbPredicates = new relationalStore.RdbPredicates('note');
  const resultSet: relationalStore.ResultSet = await store.query(predicates, ['id', 'title']);
  const titles: string[] = [];
  try {
    while (resultSet.goToNextRow()) {
      titles.push(resultSet.getString(resultSet.getColumnIndex('title')));
    }
  } finally {
    resultSet.close();
  }
  return titles;
}
```

**文件读写** 适合 JSON 导入导出、日志、图片。应用沙箱用 `context.filesDir`，不要硬编码路径。大文件用异步接口或 TaskPool，别在主线程同步读写几百 MB。

```typescript
import { fileIo as fs } from '@kit.CoreFileKit';
import { util } from '@kit.ArkTS';
import { common } from '@kit.AbilityKit';

function writeText(context: common.UIAbilityContext, text: string): void {
  const path: string = `${context.filesDir}/note.txt`;
  const file: fs.File = fs.openSync(path, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE | fs.OpenMode.TRUNC);
  try {
    fs.writeSync(file.fd, text);
  } finally {
    fs.closeSync(file);
  }
}

function readText(context: common.UIAbilityContext): string {
  const path: string = `${context.filesDir}/note.txt`;
  try {
    const stat: fs.Stat = fs.statSync(path);
    const buffer: ArrayBuffer = new ArrayBuffer(stat.size);
    const file: fs.File = fs.openSync(path, fs.OpenMode.READ_ONLY);
    try {
      const bytesRead: number = fs.readSync(file.fd, buffer);
      const decoder: util.TextDecoder = util.TextDecoder.create('utf-8');
      return decoder.decodeToString(new Uint8Array(buffer.slice(0, bytesRead)));
    } finally {
      fs.closeSync(file);
    }
  } catch (error) {
    return '';
  }
}
```

**HTTP** 用 `@kit.NetworkKit` 的 `http` 模块。必须在 `module.json5` 声明 `ohos.permission.INTERNET`（system_grant，安装即授予）。`connectTimeout` 和 `readTimeout` 单位毫秒；每个 `HttpRequest` 只对应一个请求，用完 `destroy()`。

```typescript
import { http } from '@kit.NetworkKit';
import { BusinessError } from '@kit.BasicServicesKit';

interface TodoDto {
  id: number;
  title: string;
  completed: boolean;
}

async function fetchTodo(id: number): Promise<TodoDto> {
  const client: http.HttpRequest = http.createHttp();
  try {
    const response: http.HttpResponse = await client.request(
      `https://jsonplaceholder.typicode.com/todos/${id}`,
      {
        method: http.RequestMethod.GET,
        connectTimeout: 10000,
        readTimeout: 10000,
        expectDataType: http.HttpDataType.STRING
      }
    );
    if (response.responseCode !== 200) {
      throw new Error(`HTTP ${response.responseCode}`);
    }
    const body: string = response.result as string;
    return JSON.parse(body) as TodoDto;
  } catch (error) {
    const err: BusinessError = error as BusinessError;
    const message: string = err.message ?? '网络请求失败';
    console.error(`请求失败 code=${err.code}, message=${message}`);
    throw new Error(message);
  } finally {
    client.destroy();
  }
}
```

HTTP 还要注意：真机没网、域名证书、明文 HTTP 限制、服务端跨域策略都可能让请求失败；先看 `responseCode` 和日志，不要只写一个 `catch` 吞掉。生产代码还应该加鉴权头、取消机制、有限重试和证书校验；不要把 token、手机号、完整响应体打进日志。需要 POST 时把 `method` 改成 `http.RequestMethod.POST`，用 `extraData` 传 JSON 字符串，并设置 `Content-Type`。请求返回后如果要更新 UI，记得回到主线程；ArkTS 的 `async/await` 默认在 UI 线程恢复，耗时解析大 JSON 时再考虑 TaskPool。
