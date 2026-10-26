---
title: "完整练手项目：待办清单 Todo"
description: "功能：增删改查、Preferences 本地持久化、列表页到详情页的 Navigation 跳转、返回后刷新、深色模式适配。状态管理用 V2，路由用 Navigation。按文件给出，全部放在…"
pubDate: 2026-10-26
tags: ["HarmonyOS", "ArkTS", "小白指南"]
track: "guide"
order: 10
minutes: 5
---

功能：增删改查、Preferences 本地持久化、列表页到详情页的 Navigation 跳转、返回后刷新、深色模式适配。状态管理用 V2，路由用 Navigation。按文件给出，全部放在 `entry/src/main/ets` 和 `entry/src/main/resources` 下。

代码分三层：`model/Todo.ets` 只管数据结构和 Preferences 读写，`components/TodoRow.ets` 只做展示和回调，`pages/Index.ets` 与 `pages/TodoDetail.ets` 管状态和导航。这样以后把 Preferences 换成 RDB，只改 repository，页面基本不动。详情页只接收 `todoId`，自己按 id 查数据，避免“列表一份、详情一份”两个对象互相覆盖。Preferences 里存的是整个数组的 JSON，适合几十到几百条；上千条或要按条件查询时换 RDB。

### 10.1 `AppScope/app.json5`

```json5
{
  "app": {
    "bundleName": "com.example.todo",
    "vendor": "example",
    "versionCode": 1000000,
    "versionName": "1.0.0",
    "icon": "$media:app_icon",
    "label": "$string:app_name"
  }
}
```

`app_icon.png` 用模板自带的，或换成自己的 1024×1024 图标；`app_name` 写在 `AppScope/resources/base/element/string.json`。

### 10.2 根 `build-profile.json5`

```json5
{
  "app": {
    "signingConfigs": [],
    "products": [
      {
        "name": "default",
        "signingConfig": "default",
        "compatibleSdkVersion": "5.0.0(12)",
        "runtimeOS": "HarmonyOS"
      }
    ],
    "buildModeSet": [
      { "name": "debug" },
      { "name": "release" }
    ]
  },
  "modules": [
    {
      "name": "entry",
      "srcPath": "./entry",
      "targets": [
        {
          "name": "default",
          "applyToProducts": ["default"]
        }
      ]
    }
  ]
}
```

### 10.3 `entry/src/main/module.json5`

```json5
{
  "module": {
    "name": "entry",
    "type": "entry",
    "description": "$string:module_desc",
    "mainElement": "EntryAbility",
    "deviceTypes": ["phone", "tablet"],
    "deliveryWithInstall": true,
    "installationFree": false,
    "pages": "$profile:main_pages",
    "abilities": [
      {
        "name": "EntryAbility",
        "srcEntry": "./ets/entryability/EntryAbility.ets",
        "description": "$string:EntryAbility_desc",
        "icon": "$media:layered_image",
        "label": "$string:EntryAbility_label",
        "startWindowIcon": "$media:startIcon",
        "startWindowBackground": "$color:start_window_background",
        "exported": true,
        "skills": [
          {
            "entities": ["entity.system.home"],
            "actions": ["action.system.home"]
          }
        ]
      }
    ]
  }
}
```

这里的 `layered_image`、`startIcon` 用模板生成的文件名；如果你的模板是 `icon.png`，就保留模板原来的值，不要硬改成不存在的资源。

### 10.4 `entry/src/main/resources/base/profile/main_pages.json`

```json
{
  "src": [
    "pages/Index"
  ]
}
```

详情页是 `NavDestination`，不在这里注册。

### 10.5 `entry/src/main/ets/entryability/EntryAbility.ets`

```typescript
import { AbilityConstant, UIAbility, Want } from '@kit.AbilityKit';
import { window } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { BusinessError } from '@kit.BasicServicesKit';

export default class EntryAbility extends UIAbility {
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    hilog.info(0x0000, 'Todo', 'EntryAbility onCreate');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    windowStage.loadContent('pages/Index', (err: BusinessError) => {
      if (err.code) {
        hilog.error(0x0000, 'Todo', 'loadContent 失败: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(0x0000, 'Todo', 'loadContent 成功');
    });
  }
}
```

### 10.6 `entry/src/main/ets/model/Todo.ets`

```typescript
import { preferences } from '@kit.ArkData';
import { common } from '@kit.AbilityKit';

const STORE_NAME: string = 'todo_store';
const KEY_TODOS: string = 'todos';

export class Todo {
  id: string;
  title: string;
  note: string;
  done: boolean;
  createdAt: number;

  constructor(id: string, title: string, note: string = '', done: boolean = false, createdAt: number = Date.now()) {
    this.id = id;
    this.title = title;
    this.note = note;
    this.done = done;
    this.createdAt = createdAt;
  }
}

export class TodoRepository {
  private store: preferences.Preferences | null = null;

  async init(context: common.UIAbilityContext): Promise<void> {
    this.store = await preferences.getPreferences(context, { name: STORE_NAME });
  }

  private requireStore(): preferences.Preferences {
    if (this.store === null) {
      throw new Error('TodoRepository 未初始化');
    }
    return this.store;
  }

  async list(): Promise<Todo[]> {
    const value: preferences.ValueType = await this.requireStore().get(KEY_TODOS, '[]');
    const raw: string = value as string;
    try {
      return JSON.parse(raw) as Todo[];
    } catch (error) {
      return [];
    }
  }

  async saveAll(todos: Todo[]): Promise<void> {
    await this.requireStore().put(KEY_TODOS, JSON.stringify(todos));
    await this.requireStore().flush();
  }

  async add(title: string): Promise<Todo[]> {
    const todos: Todo[] = await this.list();
    todos.unshift(new Todo(this.newId(), title));
    await this.saveAll(todos);
    return todos;
  }

  async toggle(id: string): Promise<Todo[]> {
    const todos: Todo[] = await this.list();
    for (let i = 0; i < todos.length; i++) {
      if (todos[i].id === id) {
        todos[i].done = !todos[i].done;
        break;
      }
    }
    await this.saveAll(todos);
    return todos;
  }

  async remove(id: string): Promise<Todo[]> {
    const todos: Todo[] = await this.list();
    const next: Todo[] = todos.filter((item: Todo) => item.id !== id);
    await this.saveAll(next);
    return next;
  }

  async find(id: string): Promise<Todo | null> {
    const todos: Todo[] = await this.list();
    for (let i = 0; i < todos.length; i++) {
      if (todos[i].id === id) {
        return todos[i];
      }
    }
    return null;
  }

  async update(todo: Todo): Promise<void> {
    const todos: Todo[] = await this.list();
    for (let i = 0; i < todos.length; i++) {
      if (todos[i].id === todo.id) {
        todos[i] = todo;
        break;
      }
    }
    await this.saveAll(todos);
  }

  private newId(): string {
    return `${Date.now()}-${Math.floor(Math.random() * 1000000)}`;
  }
}
```

### 10.7 `entry/src/main/ets/components/TodoRow.ets`

```typescript
@ComponentV2
export struct TodoRow {
  @Param id: string = '';
  @Param title: string = '';
  @Param note: string = '';
  @Param done: boolean = false;
  @Event onToggle: () => void = () => {};
  @Event onOpen: () => void = () => {};
  @Event onDelete: () => void = () => {};

  build() {
    Row({ space: 10 }) {
      Checkbox({ name: this.id, group: 'todo' })
        .select(this.done)
        .onChange((value: boolean) => {
          this.onToggle();
        })

      Column({ space: 2 }) {
        Text(this.title)
          .fontSize(16)
          .fontColor(this.done ? $r('app.color.text_muted') : $r('app.color.text_primary'))
          .decoration({
            type: this.done ? TextDecorationType.LineThrough : TextDecorationType.None
          })
        if (this.note.length > 0) {
          Text(this.note)
            .fontSize(12)
            .fontColor($r('app.color.text_secondary'))
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)
      .onClick(() => {
        this.onOpen();
      })

      Button('删除')
        .fontSize(12)
        .height(32)
        .backgroundColor(Color.Transparent)
        .fontColor($r('app.color.danger'))
        .onClick(() => {
          this.onDelete();
        })
    }
    .width('100%')
    .padding(12)
    .backgroundColor($r('app.color.card_bg'))
    .borderRadius(12)
  }
}
```

### 10.8 `entry/src/main/ets/pages/Index.ets`

```typescript
import { common } from '@kit.AbilityKit';
import { promptAction } from '@kit.ArkUI';
import { Todo, TodoRepository } from '../model/Todo';
import { TodoRow } from '../components/TodoRow';
import { TodoDetail, TodoDetailResult } from './TodoDetail';

@Entry
@ComponentV2
struct Index {
  private stack: NavPathStack = new NavPathStack();
  private repo: TodoRepository = new TodoRepository();

  @Local todos: Todo[] = [];
  @Local input: string = '';
  @Local loading: boolean = true;

  aboutToAppear(): void {
    this.load();
  }

  private async initRepo(): Promise<void> {
    const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
    await this.repo.init(context);
  }

  private async load(): Promise<void> {
    try {
      await this.initRepo();
      this.todos = await this.repo.list();
    } catch (error) {
      promptAction.showToast({ message: '读取待办失败' });
    } finally {
      this.loading = false;
    }
  }

  private async add(): Promise<void> {
    const title: string = this.input.trim();
    if (title.length === 0) {
      promptAction.showToast({ message: '请输入待办内容' });
      return;
    }
    try {
      this.todos = await this.repo.add(title);
      this.input = '';
    } catch (error) {
      promptAction.showToast({ message: '保存失败' });
    }
  }

  private async toggle(id: string): Promise<void> {
    try {
      this.todos = await this.repo.toggle(id);
    } catch (error) {
      promptAction.showToast({ message: '更新失败' });
    }
  }

  private async remove(id: string): Promise<void> {
    try {
      this.todos = await this.repo.remove(id);
    } catch (error) {
      promptAction.showToast({ message: '删除失败' });
    }
  }

  private openDetail(id: string): void {
    this.stack.pushPathByName('TodoDetail', id, (popInfo: PopInfo) => {
      const result: TodoDetailResult | undefined = popInfo.result as TodoDetailResult | undefined;
      if (result !== undefined && result.changed) {
        this.load();
      }
    });
  }

  @Builder
  pageMap(name: string, param: Object) {
    if (name === 'TodoDetail') {
      TodoDetail({ todoId: param as string })
    }
  }

  build() {
    Navigation(this.stack) {
      Column({ space: 12 }) {
        Row({ space: 8 }) {
          TextInput({ text: this.input, placeholder: '今天要做什么？' })
            .layoutWeight(1)
            .onChange((value: string) => {
              this.input = value;
            })
          Button('添加')
            .enabled(!this.loading)
            .onClick(() => {
              this.add();
            })
        }
        .width('100%')

        if (this.loading) {
          LoadingProgress().width(32).height(32)
        } else if (this.todos.length === 0) {
          Text('还没有待办，先添加一条')
            .fontColor($r('app.color.text_secondary'))
            .margin({ top: 48 })
        } else {
          List({ space: 10 }) {
            ForEach(this.todos, (item: Todo) => {
              ListItem() {
                TodoRow({
                  id: item.id,
                  title: item.title,
                  note: item.note,
                  done: item.done,
                  onToggle: () => {
                    this.toggle(item.id);
                  },
                  onOpen: () => {
                    this.openDetail(item.id);
                  },
                  onDelete: () => {
                    this.remove(item.id);
                  }
                })
              }
            }, (item: Todo) => item.id)
          }
          .layoutWeight(1)
          .width('100%')
        }
      }
      .padding(16)
      .width('100%')
      .height('100%')
      .backgroundColor($r('app.color.page_bg'))
    }
    .title('待办清单')
    .titleMode(NavigationTitleMode.Mini)
    .navDestination(this.pageMap)
    .mode(NavigationMode.Stack)
  }
}
```

### 10.9 `entry/src/main/ets/pages/TodoDetail.ets`

```typescript
import { common } from '@kit.AbilityKit';
import { promptAction } from '@kit.ArkUI';
import { Todo, TodoRepository } from '../model/Todo';

export class TodoDetailResult {
  changed: boolean;

  constructor(changed: boolean) {
    this.changed = changed;
  }
}

@ComponentV2
export struct TodoDetail {
  @Param todoId: string = '';
  @Local title: string = '';
  @Local note: string = '';
  @Local done: boolean = false;
  @Local loading: boolean = true;
  private createdAt: number = Date.now();
  private stack: NavPathStack = new NavPathStack();
  private repo: TodoRepository = new TodoRepository();

  aboutToAppear(): void {
    this.load();
  }

  private async load(): Promise<void> {
    try {
      const context: common.UIAbilityContext = getContext(this) as common.UIAbilityContext;
      await this.repo.init(context);
      const todo: Todo | null = await this.repo.find(this.todoId);
      if (todo !== null) {
        this.title = todo.title;
        this.note = todo.note;
        this.done = todo.done;
        this.createdAt = todo.createdAt;
      }
    } catch (error) {
      promptAction.showToast({ message: '读取详情失败' });
    } finally {
      this.loading = false;
    }
  }

  private async saveAndBack(): Promise<void> {
    const title: string = this.title.trim();
    if (title.length === 0) {
      promptAction.showToast({ message: '标题不能为空' });
      return;
    }
    try {
      const todo: Todo = new Todo(this.todoId, title, this.note.trim(), this.done, this.createdAt);
      await this.repo.update(todo);
      this.stack.pop(new TodoDetailResult(true));
    } catch (error) {
      promptAction.showToast({ message: '保存失败' });
    }
  }

  build() {
    NavDestination() {
      if (this.loading) {
        LoadingProgress().width(32).height(32)
      } else {
        Column({ space: 16 }) {
          TextInput({ text: this.title, placeholder: '标题' })
            .onChange((value: string) => {
              this.title = value;
            })

          TextArea({ text: this.note, placeholder: '备注（可选）' })
            .height(120)
            .onChange((value: string) => {
              this.note = value;
            })

          Row() {
            Text('已完成')
              .fontColor($r('app.color.text_primary'))
            Blank()
            Toggle({ type: ToggleType.Switch, isOn: this.done })
              .onChange((isOn: boolean) => {
                this.done = isOn;
              })
          }
          .width('100%')

          Button('保存并返回')
            .width('100%')
            .onClick(() => {
              this.saveAndBack();
            })
        }
        .padding(16)
        .width('100%')
        .height('100%')
        .backgroundColor($r('app.color.page_bg'))
      }
    }
    .title('待办详情')
    .onReady((context: NavDestinationContext) => {
      this.stack = context.pathStack;
    })
    .onBackPressed(() => {
      if (this.loading) {
        this.stack.pop();
        return true;
      }
      this.saveAndBack();
      return true;
    })
  }
}
```

### 10.10 资源文件

`entry/src/main/resources/base/element/string.json`：

```json
{
  "string": [
    { "name": "module_desc", "value": "待办清单" },
    { "name": "EntryAbility_desc", "value": "待办清单" },
    { "name": "EntryAbility_label", "value": "待办清单" }
  ]
}
```

`AppScope/resources/base/element/string.json`：

```json
{
  "string": [
    { "name": "app_name", "value": "待办清单" }
  ]
}
```

`entry/src/main/resources/base/element/color.json`：

```json
{
  "color": [
    { "name": "start_window_background", "value": "#F5F6F8" },
    { "name": "page_bg", "value": "#F5F6F8" },
    { "name": "card_bg", "value": "#FFFFFF" },
    { "name": "text_primary", "value": "#1A1A1A" },
    { "name": "text_secondary", "value": "#666666" },
    { "name": "text_muted", "value": "#999999" },
    { "name": "danger", "value": "#E84026" }
  ]
}
```

`entry/src/main/resources/dark/element/color.json`：

```json
{
  "color": [
    { "name": "start_window_background", "value": "#121212" },
    { "name": "page_bg", "value": "#121212" },
    { "name": "card_bg", "value": "#1E1E1E" },
    { "name": "text_primary", "value": "#F5F5F5" },
    { "name": "text_secondary", "value": "#B0B0B0" },
    { "name": "text_muted", "value": "#777777" },
    { "name": "danger", "value": "#FF6B5E" }
  ]
}
```

深色模式适配的关键不是监听开关，而是**别硬编码颜色**：所有颜色走 `$r('app.color.xxx')`，系统切深色时会自动命中 `resources/dark`。如果还要在应用内手动切换，用 `ApplicationContext.setColorMode`（API 11 起）或对应 UIContext 能力，具体以官方文档为准。

### 10.11 跑起来

用 DevEco 新建 Empty Ability 工程，把上述文件替换进去，保留模板的 `icon`、`startIcon`、`layered_image` 等媒体资源；`oh-package.json5` 保持模板默认即可。先 Build > Build Hap(s) 确认编译，再 Run 到模拟器或真机。自检：新增一条待办，杀进程重开仍在；点标题进详情，改标题/备注/完成状态，保存返回后列表更新；切系统深色模式，背景和文字颜色变化；删除一条，重开不复活。交付前至少手测空标题、超长标题、删除最后一条、进详情不改直接返回、深色切换、杀进程重开这六个场景，它们比“编译通过”更能暴露状态和持久化问题。
