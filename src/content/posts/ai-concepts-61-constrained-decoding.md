---
title: "约束解码把 JSON 合法率从 0 拉到 100%，严格合规率还是 0——它只买得到语法"
description: "250 条实测：不约束时 0/250 语法合法，加上 JSON 下推自动机掩码后 250/250 全合法，而键集合、枚举、取值几乎全错（重复键 38%、键集合错 51%）。掩码的力气全花在结构字符上：冒号前只剩 1 个 token（0.4%），字符串内部 62% 的 token 都合法。"
pubDate: 2026-10-24
tags: ["结构化输出", "约束解码", "JSON", "推理工程", "下推自动机"]
difficulty: "advanced"
series: "ai-concepts"
seriesOrder: 61
slug: "ai-concepts-61-constrained-decoding"
---

> 同一个模型、同一批语料、同一套采样。
> 不加掩码：0/250 条是合法 JSON。
> 加了下推自动机掩码：250/250 全部合法。
> 而"这条记录对不对"——严格合规率 0%。

**读这篇要会**：softmax 与 logits 的关系 / JSON 的语法结构（对象、数组、字符串、数字）/ 有限状态机与正则表达式 / Python 字典树（trie）为什么能省掉大半比较。不需要 GPU，不需要 transformers，不需要调用任何 API。

## 🎯 「请只输出 JSON」为什么治不好

所有要调工具、要写数据库、要给前端喂数据的系统，都会遇到同一件事：模型偶尔吐出一段不可解析的东西。于是 prompt 里加一句 `只输出合法的 JSON，不要任何解释`，再加一个正则把 ```json 围栏剥掉，再写一层 `try/except` 重试三次。

这套组合拳能跑，但它的本质是**概率性修补**。一个 200 token 的回答，只要中途有一个字符走进死路——引号没闭合、逗号漏掉、数字后面跟了字——整条记录报废，你付的那次推理费全白花。重试当然可以救，代价是尾延迟和成本翻倍。

约束解码（constrained decoding，也叫 grammar-constrained decoding / logit masking）把这件事从"概率"变成"零容错"：**每生成一个 token 之前，先算一遍"在当前位置，词表里哪些 token 还能让整段文本落在合法集合里"，其余位置的概率直接置零。**

$$\tilde p_\theta(v\mid y_{<t}) = \frac{p_\theta(v\mid y_{<t})\,\mathbb 1[\delta(q_t, v)\neq\bot]}{\sum_{v'} p_\theta(v'\mid y_{<t})\,\mathbb 1[\delta(q_t, v')\neq\bot]}$$

读法：$q_t$ 是当前语法状态，$\delta(q_t,v)$ 是"从 $q_t$ 出发吃下 token $v$ 之后的状态"，$\bot$ 表示这条边不存在。被置零的位置分母里没有它们，所以剩下的概率自动重新归一化——**掩码不是把概率"减掉"，而是换了一个条件分布**。注意那个 $\delta$ 是逐字符推进的：

$$\delta(q_t, v) = \delta\bigl(\cdots\delta(\delta(q_t, v_1), v_2)\cdots, v_{|v|}\bigr)$$

这一步是全部工程复杂度的来源：token 不等于语法单元，一个 token 可能横跨"数字结尾 + 逗号 + 引号"三个状态。

JSON 还有个容易被忽略的性质：**带嵌套的 JSON 不是正则语言**。`{"a":{"b":{"c":1}}}` 的括号配对需要计数，正则表达式表达不了无限深度。工程上有两条路——把嵌套深度截断成正则（有的库把它编译成巨大的 DFA），或者老老实实上一个**下推自动机**（栈 + 状态）。我这个实验走第二条，因为栈才能让我把"现在在第几层对象里"变成可以测量的东西。

## 🔬 装置：一个能精确读出"合法集合"的小世界

真实 LLM 里拿不到干净的对照：你不知道换一个掩码它会不会自己想对。所以我把问题缩小到一个可以被逐字符测量的场合：

- **被测模型**：从 4000 条合成 JSON 记录里统计出来的**字符级 3-gram**（带 backoff 与 α 平滑），字符表 37 个。它局部语法学得很像样，长程结构基本没有——正好把"掩码能补什么、不能补什么"分开。
- **语料 schema**：`{"id": int, "status": "ok"|"error"|"pending", "score": 0..1 两位小数, "tags": [1-3 个已知标签]}`
- **词表**：在这批语料上跑 194 次 BPE 合并，得到 230 个 token（字符 + 合并片段）。用来测"token 级掩码"和"字符级掩码"的差别。
- **语法**：我手写的 JSON 下推自动机，状态是 `(模式, 括号栈, 附加信息)`，共 11 种模式（`value / key / key2 / arrval / arrval2 / instr / colon / num / lit / after / done`）。`key2` 和 `arrval2` 的存在是因为**逗号之后不允许直接右括号**——尾随逗号必须被拒，这是"看起来对"和"真的对"的区别。
- **评测**：char 级与 token 级两种粒度、三种采样温度（0.35 / 0.5 / 0.75），各 250 条，共 1500 条生成。

## 🏗️ 结果：语法是 100%，语义是 0%

先看最干净的三个对照：

| 生成方式 | JSON 语法合法 | 严格 schema 合规 |
|---|---|---|
| 无约束（200 字符截断） | **0 / 250** | 0 |
| 无约束 + 先知截断（在每个 `}` 处试解析，取能解析的最短前缀） | 103 / 250（41.2%） | 未单独统计 |
| 约束解码（char / token 级，三种温度） | **250 / 250 × 6 组** | 0 – 1.2% |

中间那一行是我加的**公平性对照**，也是这张表里最值钱的数字：不约束的时候有 41.2% 的生成其实"内嵌着"一段合法 JSON，只是模型不知道在哪停。也就是说，掩码真正干掉的不只是"写错"，还有"不知道何时收尾"——而后者在生产里同样致命（多吐一个字符，下游 `json.loads` 就炸）。

然后是坏消息。250 条全部语法合法，具体死在哪：

| 温度 | 粒度 | 语法合法 | 严格合规 | 重复键 | 键集合不对 | 取值越界 |
|---|---|---|---|---|---|---|
| 0.35 | char | 250/250 | 0 | 114 | 128 | 8 |
| 0.35 | token | 250/250 | 1 | 94 | 148 | 7 |
| 0.50 | char | 250/250 | 3 | 99 | 140 | 8 |
| 0.75 | char | 250/250 | 1 | 106 | 136 | 7 |

两种典型死法长这样：

```
{"id":733,"tags":"ok","score":0.96,"score":0.72,"score":0.55,"tags":["graph"...
{"id":362,"score":0.61,"status":["nlp"]}
```

第一条是**重复键**——JSON 语法完全允许，RFC 8259 §4 只说名字"SHOULD be unique"（注意是 SHOULD 不是 MUST），多数解析器取最后一个，于是你拿到 `tags: ["graph"]`，前一个 `"ok"` 悄悄消失。第二条把 `status` 写成数组，键序也乱了：语法合法，schema 全错。

这就是标题里那个反差的来源：**语法是可达集，语义不是。** 掩码只回答"下一个字符能不能落在合法前缀上"，它不知道"这段 JSON 该有什么字段"。

## 🤏 掩码的力气花在哪：一张按模式拆开的表

把一条典型生成逐步拆开，看每个位置上还有多少东西可选（token 级，词表 230）：

| 当前模式 | 步数 | 允许字符 | 允许 token（占词表） |
|---|---|---|---|
| `colon`（等冒号） | 2 | 1 | **1.0 / 230（0.4%）** |
| `key` / `key2`（等键） | 2 | 1 – 2 | 6 – 7 / 230（2.6 – 3.0%） |
| `after`（等逗号或右括号） | 5 | 2 | 16.8 / 230（7.3%） |
| `value` / `arrval`（等值） | 4 | 17 – 18 | 100 – 112 / 230（44 – 49%） |
| `num`（数字中） | 5 | 14 | 144 / 230（62.6%） |
| `instr`（字符串内部） | 33 | 96 | 143.4 / 230（62.4%） |

这张表解释了上面所有失败：**33 步 51% 的时间待在字符串里，而字符串内部几乎不剪枝**（除引号和反斜杠，什么都能进）。`{"status":"` 之后模型可以在 `ok` 和任意 3 个字符之间自由选择，掩码管不了——`status` 那个枚举约束在语法层根本不存在。

还有一条容易被忽略的反直觉数据：**约束解码只改写了 1.6% 的预测。** 我逐步比对了"模型自己的 top-1"和"掩码允许的集合"，12957 步里只有 212 步（1.6%）本该输出非法字符。3-gram 在局部语法上其实很准。

那为什么无约束还是 0/250？因为 1.6% 是**每步**的失手率，一条记录平均 65 步：$1-(1-0.016)^{65} \approx 65\%$，每条生成有 65% 的概率至少踩一次雷，而踩一次就整条报废。**约束解码换来的不是"更准的预测"，是把一个 65% 的单条失败概率压到 0。** 这个视角比"提升 xx%"有用得多：它说明掩码的价值随输出长度指数增长——短回答看不出差别，长 JSON、长 SQL、长 function call 里它是刚需。

## 💻 核心代码

完整脚本（含 BPE、温度扫描、成本测量）已发到仓库：[demos/ai-concepts-61-constrained-decoding.py](https://github.com/shenlv7/blog/blob/main/demos/ai-concepts-61-constrained-decoding.py)。下面是决定成败的三段，删掉了统计与打印。

```python
# 1) 语法状态：不可变元组，才能被缓存和回溯
#    state = (mode, stack, extra)，stack 里是 'o'（对象）/ 'a'（数组）
def allowed(st):
    mode, stack, extra = st
    if mode in ('value', 'arrval', 'arrval2'):
        a = {'{', '[', '"', '-'} | DIG | {'t', 'f', 'n'}
        if mode == 'arrval':
            a |= {']'}
        return a
    if mode == 'key':   return {'"', '}'}
    if mode == 'key2':  return {'"'}          # 逗号后不许右括号：拒尾随逗号
    if mode == 'colon': return {':'}
    if mode == 'instr':
        role, esc = extra
        return set('"\\/bfnrt') if esc else (set('\x22\\') | ESC_OK)
    if mode == 'num':
        a = set(DIG)
        if '.' not in extra: a |= {'.'}
        if extra != '-' and not extra.endswith('.'): a |= {',', '}', ']'}
        return a
    if mode == 'after':
        return {',', '}'} if stack[-1] == 'o' else {',', ']'}

def step(st, c):
    mode, stack, extra = st
    if mode in ('value', 'arrval', 'arrval2'):
        if c == '{':  return ('key', stack + ('o',), None)
        if c == '[':  return ('arrval', stack + ('a',), None)
        if c == '"':  return ('instr', stack, ('val', False))
        if c in DIG or c == '-': return ('num', stack, c)
        if c in 'tfn': return ('lit', stack, {'t': 'rue', 'f': 'alse', 'n': 'ull'}[c])
        return None
    if mode == 'instr':
        role, esc = extra
        if esc:  return ('instr', stack, (role, False)) if c in '"\\/bfnrt' else None
        if c == '\\': return ('instr', stack, (role, True))
        if c == '"':  return ('after', stack, None) if role == 'val' else ('colon', stack, None)
        return ('instr', stack, (role, False)) if 32 <= ord(c) < 127 else None
    if mode == 'num':
        if c in DIG: return ('num', stack, extra + c)
        if c == '.' and '.' not in extra: return ('num', stack, extra + '.')
        if c in ',}]' and extra != '-' and not extra.endswith('.'):
            return step(('after', stack, None), c)    # 数字收尾，分隔符交给 after
        return None
    if mode == 'after':
        if c == ',':
            return ('key2' if stack[-1] == 'o' else 'arrval2', stack, None)
        if c == '}' and stack[-1] == 'o' or c == ']' and stack[-1] == 'a':
            ns = stack[:-1]
            return ('after', ns, None) if ns else ('done', ns, None)
        return None

# 2) 用词表前缀树枚举合法 token：节点数 ∝ 合法前缀数，不是 |V|
def allowed_tokens(trie, st):
    out = []
    def dfs(node, s):
        if '#' in node: out.append(node['#'])
        A = allowed(s)
        for ch, child in node.items():
            if ch != '#' and ch in A:
                ns = step(s, ch)
                if ns is not None: dfs(child, ns)
    dfs(trie, st)
    return out

# 3) 生成循环：唯一的改动就是"候选集从词表换成合法集"
def gen_constrained(model, rng, trie, max_steps=400, temp=0.75):
    ctx, st, out = '\x02' * model.order, s_value(), []
    while len(out) < max_steps and st[0] != 'done':
        cand = allowed_tokens(trie, st)
        if not cand: return ''.join(out), False
        p, _ = model.probs(ctx)
        w = np.array([np.prod([p.get(ch, 1e-6) for ch in t[:3]]) for t in cand]) ** (1 / temp)
        w /= w.sum()
        tok = cand[min(int(np.searchsorted(np.cumsum(w), rng.random())), len(cand) - 1)]
        for ch in tok:
            st = step(st, ch)
        out.append(tok)
        ctx = (ctx + tok)[-model.order:]
    return ''.join(out), st[0] == 'done'
```

这套东西跑 250 条、三种温度、两种粒度大约两分钟（本机 ARM，无 GPU）。

## 🩹 我第一版为什么是废的

**第一版：我以为前缀树一定更快。** 教科书写法是"用 trie 共享 token 前缀，避免重复推进状态"。实测（同一段 28 步生成）：|V|=230 时前缀树 57 ms，朴素逐 token 模拟 53 ms——**前缀树更慢 0.9x**；我把词表扩到 18658 个合成 token 再测，前缀树 9979 ms、朴素 7286 ms，**还是慢（0.7x）**。原因是 Python 里每访问一个字典节点都要建一个 `allowed(state)` 集合，常数开销压过了省下的字符转移。真正的库（Rust / C++）靠的是三件事：状态转移预编译成整数数组、掩码用位图、缓存 `(state)` 的合法集。**"渐近更优"在脚本语言里可以被常数项吃掉，这话在别的地方也一样。** 所以表里我保留了这个负结果。

**第二版之前的坑更致命：我的对照是假的。** 最初我直接比"无约束 0/250 vs 约束 250/250"，看起来漂亮得可疑——因为无约束那条路我截断在 200 字符，它连"该在哪停"都不知道，输赢有一半来自这个设定。加上"先知截断"对照（41.2%）才说得清：掩码真正买的是「零容错 + 知道何时收尾」，不是「模型变聪明」。**任何"约束组 vs 无约束组"的实验，都得先让对照组的失败原因可分解，否则你测的是自己的截断策略。**

## 🚧 这篇不能证明什么

- **不是真实 LLM 实测。** 被测模型是从合成 JSON 语料统计出的字符级 3-gram，参数量与训练目标都和 Transformer 无关。真实 LLM 在 `{"status":"` 之后大概率自己就会写 `ok`，所以真实的"严格合规率"远高于我这里那个 0–1.2%——那 0% 说明的是"小模型不懂 schema"，不是"掩码没用"。
- **掩码本身的合法性保证是可迁移的**：状态机 + 逐 token δ 推进 + 置零重归一化，这部分与模型无关，换任何 LLM 都成立。1.6% 的改写率、65% 的单条失败概率、按模式的剪枝比例都是这台 3-gram 上的数字，**别当成通用常数**。
- **我没做 token 对齐的死局测试。** 我的词表保证字符表全覆盖，所以每步至少有一个单字符 token 可选，从不卡死（实测"未产出完整值" 0 次）。真实词表里如果某个语法位置的可选字符没有任何 token 以它开头，你就需要回退或 token healing——这条我没测。
- **成本只测了掩码枚举，没测推理。** 真实部署里掩码是每步 O(|合法前缀节点|)，相对一次 forward 通常很小，但在超大 schema（几百个字段）上会变成瓶颈，这不是我这台机器能给出的数。
- **schema 层没做。** 我没把"键集合 / 枚举 / 数值范围"编进自动机。这是下一步，也是生产中真正需要的那一半。

## 💡 收尾

约束解码的定位，一句话：**它是把"语法正确"从概率问题变成工程保证的那一层，也仅仅只是那一层。**

所以正确的用法是三层各司其职：

1. **语法**交给掩码（或退一步，交给 `json.loads` + 重试）；
2. **schema** 交给结构化校验器（Pydantic / JSON Schema），而且**必须仍在校验**——重复键这种坑，语法层永远不会报，用 `object_pairs_hook` 抓一下才看得见；
3. **语义**交给业务断言（这个 `id` 在库里存在吗？这个 `score` 合理吗？）。

把第 1 层当成第 2、3 层来信任，就是系统性事故的来源。而知道"掩码在字符串里 62% 的时间什么都不管"，你就不会再问"为什么我加了 constrained decoding，模型还是给我编了个不存在的 `status`"。

下一篇（62）往上走一层：把 schema 本身编译成语法——枚举、必填、顺序到底能被吃掉多少，剩下的又必须回到运行时代码里。
