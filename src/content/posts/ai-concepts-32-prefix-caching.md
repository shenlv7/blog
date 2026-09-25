---
title: "32 | 前缀缓存：共享的 2048 token 系统提示词把 prefill 从 599,534 token 砍到 35,182（省 94.1%），但把时间戳写在提示词开头就归零"
description: "给推理服务算一笔最容易被忽略的账：每来一个请求就重算一遍的那段系统提示词。200 条请求、共享 2048 token 系统提示词 + 4 个 800 token few-shot 模板，总输入 599,534 token，无缓存要 15,588 TFLOP prefill；token 级前缀缓存把实算压到 35,182 token（省 94.13%，等于 564,352 token 的算力白捡）。块哈希 APC 在提示词全是 16/32 倍数时与 token 级完全一致，块=64 掉到 93.09%；多轮对话里前缀永远不对齐块边界，块=16 多算 1.93%、块=32 多算 3.41%、块=64 多算 9.54%。容量实验给出最硬的结论：热前缀集 328 块（5248 token），容量 2000 token 时命中率 0%（比不开缓存还差，白占 1.53 GiB），4000 时 79.59%，8000 时触顶 94.13%——共享前缀的长度就是缓存容量的下限。最后是提示词污染：同一个动态字段放在开头命中率 0.00%、放在末尾 67.96%、放在 user 消息里 94.09%。省的是 prefill，80,000 个 decode step 一个也省不掉。"
pubDate: 2026-09-25
tags: ["AI核心概念", "前缀缓存", "Prefix Caching", "RadixAttention", "vLLM", "SGLang", "LLM推理优化"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 32
slug: ai-concepts-32-prefix-caching
---

# AI核心概念(32)：前缀缓存——把重复的提示词变成免费的提示词

> 推理服务里最贵的 token 从来不是你生成的那批，而是**每来一个请求就从头重算一遍的那 2048 个系统提示词 token**。它们每个字节都一模一样，每个字节都被算过上万遍。前缀缓存（prefix caching）干的事极其朴素：既然 K/V 只由前缀决定，那就别算了——直接从上一次的缓存里接上。

## 🎯 免费的那一段，为什么真的免费

先讲清楚"为什么可以跳"。第 22 篇讲的 KV cache 机制里有一个关键性质：因果注意力 + 自回归生成，第 _i_ 个位置的 K/V 向量只依赖 **前 i 个 token 本身**，跟后面说什么、跟采样温度、跟跟谁并发都无关。

于是同一段 token 序列，在任何一次请求里算出来的 K/V 都是逐比特相同的（同一模型、同一精度、同一 adapter）。既然如此，这段计算就是纯粹的重复劳动。

算一遍账就明白量级：

| | 无缓存 | 有前缀缓存 |
|---|---|---|
| 需要 prefill 的 token | 599,534 | 35,182 |
| 折算算力（2×13B FLOPs/token） | 15,587.88 TFLOP | 914.73 TFLOP |
| 折算时间（假设 3000 tok/s） | 199.84 s | 11.73 s |

**省掉 564,352 token 的 prefill，等于 14,673 TFLOP 白捡。** 而且这份收益是无损的——vLLM 官方文档的原话是 prefix caching "won't change model outputs"，它不改变任何输出 token，只改变这些 token 是怎么被算出来的。

## 🔬 实验一：200 条请求的账是怎么算出来的

工作负载 W1（模拟最常见的"共享系统提示词 + few-shot"服务）：

- 2048 token 系统提示词，**所有 200 条请求完全一样**；
- 4 个 800 token 的 few-shot 模板，每条请求轮换用其中一个；
- 每条请求自己的 user query 80~220 token（只有这一段真正独一无二）。

| 实现 | prefill token | prefill 节省 | 平均命中前缀 |
|---|---|---|---|
| 无缓存 | 599,534 | — | 0 |
| 基数树（token 级精确匹配） | **35,182** | 94.13% | 2821.8 |
| 块哈希 APC（块=16） | 35,182 | 94.13% | 2821.8 |
| 块哈希 APC（块=32） | 35,182 | 94.13% | 2821.8 |
| 块哈希 APC（块=64） | 41,454 | 93.09% | 2790.4 |

注意 35,182 这个数字的构成：它是所有**唯一**前缀 token 之和——200 条请求里每段系统提示词只算一次、每个模板只算一次、每条 query 各算一次。94% 的节省不是靠什么近似，而是因为这 599,534 个 token 里本来就只有 35,182 个是"没见过的"。

块 = 16 和块 = 32 时结果与 token 级完全一致，因为 2048 和 800 都是 16、32 的整数倍——**块粒度只在"不对齐"时才付出代价**。块 = 64 时掉到 93.09%，因为 800 ÷ 64 = 12.5，每个模板的最后 32 个 token 永远凑不满一个块，只能白算。

## 🏗️ 两种主流实现：块哈希 vs 基数树

**vLLM：块哈希链（Automatic Prefix Caching）**

vLLM 的 KV cache 本来就是按固定大小的块管理的（默认 16 token，正是第 31 篇 PagedAttention 那套块表）。前缀缓存做的事是给每个块算一个**链式哈希**：

```
block_hash[i] = H( block_hash[i-1] , block[i] 的 token 序列 , extra )
```

`extra` 是让这个块真正唯一的东西：LoRA ID、多模态输入的哈希、以及多租户隔离用的 cache salt。链式是精髓——只要前一个块不同，后面所有哈希全变，所以"命中"等价于"从头到这里逐 token 完全一致"，不存在"大致相似"这回事。找到哈希就复用对应的物理块，refcount +1（跟第 31 篇同一套共享机制：一次并行采样 8 份的 prompt 只存一份 KV）。

两个必须知道的限制：**只缓存完整块**（前缀长度落在块中间的那几个 token 不给记账）；块被淘汰的规则是 LRU，但 refcount > 0 的块（还在给活跃请求供数）永远不会被踢。

**SGLang：RadixAttention**

SGLang 走另一条路，用一棵压缩基数树（compressed trie）按 token 粒度管理已缓存前缀。它不需要哈希，天然支持任意长度的公共前缀，命中粒度是单个 token；代价是淘汰要按子树为单位、插入时要分裂边。

| | vLLM APC | SGLang RadixAttention |
|---|---|---|
| 数据结构 | 块 → 哈希链 + 字典 | 压缩基数树 |
| 命中粒度 | 块（默认 16 token） | token（任意边界） |
| 部分重叠 | 只能对齐块边界 | 任意边界 |
| 淘汰 | 块级 LRU | 子树级 |
| 默认状态 | vLLM V1 默认开启（`enable_prefix_caching` 默认 True） | 开启 |

vLLM 侧几个现场会用到的开关：`--no-enable-prefix-caching` 关掉它；`--block-size` 调块大小；`--prefix-caching-hash-algo` 选哈希（v0.11 起默认 sha256，想要跨语言可复现选 `sha256_cbor`，追求速度选 `xxhash`）；多租户用 `cache_salt` 硬隔离。指标看 `vllm:prefix_cache_queries` 和 `vllm:prefix_cache_hits` 两个计数器相除，旧版的 `gpu_prefix_cache_hit_rate` 已弃用。

## 🔬 实验二：多轮对话里，对齐损失有多少

块粒度真正的代价在**前缀长度永远不对齐**的场景——多轮对话。W2：共享 2048 token 系统提示词 + 每会话 300 token 人格前缀，之后每轮把"问 + 答"追加进历史（问 60~140、答 90~200 token），8 个会话交错跑 6 轮，共 48 条请求、148,285 token：

| 实现 | prefill token | 命中率 | 相对 token 级多算 |
|---|---|---|---|
| token 级基数树 | 15,129 | 89.80% | — |
| 块哈希 APC（块=16） | 15,421 | 89.60% | +292（1.93%） |
| 块哈希 APC（块=32） | 15,645 | 89.45% | +516（3.41%） |
| 块哈希 APC（块=64） | 16,573 | 88.82% | +1,444（9.54%） |

每轮追加的问答长度是随机的，前缀长度就在块边界上随机游走，平均每次丢掉半个块。块越小丢得越少，但块越小索引里的哈希条目越多、查表开销越大——这就是块大小这个旋钮的全部含义。

## 🔬 实验三：容量不够时的悬崖（本节最重要）

前缀缓存不是免费的：缓存里的块**本身就占着显存**，而且和正在解码的请求抢同一个池子。所以真问题是：池子该留多大？

W1 的块 = 16，缓存容量从 2000 token 扫到 64000 token（LRU 淘汰）：

| 容量(token) | prefill token | 命中率 | 相对无上限 | 容量占用 KV |
|---|---|---|---|---|
| 2,000 | 599,534 | **0.00%** | 17.04x | 1.53 GiB |
| 4,000 | 122,382 | 79.59% | 3.48x | 3.05 GiB |
| 8,000 | 35,182 | 94.13% | 1.00x | 6.10 GiB |
| 16,000 及以上 | 35,182 | 94.13% | 1.00x | 12.21 GiB+ |

容量 2000 token 时命中率**是 0**——比不开缓存还差，因为那 1.53 GiB 白占了显存却一个块都没留住。原因是热前缀集 = 系统 128 块 + 4 个模板 × 50 块 = **328 块 = 5248 token**：容量只有 125 块时，LRU 在插入过程中就把链子开头的系统提示词块淘汰了，**链式哈希只要断在中间，后面全部作废**。

这就是前缀缓存最硬的工程约束：**共享前缀的长度就是缓存容量的下限**。系统提示词 2048 token，缓存至少要留得下 2048 token 的完整前缀链，否则一分钱都省不到。W2（多轮对话）更苛刻，容量要到 16000 token 才追上无上限的 89.60%，因为每个会话的历史都在持续占位。

顺带一个好消息：淘汰只发生在 refcount = 0 的块上，正在跑的请求的 KV 不会被缓存策略抢走。

## 🔬 实验四：一个动态字段放在哪里（0% / 68% / 94%）

我把同一个"每次都不同"的字段（时间戳 / request_id，只差 1 个 token）放在四个位置，其余完全一样，各跑 200 条：

| 变体 | prefill token | 命中率 | 平均命中前缀 |
|---|---|---|---|
| A 动态字段放在提示词**开头** | 600,074 | **0.00%** | 0.0 |
| B 动态字段放在系统提示词**末尾** | 192,098 | 67.96% | 2037.8 |
| C 动态字段放进 **user 消息** | 35,454 | 94.09% | 2821.8 |
| D 提示词两版混跑（灰度/AB） | 35,766 | 94.03% | 2816.6 |

- **A = 0%**：第一个 token 就不一样，整条哈希链、整棵树全废。生产上最常见的自杀方式就是在系统提示词开头写 `当前时间：{{now}}` 或 `用户ID：{{uid}}`。
- **B = 68%**：共享的 2048 token 还是完整命中了，但系统提示词后面的模板、以及后面所有内容全部重算——因为分叉点之后的每一块哈希都变了。
- **C = 94%**：共享内容原封不动，动态字段下沉到 user 消息里，接近理论上限。
- **D = 94.03%**：命中率看不出问题，但缓存要同时留住两套前缀（7296 token，+39% 的前缀占用、多 1.56 GiB KV），两版之间一次都不复用。灰度期间缓存被别人悄悄劈成两半，就是这么来的。

结论一句话：**静态的放最前面，动态的往后推，越靠后越便宜。**

## 💻 可运行 Demo

下面这段是完整实验的核心（本机跑通，约 30 秒，无网络依赖；脚本与输出留在 `~/scripts/blog_daily/32-prefix-caching.py`）。这是 token 级基数树与块哈希 APC 的最小实现：

```python
import hashlib
from collections import OrderedDict

class RadixNode:
    __slots__ = ("children", "edge", "parent", "depth")
    def __init__(self, edge=(), parent=None, depth=0):
        self.children, self.edge, self.parent, self.depth = {}, edge, parent, depth

class RadixTree:
    """压缩前缀树：match() 返回与已缓存序列的最长公共前缀长度（token 数）。"""
    def __init__(self):
        self.root, self.hops, self.cmps = RadixNode(), 0, 0

    def insert(self, seq):
        node, i, n = self.root, 0, len(seq)
        while i < n:
            child = node.children.get(seq[i])
            if child is None:                       # 新开一条边
                node.children[seq[i]] = RadixNode(tuple(seq[i:]), node, node.depth + n - i)
                return
            edge, j = child.edge, 0
            while j < len(edge) and i + j < n and seq[i + j] == edge[j]:
                j += 1                              # 逐 token 比较，边被压缩但比较不能省
            if j == len(edge):                      # 整条边匹配，继续往下走
                i, node = i + j, child
                continue
            mid = RadixNode(edge[:j], node, node.depth + j)   # 在 j 处劈开这条边
            node.children[seq[i]] = mid
            child.edge, child.parent, child.depth = edge[j:], mid, mid.depth + len(edge) - j
            mid.children[child.edge[0]] = child
            if i + j < n:
                mid.children[seq[i + j]] = RadixNode(tuple(seq[i + j:]), mid, mid.depth + n - i - j)
            return

    def match(self, seq):
        node, i, n, best = self.root, 0, len(seq), 0
        while i < n:
            child = node.children.get(seq[i])
            if child is None:
                break
            self.hops += 1
            edge, j = child.edge, 0
            while j < len(edge) and i + j < n and seq[i + j] == edge[j]:
                j += 1
            self.cmps += j
            best, i = best + j, i + j
            if j < len(edge):                       # 在边中间分叉 → 命中到此为止
                break
            node = child
        return best

class BlockCache:
    """块哈希 APC：h_i = H(h_{i-1}, block_i)，只认完整块，LRU 淘汰。"""
    def __init__(self, block_size=16, cap_tokens=None):
        self.bs, self.cap, self.blocks, self.used, self.n_hash = block_size, cap_tokens, OrderedDict(), 0, 0

    def _hash(self, h, blk):
        return hashlib.blake2b(h + b"".join(t.to_bytes(4, "big") for t in blk), digest_size=8).digest()

    def match_len(self, tokens):
        h, n_blk = b"\x00" * 8, 0
        for b in range(len(tokens) // self.bs):      # 非完整块的尾部不记账
            blk = tokens[b * self.bs:(b + 1) * self.bs]
            h = self._hash(h, blk)
            self.n_hash += 1
            if h not in self.blocks:
                break                                # 链断在中间，后面全作废
            n_blk += 1
            self.blocks.move_to_end(h)               # LRU 触摸
        return n_blk * self.bs

    def insert(self, tokens):
        h = b"\x00" * 8
        for b in range(len(tokens) // self.bs):
            h = self._hash(h, tokens[b * self.bs:(b + 1) * self.bs])
            if h not in self.blocks:
                self.blocks[h] = True
                self.used += self.bs
                while self.cap is not None and self.used > self.cap and self.blocks:
                    self.blocks.popitem(last=False)  # 淘汰最久未用的块
                    self.used -= self.bs
            else:
                self.blocks.move_to_end(h)

def prefill_tokens(reqs, engine):
    """统计一批请求实际需要 prefill 的 token 数（命中部分跳过）。"""
    total = 0
    for _, seq in reqs:
        hit = engine.match_len(seq) if isinstance(engine, BlockCache) else engine.match(seq)
        engine.insert(seq)
        total += len(seq) - hit
    return total
```

跑出来的结果就是上面几张表。库里实测：Python 里两种索引单次查询分别是 1075.4 µs（基数树，2998 次 == 比较）和 1939.6 µs（块哈希，187 次 blake2b 哈希）——两者都是 O(命中 token 数) 的线性扫描，差别只在常数（基数树比较一次、块哈希每 16 个 token 要算一次哈希），C++ 实现里这个常数要再小两三个数量级，索引开销相对几十毫秒的 prefill 完全可以忽略。

## 💡 工程细节与陷阱

1. **省的是 prefill，不是 decode。** 假设这 200 条请求各生成 400 token：prefill 省 94.13%，但端到端 token 处理量只省 **83.05%**——80,000 个 decode step 一个也省不掉。所以前缀缓存对"长输入短输出"（RAG、分类、抽取、Agent 每步都带全量历史）收益最大；对长文本生成收益被稀释。
2. **必须逐 token 一致，不是"语义相同"。** 多一个空格、换行不同、JSON 字段顺序变了、模板里少一个 `\n`、换了 tokenizer 版本、换了量化权重——全部归零。跨模型之间更是绝对不共享（KV 是权重相关的量）。
3. **多租户要显式隔离。** vLLM 的块哈希里带 `cache_salt`，不同 salt 的请求永不共享块。历史上哈希碰撞曾是理论风险，v0.11 起默认换成 sha256。另外"命中/未命中的时延差"本身就是可观察量，严格的隔离场景（比如同一台机器上跑竞争方流量）要按安全边界处理，而不是按性能开关处理。
4. **它和 PagedAttention 是同一套机制的两面。** 第 31 篇里"并行采样 8 份只存一份 KV"（6.40x）用的是块表共享；前缀缓存只是把这种共享从"同一请求内部"扩展到"跨请求"，判定依据从"我指定了同一块"变成"内容哈希相同"。
5. **别信"命中率 100%"。** 看 `prefix_cache_queries / prefix_cache_hits` 的对子。命中率是过程指标，真正要盯的是 prefill 实算 token 和 TTFT 的变化。
6. **缓存与在线请求抢显存。** 池子小了命中率崩（实验三的 2000 token → 0%），池子大了挤占并发。经验法则：先保证"共享前缀那条完整链" + 并发请求各自的 KV 都放得下，再谈给缓存留多少。

## 📌 小结

| 结论 | 数字 |
|---|---|
| 共享系统提示词的收益 | prefill 599,534 → 35,182 token，省 94.13%（≈14,673 TFLOP） |
| 块粒度在完美对齐时无损失 | 块=16/32 与 token 级完全一致 |
| 块粒度在不对齐时有损失 | 多轮对话：块=16 +1.93%、块=32 +3.41%、块=64 +9.54% |
| 容量必须覆盖共享前缀全长 | 热前缀集 328 块/5248 token；容量 2000 时命中率 0% |
| 动态字段的位置决定一切 | 开头 0.00% / 系统末尾 67.96% / user 消息里 94.09% |
| 省的是 prefill | prefill −94.13%，端到端 token −83.05% |

最后一句：前缀缓存是这套推理优化系列里唯一"什么都不用改就能白拿"的优化——只要你把提示词组织对了。它的失效方式也很少：前缀变了、容量不够了、或者你以为缓存了其实差一个空格。前两个可以测，第三个只有靠实测才看得见。

**下一篇预告**：既然 KV cache 可以跨请求共享，那能不能干脆**跨请求复用整段计算**？从连续批处理说到有状态推理（stateful inference）与对话缓存——以及为什么"共享得越多，调度就越像操作系统"。
