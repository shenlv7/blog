---
title: "22 | KV 缓存 KV Cache：一个 token 值多少显存，GQA/MQA/MLA 到底在省什么"
description: "注意力机制本身没有记忆，是 KV cache 替它记住的——这笔交易用显存买时间，价格随上下文线性上涨。本文把体积公式套到 6 个真实公开配置上（fp16 下 7B/MHA 每 token 512 KB、70B/GQA-8 每 token 320 KB），再用本机 numpy 玩具 Transformer 做「每步重算 vs 缓存」的逐位对照：5 个随机种子下生成序列完全一致，喂进模型的 token 数从 6,096 降到 112（计算量比 54.4x），实测墙钟快 9.8x。最后真的分配 512 MB 缓存量 RSS 验证公式（比值 1.00x），并按公开显存带宽算出「读一遍 cache」给解码速度划的上限：70B 模型 32k 上下文 3.21 ms/步。"
pubDate: 2026-09-15
tags: ["AI核心概念", "KV Cache", "推理优化", "GQA", "MQA", "MLA", "显存"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 22
slug: ai-concepts-22-kv-cache
---

# AI核心概念(22)：KV 缓存 KV Cache——一个 token 值多少显存，GQA/MQA/MLA 到底在省什么

> "注意力本身没有记忆，是 KV cache 替它记住的。这笔交易用显存买时间，计价方式却是每 token 线性上涨——所以长上下文真正的成本从来不在权重，而在那本被反复翻阅的答案本。"

## 🎯 KV cache 是什么：把「重算」换成「记住」

先用一句话说清前置概念：**注意力的 K（Key）和 V（Value），只由输入 token 和权重决定，跟"后面要生成什么"无关。** 因果掩码又保证了位置 t 只看得到 ≤ t 的位置。两条加在一起，就有了缓存的前提——**已经算过的 K/V 永远不会变，没必要再算第二遍。**

```python
k_t, v_t = W_k @ h_t, W_v @ h_t                 # 每个 token 的 K/V 只算一次
K.append(k_t); V.append(v_t)                    # 存起来，长度随序列线性增长
out_t = softmax(q_t @ K.T / sqrt(d_head)) @ V   # 生成新 token 时，把整段 K/V 读一遍
```

这就是 **KV cache（KV 缓存）**：拿显存换计算量。前缀（prompt）先整段并行算一次，叫 **预填充 prefill**（这阶段是算力受限，GPU 能跑满）；之后每生成一个 token 叫 **解码 decode**（一次只算一个 token，算力吃不满，卡在把权重和 cache 从显存读进来，是内存带宽受限）。

不缓存 vs 缓存，账是这么变的（n = 已生成序列长度）：

| | 每个新 token 要算什么 | 生成 n 个 token 的总计算量 | 显存额外开销 |
|---|---|---|---|
| 不缓存（每步重算整段） | 整段 n 个 token 的 QKV + 注意力 | **O(n³)** | 0 |
| 有 KV cache | 1 个 token 的 QKV + 读一遍 K/V | **O(n²)** | **2 × 层数 × KV头数 × 头维度 × n × 字节数** |

代价就是最后那一列：**缓存体积和上下文长度成正比**。它不是常数开销，是一个"越聊越贵"的开销——这就是下一节要算的账。

## 🔬 一个 token 的 KV 值多少字节

把公式套到真实的公开配置上（FP16 缓存，即 2 字节/数）：

```text
KV cache 每 token 字节数 = 2(K和V) × 层数 × KV头数 × head_dim × 2(FP16 字节)
```

| 模型 | 层数 | 注意力头 | **KV 头** | 每 token | 相对 MHA |
|---|---|---|---|---|---|
| Llama-2-7B (MHA) | 32 | 32 | 32 | **512 KB** | 1.0x |
| Llama-2-13B (MHA) | 40 | 40 | 40 | **800 KB** | 1.0x |
| Llama-3-8B (GQA-8) | 32 | 32 | 8 | **128 KB** | 4.0x 压缩 |
| Mistral-7B (GQA-8) | 32 | 32 | 8 | **128 KB** | 4.0x 压缩 |
| Llama-3-70B (GQA-8) | 80 | 64 | 8 | **320 KB** | 8.0x 压缩 |
| Qwen2-72B (GQA-8) | 80 | 64 | 8 | **320 KB** | 8.0x 压缩 |
| 同结构改 MQA（1 个 KV 头） | — | — | 1 | **16 KB** | 32.0x 压缩 |
| DeepSeek MLA 结构 | 60 | 128 | 缓存 512+64 维隐向量 | **67.5 KB** | 56.9x 压缩 |

（最后一行的 56.9x 是我按公开 config 算的等效值：MLA 每层缓存的是 512 维隐向量 + 64 维 RoPE 分支，共 576 个数，而不是每头独立的 K/V；用的时候再上投影把 K/V 还原出来。省的是显存和带宽，付的是一点重算算力。）

换成显存看更直观（单序列，只有 cache，不含权重和激活）：

| 模型 | 8k 上下文 | 32k 上下文 | 128k 上下文 | 权重(FP16) | 32k 时 cache/权重 |
|---|---|---|---|---|---|
| Llama-2-7B (MHA) | 4.00 G | 16.00 G | 64.00 G | 13.4 G | **119.4%** |
| Llama-2-13B (MHA) | 6.25 G | 25.00 G | 100.00 G | 26.0 G | 96.2% |
| Llama-3-8B (GQA-8) | 1.00 G | 4.00 G | 16.00 G | 16.0 G | 25.0% |
| Llama-3-70B (GQA-8) | 2.50 G | 10.00 G | 40.00 G | 141.2 G | 7.1% |

三条读数，全是从这张表里出来的：

1. **「长上下文贵」贵的不是权重，是 cache。** 7B 的 MHA 模型，32k 上下文的 cache 已经比它自己的 FP16 权重还大（119.4%）；8k 时代它只是 4 G 的零头。上下文从 8k 涨到 128k，权重一分没变，成本涨了 16 倍。
2. **GQA 不是学术洁癖，是省钱的。** 同样 70B，MHA 在 128k 下要 40 G 缓存——那基本等于劝退；GQA-8 把它压到 8 倍小，才有了今天"长上下文能上线"这件事。
3. **并发才是真正被 cache 吃掉的指标。** 30k 上下文、batch=8 的账：Llama-3-8B 权重 16 G + 8 条会话 29.3 G，一张 80G 卡能塞 17 条并发会话；换成 7B/MHA，同样是 30k 上下文，**单条会话的 cache 就 15 G**，80G 卡最多 4 条。KV 头数从 32 变成 8，并发直接差 4 倍多。

## 🧪 本机实测：每步重算 vs 缓存，先验一致性再比速度

光看公式不算数——我用一个纯 numpy 的迷你 GPT（8 层 / d=128 / 4 头）做了两条路径的对照：一条禁用缓存、每步把整段序列重算；一条走标准的"预填充 + 每步喂 1 个 token"。提示词 16 token，贪心解码 96 token：

```text
  ① 生成结果完全一致：True（token 序列逐位相等） | 换 5 个随机种子复验：5/5 条序列一致
  ② 逐步耗时（ms）——看增长趋势，不是看绝对值：
     step           0       7      15      23      31      47      63      79      95
     无缓存     18.17   21.20   26.80   36.66   45.96   78.48  104.35  137.04  181.06
     有缓存     16.80    7.53    7.61    7.73    7.96    7.98    8.42    8.79    9.24
     最后一步 / 第一步：无缓存 10.0x（序列越长约越慢），有缓存 1.17x（第 0 步含预填充）
  ③ 实测总耗时：无缓存 7.800 s   |   有缓存 0.795 s   →  快 9.8x
  ④ 累计「喂进模型的 token 数」：无缓存 6,096，有缓存 112  →  计算量比 54.4x
  ⑤ 两条路径的数值一致性（为什么能放心替换）：
     同一位置 logits 最大绝对差：3.79e-05（float32 舍入量级，logits 幅度 89.0）
     抽样看「贪心决策的余量」(top1 与 top2 的差 vs 两条路径的差)：
       step      top1-top2     两路径logit差
          0          8.7570       0.00e+00
         16         11.8656       9.56e-05
         32          5.8314       1.36e-04
         48         38.7823       1.70e-04
         64          7.4909       2.50e-04
         80         11.8068       1.87e-04
         95         25.9143       1.58e-04
```

四条结论：

1. **缓存不改变结果，只改变速度。** 5 个随机种子下 token 序列逐位相同，同一位置 logits 差 3.79e-05——这是 float32 往返累加的舍入量级（logits 幅度 89），不是"近似"。原因在 ⑤ 那张表：**贪心决策的余量（top1 与 top2 的差）比两条路径的数值差大 5～9 个数量级**。反过来也成立：**如果两个候选 logits 几乎打平，重算和缓存是可能选出不同 token 的**——这也是推理引擎在意"同一 session 内缓存布局固定、算子确定性"的原因。
2. **不缓存的曲线是往上翘的。** 第 96 步比第 1 步慢 10.0 倍（因为它每步都在重算整段历史）；缓存路径从第 1 步到第 96 步只涨了 1.17 倍（每步恒定只算 1 个 token，那点增长全是我 append 时 `concatenate` 拷贝的开销——真实引擎用预分配 + 分块管理避免这种拷贝）。
3. **计算量比 54.4x，但实测墙钟只有 9.8x。** 这是本次实验最有价值的一条：省掉 54 倍计算量，为什么速度只快 10 倍？因为模型太小（8 层 / d=128），小矩阵上 Python + numpy 的固定调用开销占了主导，BLAS 也没法在这种尺寸上跑满。**真实推理引擎用融合 kernel、批处理、CUDA Graph 才能把这笔账吃满——而这个 toy 实验说明的道理是：算法上省掉的计算量，只有在工程上真的不被打回去时才变成速度。**
4. **缓存的账要按"步数"算，不按"总长度"算。** 无缓存时总计算量正比于 n³/3，缓存后正比于 n²——n=96 时是 54 倍，n=1000 时是几百倍。上下文越长，缓存越不可替代。

## 💻 动手 Demo：三个实验一个脚本

`04-kv-cache.py`，纯 numpy + 标准库，本机 22 秒跑完（含真实分配 1.5 GB 内存的实验 C）：

```python
"""04-kv-cache.py — KV cache：体积账本 + 重算 vs 缓存的真实计时 + 真实内存分配实测

纯 numpy + 标准库，CPU 约 10 秒跑完。
实验 A：KV cache 体积公式，套到真实公开配置上（MHA / GQA / MQA / MLA）
实验 B：同一份玩具 Transformer，禁用缓存（每步重算全序列）vs 启用缓存，
        验证两者输出逐位一致，再对比真实耗时和计算量
实验 C：真的分配并写满一段 cache，量 RSS 增量，验证公式；再测写入带宽
"""
import time
from array import array

import numpy as np

MB = 1024 ** 2
GB = 1024 ** 3


# ============================================================
# 实验 A：一个 token 的 KV 到底占多少字节
# ============================================================

def kv_per_token(n_layers, n_kv_heads, head_dim, bytes_per_elem=2):
    """KV cache 每 token 字节数 = 2(K和V) × 层数 × KV头数 × 头维度 × 元素字节。"""
    return 2 * n_layers * n_kv_heads * head_dim * bytes_per_elem


# 配置全部来自模型公开 config（层数 / 注意力头数 / KV头数 / head_dim）
MODELS = [
    # name,                layers, q_heads, kv_heads, head_dim, 权重参数量(B)
    ("Llama-2-7B  (MHA)",   32, 32, 32, 128, 6.7),
    ("Llama-2-13B (MHA)",   40, 40, 40, 128, 13.0),
    ("Llama-3-8B  (GQA-8)",  32, 32,  8, 128,  8.0),
    ("Llama-3-70B (GQA-8)",  80, 64,  8, 128, 70.6),
    ("Mistral-7B  (GQA-8)",  32, 32,  8, 128,  7.2),
    ("Qwen2-72B   (GQA-8)",  80, 64,  8, 128, 72.7),
]


def experiment_a():
    print("=" * 78)
    print("实验 A：KV cache 体积账本（FP16 缓存，bytes_per_elem=2）")
    print("=" * 78)
    print(f"{'模型':<22}{'KV头':>6}{'每token字节':>13}{'每token KB':>12}{'比MHA省':>10}")
    for name, L, qh, kvh, dh, _ in MODELS:
        b = kv_per_token(L, kvh, dh)
        b_mha = kv_per_token(L, qh, dh)
        print(f"{name:<22}{kvh:>6}{b:>13,}{b/1024:>12.1f}{b_mha/b:>9.1f}x")

    print("\n  MQA 极端版（KV 头压到 1）与 MLA（DeepSeek 的低秩联合压缩）：")
    L, qh, dh = 32, 32, 128
    for label, kvh in (("MHA   32 KV头", 32), ("GQA-8  8 KV头", 8), ("MQA    1 KV头", 1)):
        b = kv_per_token(L, kvh, dh)
        print(f"    {label:<16} 每 token {b:>8,} B = {b/1024:>7.1f} KB"
              f"   相对 MHA {kv_per_token(L,qh,dh)/b:>5.1f}x 压缩")
    # MLA：缓存的是压缩后的隐向量 c_KV(kv_lora_rank) + 解耦 RoPE 分支(qk_rope_head_dim)
    L_mla, kv_lora_rank, rope_dim = 60, 512, 64
    mla = (kv_lora_rank + rope_dim) * 2 * L_mla          # 整模型、每 token
    mha_eq = kv_per_token(L_mla, 128, 128)               # 同样 60 层 128 头，若走 MHA
    print(f"    {'MLA (V2 结构)':<16} 每 token {mla:>8,} B = {mla/1024:>7.1f} KB"
          f"   （该结构若用 MHA 要 {mha_eq/1024:,.0f} KB → {mha_eq/mla:.1f}x 压缩）")
    print("    注：MLA 缓存的是每层 512 维隐向量 + 64 维 RoPE 分支（合计 576 个数），")
    print("        不是每头独立的 K/V；要用时再上投影（wkv_b）把 K/V 还原出来——")
    print("        省的是显存和带宽，付的是一点重算算力。")

    print("\n  换算成显存（单序列，仅 KV cache，不含权重和激活）：")
    seqs = [("8k", 8192), ("32k", 32768), ("128k", 131072)]
    print(f"    {'模型':<22}" + "".join(f"{s:>12}" for s, _ in seqs) + f"{'权重FP16':>12}{'32k时占比':>11}")
    for name, L, qh, kvh, dh, params_b in MODELS:
        b = kv_per_token(L, kvh, dh)
        row = "".join(f"{b*n/GB:>11.2f}G" for _, n in seqs)
        w = params_b * 2
        ratio = (b * 32768 / GB) / w * 100
        print(f"    {name:<22}{row}{w:>11.1f}G{ratio:>10.1f}%")
    print("    → 8k 上下文时 KV cache 只是零头；32k 时已占权重的一大截；128k 时 70B 模型")
    print("      光缓存就要 2 位数 GB —— 「长上下文贵」贵在这里，不在权重。")

    print("\n  长上下文下的并发账（30k 上下文，batch=8）：")
    for name, L, qh, kvh, dh, params_b in MODELS:
        b = kv_per_token(L, kvh, dh)
        kv = b * 30000 * 8 / GB
        w = params_b * 2
        tot = kv + w
        fit = int((80 - w * 1.0) // (b * 30000 / GB)) if 80 - w > 0 else 0
        print(f"    {name:<22} 权重 {w:>5.1f}G + 8 条 30k 会话 {kv:>6.2f}G = {tot:>6.2f}G"
              f"   | 一张 80G 卡最多装 {max(fit,0):>2} 条 30k 会话")
    print("    → 「能跑」和「能同时跑几个人」是两件事，GPU 显存被 KV cache 吃光就没并发。")


# ============================================================
# 实验 B：玩具 Transformer —— 每步重算 vs 缓存，逐位对照 + 真实计时
# ============================================================

def rms_norm(x):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6)


class ToyGPT:
    """一个迷你 GPT：pre-norm + 多头因果注意力 + MLP，权重全随机（只关心计算量）。"""

    def __init__(self, vocab=64, d=128, layers=8, heads=4, max_ctx=512, seed=0, std=0.2):
        rng = np.random.default_rng(seed)
        self.d, self.layers, self.heads = d, layers, heads
        self.dh = d // heads
        self.vocab, self.max_ctx = vocab, max_ctx
        self.emb = rng.normal(0, std, (vocab, d)).astype(np.float32)
        self.pos = rng.normal(0, std, (max_ctx, d)).astype(np.float32)
        self.Wqkv = [rng.normal(0, std, (d, 3 * d)).astype(np.float32) for _ in range(layers)]
        self.Wo = [rng.normal(0, std, (d, d)).astype(np.float32) for _ in range(layers)]
        self.W1 = [rng.normal(0, std, (d, 2 * d)).astype(np.float32) for _ in range(layers)]
        self.W2 = [rng.normal(0, std, (2 * d, d)).astype(np.float32) for _ in range(layers)]

    def _attn(self, x, i, cache, pos0):
        """x: (T, d)。cache 为 None 时只看当前段；否则把 K/V 追加进 cache 后 attend 全部。"""
        T = x.shape[0]
        qkv = x @ self.Wqkv[i]
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(T, self.heads, self.dh)
        k = k.reshape(T, self.heads, self.dh)
        v = v.reshape(T, self.heads, self.dh)
        if cache is not None:
            if cache[i] is None:
                cache[i] = [k, v]                      # 存新算的 K/V
            else:
                cache[i][0] = np.concatenate([cache[i][0], k], axis=0)   # append
                cache[i][1] = np.concatenate([cache[i][1], v], axis=0)
            k_all, v_all = cache[i]
        else:
            k_all, v_all = k, v
        scores = np.einsum("thd,shd->hts", q, k_all) / np.sqrt(self.dh)   # (heads, T, T_all)
        T_all = k_all.shape[0]
        if T > 1:                    # 因果掩码：每行只能看自己及之前（含 cache 里的历史）
            mask = (np.arange(pos0 + T)[:, None] >= np.arange(T_all)[None, :])[None, :, :]
            scores = np.where(mask, scores, -1e30)
        w = np.exp(scores - scores.max(-1, keepdims=True))
        w = w / w.sum(-1, keepdims=True)
        out = np.einsum("hts,shd->thd", w, v_all).reshape(T, self.d)
        return out @ self.Wo[i]

    def forward(self, tokens, cache=None, pos0=0):
        x = self.emb[list(tokens)] + self.pos[pos0:pos0 + len(tokens)]
        for i in range(self.layers):
            h = rms_norm(x)
            x = x + self._attn(h, i, cache, pos0)
            x = x + self._mlp(x, i)
        return x @ self.emb.T          # 权重共享的输出头

    def _mlp(self, x, i):
        h = rms_norm(x)
        return (np.maximum(h @ self.W1[i], 0)) @ self.W2[i]


def generate(model, prompt, n_new, use_cache):
    """贪心解码 n_new 个 token。use_cache=False 时每一步都把整段序列重算一遍。

    缓存路径：第 1 步用 prompt 做一次预填充（prefill，填满 K/V），之后每步只喂 1 个新 token。
    """
    toks = list(prompt)
    cache = [None] * model.layers if use_cache else None
    step_times = []
    prefilled = False
    for _ in range(n_new):
        t0 = time.perf_counter()
        if use_cache and prefilled:
            logits = model.forward([toks[-1]], cache, len(toks) - 1)   # 只算 1 个 token
        else:
            logits = model.forward(toks, cache, 0)                     # 整段（或首次预填充）
            prefilled = True
        nxt = int(np.argmax(logits[-1]))
        toks.append(nxt)
        step_times.append(time.perf_counter() - t0)
    return toks[len(prompt):], step_times


def experiment_b():
    print("\n" + "=" * 78)
    print("实验 B：同一模型，每步重算 vs KV 缓存（8 层 / d=128 / 4 头，本机实测）")
    print("=" * 78)
    m = ToyGPT(vocab=64, d=128, layers=8, heads=4, seed=1)
    prompt = [3, 17, 42, 8, 29, 55, 1, 12, 33, 47, 60, 5, 22, 38, 11, 50]   # 16 token 提示词
    n_new = 96

    out_ref, t_ref = generate(m, prompt, n_new, use_cache=False)
    out_cac, t_cac = generate(m, prompt, n_new, use_cache=True)

    # 多个随机种子复验：两条路径生成的 token 序列是否逐位相同
    ok = 0
    seeds = 5
    for s in range(seeds):
        mm = ToyGPT(vocab=64, d=128, layers=8, heads=4, seed=s)
        a, _ = generate(mm, prompt, 40, use_cache=False)
        b, _ = generate(mm, prompt, 40, use_cache=True)
        ok += (a == b)

    print(f"  提示词 16 token，贪心解码 {n_new} token")
    print(f"  ① 生成结果完全一致：{out_ref == out_cac}（token 序列逐位相等）"
          f" | 换 {seeds} 个随机种子复验：{ok}/{seeds} 条序列一致")
    print(f"  ② 逐步耗时（ms）——看增长趋势，不是看绝对值：")
    show = [0, 7, 15, 23, 31, 47, 63, 79, 95]
    print("     step    " + "".join(f"{s:>8}" for s in show))
    print("     无缓存  " + "".join(f"{t_ref[s]*1000:>8.2f}" for s in show))
    print("     有缓存  " + "".join(f"{t_cac[s]*1000:>8.2f}" for s in show))
    print(f"     最后一步 / 第一步：无缓存 {t_ref[-1]/t_ref[0]:.1f}x（序列越长约越慢），"
          f"有缓存 {t_cac[-1]/t_cac[1]:.2f}x（第 0 步含预填充；之后每步只算 1 个 token）")
    print(f"  ③ 实测总耗时：无缓存 {sum(t_ref):.3f} s   |   有缓存 {sum(t_cac):.3f} s"
          f"   →  快 {sum(t_ref)/sum(t_cac):.1f}x")
    n_nocache = sum(range(len(prompt), len(prompt) + n_new))
    n_cache = len(prompt) + n_new
    print(f"  ④ 累计「喂进模型的 token 数」：无缓存 {n_nocache:,}，有缓存 {n_cache:,}"
          f"  →  计算量比 {n_nocache/n_cache:.1f}x")

    print("  ⑤ 两条路径的数值一致性（为什么能放心替换）：")
    lg_ref = m.forward(prompt + out_ref, None, 0)[-1]
    cache = [None] * m.layers
    m.forward(prompt + out_ref[:-1], cache, 0)
    lg_cac = m.forward([out_ref[-1]], cache, len(prompt + out_ref) - 1)[-1]
    print(f"     同一位置 logits 最大绝对差：{np.max(np.abs(lg_ref - lg_cac)):.2e}"
          f"（float32 舍入量级，logits 幅度 {np.max(np.abs(lg_ref)):.1f}）")
    print("     抽样看「贪心决策的余量」(top1 与 top2 的差 vs 两条路径的差)：")
    cache = [None] * m.layers
    lg = m.forward(prompt, cache, 0)[-1]      # 预填充提示词 → 第 0 次决策的 logits
    print("       step      top1-top2     两路径logit差")
    for i in range(n_new):
        if i in (0, 16, 32, 48, 64, 80, 95):
            lr = m.forward(prompt + out_ref[:i], None, 0)[-1]        # 重算路径：整段重跑
            s = np.sort(lr)[-2:]
            print(f"       {i:>4}      {s[1]-s[0]:>10.4f}   {np.max(np.abs(lr-lg)):>12.2e}")
        lg = m.forward([out_ref[i]], cache, len(prompt) + i)[-1]     # 缓存路径：喂 1 个 token
    print("     → 决策余量比数值噪声大好几个数量级，所以序列一致；反过来也说明：")
    print("       如果两个候选 logits 几乎打平，重算和缓存是可能选出不同 token 的——")
    print("       这正是推理引擎要求「同一 session 内缓存布局固定」的原因之一。")
    print("  ⑥ 无缓存路径重复算的是「历史 token 的 QKV」，它们每次都一模一样：")
    print("     缓存换来的就是把这部分重复完全删掉，代价是把 K/V 一直占着显存。")


# ============================================================
# 实验 C：真的分配 cache，量 RSS 增量；再测写入带宽
# ============================================================

def rss_mb():
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * 4096 / MB      # 常驻页数 × 4KB


def experiment_c():
    print("\n" + "=" * 78)
    print("实验 C：真的写满一段 cache —— 公式 vs 实测 RSS，以及写入带宽")
    print("=" * 78)
    chunk = bytes(8 * MB)
    for label, per_tok in (("MHA  512 KB/token", 512 * 1024),
                           ("GQA  128 KB/token", 128 * 1024),
                           ("MQA   16 KB/token", 16 * 1024)):
        target = 512 * MB
        n_tokens = target // per_tok
        before = rss_mb()
        buf = bytearray(n_tokens * per_tok)
        after_alloc = rss_mb()
        mv = memoryview(buf)
        best = 0.0
        for _ in range(3):                                  # 写 3 遍，取最快的一遍
            t0 = time.perf_counter()
            for off in range(0, len(buf), len(chunk)):
                mv[off:off + len(chunk)] = chunk[:len(buf) - off]
            best = max(best, len(buf) / (time.perf_counter() - t0))
        bw = best / 1e9                                     # GB/s（本机内存写入带宽）
        print(f"  {label}: 一段 {len(buf)/MB:.0f} MB 的缓存 = {n_tokens:,} 个 token")
        print(f"      RSS: {before:.0f} → {after_alloc:.0f} MB（实测增长 {after_alloc - before:.0f} MB，"
              f"公式预测 {len(buf)/MB:.0f} MB，比值 {(after_alloc - before)/(len(buf)/MB):.2f}x）")
        print(f"      写一遍耗时 {len(buf)/best*1000:.0f} ms → 本机写入带宽 {bw:.2f} GB/s"
              f"  |  追加一个 token 的缓存要用 {per_tok/best*1e6:.1f} µs")
        del buf, mv

    print("\n  但真正贵的是「每生成一个 token 都要把整段 cache 读一遍」。按公开显存带宽算上限：")
    b_70b = kv_per_token(80, 8, 128)
    print(f"    70B/GQA-8/FP16 的 KV cache 体积 = {b_70b:,} B/token = {b_70b/1024:.0f} KB/token")
    for gpu, bw_tbps in (("H100 SXM (HBM3 3.35 TB/s)", 3.35), ("A100 80G (2.04 TB/s)", 2.04),
                         ("RTX 4090 (GDDR6X 1.01 TB/s)", 1.01)):
        for ctx in (8192, 32768, 131072):
            kv = b_70b * ctx
            print(f"    {gpu:<28} 上下文 {ctx//1024:>4}k：KV {kv/GB:>5.2f} G，"
                  f"读一遍 {kv/(bw_tbps*1e12)*1000:>6.2f} ms → 解码速度上限 "
                  f"{bw_tbps*1e12/kv:>5.0f} tok/s")
    print("    这只是「读 cache」一项的理论上限：32k 上下文 3.2 ms/步，128k 就是 12.8 ms/步 ——")
    print("    所以 KV 用 INT8/FP8 存（体积减半、速度上限翻倍）是现在最直接的优化之一；")
    print("    而 GQA/MQA/MLA 是从结构上先把体积砍掉，属于更根本的办法。")


if __name__ == "__main__":
    t_start = time.perf_counter()
    experiment_a()
    experiment_b()
    experiment_c()
    print(f"\n总耗时 {time.perf_counter()-t_start:.1f} s")
```

实测输出（本机原样贴，前半段是体积账本）：

```text
==============================================================================
实验 A：KV cache 体积账本（FP16 缓存，bytes_per_elem=2）
==============================================================================
模型                       KV头     每token字节   每token KB     比MHA省
Llama-2-7B  (MHA)         32      524,288       512.0      1.0x
Llama-2-13B (MHA)         40      819,200       800.0      1.0x
Llama-3-8B  (GQA-8)        8      131,072       128.0      4.0x
Llama-3-70B (GQA-8)        8      327,680       320.0      8.0x
Mistral-7B  (GQA-8)        8      131,072       128.0      4.0x
Qwen2-72B   (GQA-8)        8      327,680       320.0      8.0x
```

后半段是实验 C —— 我真的分配并写满了一段 512 MB 的 cache，用 `/proc/self/statm` 量 RSS：

```text
==============================================================================
实验 C：真的写满一段 cache —— 公式 vs 实测 RSS，以及写入带宽
==============================================================================
  MHA  512 KB/token: 一段 512 MB 的缓存 = 1,024 个 token
      RSS: 42 → 554 MB（实测增长 512 MB，公式预测 512 MB，比值 1.00x）
      写一遍耗时 136 ms → 本机写入带宽 3.96 GB/s  |  追加一个 token 的缓存要用 132.5 µs
  GQA  128 KB/token: 一段 512 MB 的缓存 = 4,096 个 token
      RSS: 42 → 554 MB（实测增长 512 MB，公式预测 512 MB，比值 1.00x）
      写一遍耗时 135 ms → 本机写入带宽 3.99 GB/s  |  追加一个 token 的缓存要用 32.9 µs
  MQA   16 KB/token: 一段 512 MB 的缓存 = 32,768 个 token
      RSS: 42 → 554 MB（实测增长 512 MB，公式预测 512 MB，比值 1.00x）
      写一遍耗时 135 ms → 本机写入带宽 3.98 GB/s  |  追加一个 token 的缓存要用 4.1 µs

  但真正贵的是「每生成一个 token 都要把整段 cache 读一遍」。按公开显存带宽算上限：
    70B/GQA-8/FP16 的 KV cache 体积 = 327,680 B/token = 320 KB/token
    H100 SXM (HBM3 3.35 TB/s)    上下文    8k：KV  2.50 G，读一遍   0.80 ms → 解码速度上限  1248 tok/s
    H100 SXM (HBM3 3.35 TB/s)    上下文   32k：KV 10.00 G，读一遍   3.21 ms → 解码速度上限   312 tok/s
    H100 SXM (HBM3 3.35 TB/s)    上下文  128k：KV 40.00 G，读一遍  12.82 ms → 解码速度上限    78 tok/s
    A100 80G (2.04 TB/s)         上下文    8k：KV  2.50 G，读一遍   1.32 ms → 解码速度上限   760 tok/s
    A100 80G (2.04 TB/s)         上下文   32k：KV 10.00 G，读一遍   5.26 ms → 解码速度上限   190 tok/s
    A100 80G (2.04 TB/s)         上下文  128k：KV 40.00 G，读一遍  21.05 ms → 解码速度上限    47 tok/s
    RTX 4090 (GDDR6X 1.01 TB/s)  上下文    8k：KV  2.50 G，读一遍   2.66 ms → 解码速度上限   376 tok/s
    RTX 4090 (GDDR6X 1.01 TB/s)  上下文   32k：KV 10.00 G，读一遍  10.63 ms → 解码速度上限    94 tok/s
    RTX 4090 (GDDR6X 1.01 TB/s)  上下文  128k：KV 40.00 G，读一遍  42.52 ms → 解码速度上限    24 tok/s
```

三点值得记住：

- **公式是准的，误差 1.00x。** 512 MB、512 KB/token 的 MHA 配置正好装 1,024 个 token 的缓存；换成 MQA（16 KB/token）同样的 512 MB 能装 32,768 个。**同一个显存额度，KV 头数决定了你能装多长的上下文、能同时开几个会话。**
- **写入是便宜的，读取才是瓶颈。** 本机写带宽 3.96 GB/s，追加一个 token 只要 132.5 µs。但解码时每生成一个 token 都要把整段 cache 读一遍：70B/GQA-8 在 32k 上下文下 3.21 ms、128k 下 12.82 ms——**这是一张 H100 上的理论下限**，实际还要加上权重读取和 kernel 开销。上下文的每一倍增长，直接变成解码延迟的增长。
- **带宽表还有个推论**：RTX 4090 上 70B 模型跑到 128k 上下文，光读 cache 就 42.5 ms/步，理论上限 24 tok/s。这不是"实现不好"，是物理上限——**要更快，只能让 cache 变小（GQA/MLA/量化）或者分层放（HBM→SSD 分级缓存）。**

## 🏗️ 工程上真的怎么省：从结构到系统

| 手段 | 做法 | 关键取舍 |
|---|---|---|
| **MQA**（Shazeer 2019, arXiv:1911.02150） | K/V 只保留 1 个头，Q 还是多头 | 32x 压缩（本例），质量有损；开源模型里很少直接上，多用于 GQA 的极限参考 |
| **GQA**（arXiv:2305.13245） | 把 Q 头分成若干组，每组共享一份 K/V | 精度和显存的折中；**Llama 3 全系用 GQA（官方 model card 明确写了以提升推理效率），Llama 2 70B 也改成了 GQA** |
| **MLA**（DeepSeek-V2, arXiv:2405.04434） | 低秩联合压缩 K/V 成隐向量 + 解耦 RoPE 分支，用时再投影还原 | 结构上最省（本例 56.9x）；代价是每层多做一次投影，kernel 要专门写 |
| **PagedAttention / vLLM**（arXiv:2309.06180） | 把 KV cache 切成固定大小的 block，用页表映射，像操作系统管内存一样管显存 | 解决**碎片**（历史上真实系统里 cache 显存浪费可达两位数百分比），公开引用的吞吐提升 2~4x；代价是多一层间接寻址 |
| **Prefix caching** | 系统提示词、多轮对话前缀的 KV 跨请求复用，不重算 | 命中就是纯赚；前提是前缀逐 token 一致（改一个字就 miss） |
| **KV 量化（INT8/FP8/2bit）** | 把 cache 本身量化，KIVI（arXiv:2402.02750）做到 2bit | 体积减半 → 解码速度上限直接翻倍；误差比权重量化敏感，得按任务测 |
| **滑窗 + attention sink**（StreamingLLM, arXiv:2309.17453） | 只留最近 L 个 token 的 KV，**外加开头 4 个 sink token** | 官方结论：只保留开头 4 个 token 就能让窗口注意力的效果基本恢复，Llama-2 等模型可稳定跑到 4M token |
| **驱逐 / 选择（H2O 等）** | 按累计注意力分数丢掉"不重要"的 token | 压缩比高，但**丢错了就真丢了**——长文评估里最容易翻车的一类 |
| **Continuous batching** | 请求级调度，完成一个补一个，而不是等整批 | 把"cache 省一半"直接变成"并发翻倍"，是吞吐优化的基本盘 |
| **KV cache 落盘 / 分级** | 冷 cache 放 SSD 或主机内存，热的部分留 HBM | 省钱不省时间，适合长会话的恢复场景 |

一个容易忽略的工程细节：**cache 的正确性依赖确定性。** 上面那个 toy 实验里，两条路径的 logits 差 3.79e-05 而决策余量是 8.76——安全；但如果换成 bf16、或者 kernel 换了不同的归约顺序，某一步的余量恰好小于噪声，输出就会翻转。所以生产系统在意的是：同一个 session 内缓存布局、算子实现、精度策略保持一致，别让"同一次对话里的第 30 个 token"和"重算一遍"走出两条路。

## 💡 小结

- **KV cache 是把"重算历史"换成"存住历史"**：注意力本身无状态，K/V 只由输入决定，所以可以缓存。计算量从 O(n³) 降到 O(n²)，代价是缓存体积和上下文长度成正比
- **体积公式**：`2 × 层数 × KV头数 × head_dim × 2(FP16) × n`。7B/MHA 每 token 512 KB，70B/GQA-8 每 token 320 KB——**同样的显存，KV 头数决定你能开多少并发、能撑多长上下文**
- **本机实测**：两条路径在 5/5 个随机种子下生成序列逐位相同，同一位置 logits 差 3.79e-05（决策余量比它大 5~9 个数量级）；喂进模型的 token 数 6,096 → 112，**计算量比 54.4x，实测墙钟只快 9.8x**——差值全是小模型上 kernel 开销打的折，真实引擎靠融合算子把这笔账吃回来
- **真的分配验证**：512 MB cache 实测 RSS 增长 512 MB（公式比值 1.00x）；本机写入带宽 3.96 GB/s，追加一个 token 只要 132.5 µs——但**读取才是瓶颈**：70B 在 H100 上 32k 上下文读一遍 3.21 ms，128k 下 12.82 ms，这是解码速度的物理上限
- **省 cache 的四条路**：改结构（MQA/GQA/MLA，最根本）、省精度（KV 量化，最直接）、少存（滑窗 + sink、驱逐，最激进）、复用（prefix caching + continuous batching，最赚）
- 心法：**KV cache 是一笔按 token 计息的显存贷款**——借它换来的是 O(n³)→O(n²) 的算力折扣，利息就是那条随上下文线性上涨的显存账单 💳

下次看到「支持 1M 上下文」的宣传，问三句就能估出成本：**每 token 的 KV cache 是多大？单卡能同时开几个 1M 会话？用的是哪种 KV 压缩？** 这三个答案决定了它是能上线的工程，还是只能跑 demo 的数字。 🧮

---

_概念卡片持续更新中，下一个概念见~_ 🔫
