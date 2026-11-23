---
title: "吞吐 6.37 req/s 的那档 goodput 是 0.00，吞吐 5.21 那档是 5.21：容量指标该换成「达标的请求数」"
description: "同一套服务、同一份成本模型（离散事件模拟，非 GPU 实测）：到达率从 5 提到 10 req/s，吞吐从 5.21 涨到 6.37（+22%），goodput 从 5.21 掉到 0.00，P50 首 token 从 135ms 变成 40.8 秒。反过来把最大批从 8 扩到 128：吞吐 2.38→5.21，goodput 0.02→5.21，但拐点在 48——32 并发时吞吐 4.75、达标率只有 5%。两种失败模式分开看：批太大先破 TPOT（λ=6 时 75ms 超过 60ms 的 SLO，达标率从 100% 掉到 14%），负载超容量则破 TTFT。"
pubDate: 2026-11-23
tags: ["AI核心概念", "推理服务", "goodput", "SLO", "连续批处理", "排队论"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 91
slug: "ai-concepts-91-goodput"
---

> 压测报告上写「吞吐 6.4 req/s」，评审会上没人反对，上线后用户说卡。
> 同一个服务把到达率从 5 加到 10 req/s：吞吐从 5.21 涨到 6.37（+22%），**goodput 从 5.21 掉到 0.00**，P50 首 token 从 135ms 变成 40.8 秒。
> 这两条线在负载翻倍的全过程里都「看起来还好」——直到你只统计达标的那部分。

读这篇要会：TTFT（首 token 延迟）与 TPOT（相邻 token 间隔）的定义 / 排队系统的利用率与排队长度大致什么关系 / KV cache 显存怎么按 token 数算。

## 🎯 为什么 tok/s 是个会骗人的容量指标

任何推理服务的压测都会输出 tokens/s。它的三个毛病叠在一起，正好让最坏的配置看起来最好。

第一，它把 prefill 和 decode 混在一起算。一个 4k prompt 的请求 prefill 要吃掉几千 token 的算力，只产出几百 token；一个长输出的请求反过来。两个不同的流量组合可以在同一 tok/s 下对应完全不同的用户体验。

第二，它没有约束。把批拉到 128、把队列堆到几百条，GPU 会一直满载，tok/s 甚至更好看——因为延迟约束不在分母里。

第三，它不可加。用户不会因为系统吐了 1400 tok/s 而满意，他满意的是「我这句话 300ms 内开始回、每秒回 17 个字」。

所以容量指标得换成 goodput：

```text
goodput = (1/T) · Σ_i 1[ TTFT_i ≤ S_ttft ∧ TPOT_i ≤ S_tpot ]
```

一句话：**每秒有多少个请求，是在合约内完成的**。S_tpot 取下界时它就是「满足交互体验的请求率」，写进 SLA 里可以直接对账。

## 🔬 三个约束，把「更快的批」变成「更慢的用户」

**① 显存决定同时能跑几条。** KV cache 每 token 的字节数是固定的：

```text
KV/token = 2(K,V) × 32 层 × 8 个 KV 头 × 128 维 × 2 字节 = 128 KB
B_max    = (80 GB − 16 GB 权重) / (128 KB × 4096) ≈ 128 条
```

这个 128 是硬上限，不是调参建议。把上下文从 4k 提到 32k，它立刻掉到 16 条——上下文窗口是拿并发数换的。

**② 批大小线性推高 TPOT。** decode 每步要读一遍 KV，序列越多读得越久，近似 `t_step = 10ms + 0.28ms × n`：批 8 约 12ms，批 64 约 28ms，显存上限的 128 条约 46ms。批越大，单条越慢。表里的 P50 TPOT 比这个式子略高（多出来的是同一迭代里分摊的 prefill 时间），而且从 64 往上就停在 38ms——因为同时在跑的序列条数由流量决定，批上限再大也用不上。

**③ 负载逼近容量时，队列长度按 Little 定律放大。** `L = λW`：系统内请求数等于到达率乘以平均停留时间。服务能力 6.4 req/s、到达 10 req/s，多出来的 3.6 req/s 全部堆在队列里，`W` 随时间线性增长，TTFT 跟着涨。这不是抖动，是算术。

两种失败因此长得完全不一样：**批太大 → TPOT 先破 SLO；负载超容量 → TTFT 先破 SLO**。goodput 把两者压成一个数，但要定位原因还得分开看。

## 🏗️ 实验设置：先承认这是模拟

没有 GPU 可用，所以这是一份**离散事件模拟，不是 GPU 实测**。成本模型只有三行：prefill 聚合吞吐 11000 tok/s、每迭代固定开销 10ms、每并发序列附加 0.28ms；分块 prefill 每迭代最多推进 2048 token；请求的 prompt 取对数正态（中位数 700 token），输出 128–320 token；到达是泊松过程。SLO 取 TTFT ≤ 1500ms、TPOT ≤ 60ms（≈17 tok/s）。

模型的盲区也要说清：它假设 prefill 与 decode 的时间直接相加、没有 kernel 级重叠，也不模拟投机解码、PD 分离、前缀缓存——这些都会改变绝对数值。它可靠的部分是**结构性关系**：批与 TPOT 的线性、队列与 TTFT 的发散、以及 goodput 与吞吐在饱和点附近分岔。

## 💻 实测：两条曲线分岔的位置

**实验 A**：负载固定 5 req/s，扫显存允许的最大批大小。

```text
最大批  吞吐req/s  P50TTFT  P99TTFT  P50TPOT  goodput  达标率
    8     2.38    172.1s   339.4s    14ms     0.02     1%
   16     3.58     66.1s   131.4s    19ms     0.11     3%
   32     4.75     15.2s    28.8s    29ms     0.23     5%
   48     5.19    677ms    3144ms    38ms     3.86    74%
   64     5.21    146ms    1250ms    38ms     5.20   100%
   96     5.21    135ms     678ms    38ms     5.21   100%
  128     5.21    135ms     678ms    38ms     5.21   100%
```

读这张表要按「谁先坏」来读。批 8 的 TPOT 是全场最好的 14ms——单条最快——但服务能力只有 2.4 req/s，喂 5 req/s 的流量等于每秒欠 2.6 个请求，队列堆到 P50 首 token 172 秒，goodput 0.02。**慢不是问题，能力不够才是问题。**

批 32 是最容易误判的一档：吞吐 4.75 req/s 已经不低，P50 TPOT 29ms 看着健康，goodput 只有 0.23——达标率 5%。因为服务能力 4.75 仍小于到达 5。

拐点在 48–64 之间：48 时容量 5.19 略低于需求，达标率 74%，`goodput 3.86`；64 时容量 5.21 刚刚够，达标率 100%。再往上 96、128 完全一样——**这时候加批大小一分钱收益都没有**，因为瓶颈已经不在 GPU，而在流量本身。这就是「扩容扩到底」的真实形状：不是收益递减，是收益归零。

**实验 B**：批固定 128（显存上限），扫到达率。

```text
  λ   吞吐req/s  P50TTFT  P99TTFT  P50TPOT  goodput  达标率
 4.0    4.19     113ms     557ms    25ms     4.19   100%
 5.0    5.21     135ms     678ms    38ms     5.21   100%
 6.0    6.16     219ms    1721ms    75ms     0.86    14%
 7.0    6.33     12.5s     27.3s    87ms     0.10     2%
 8.0    6.35     23.8s     51.5s    87ms     0.03     1%
10.0    6.37     40.8s     85.8s    87ms     0.00     0%
```

λ=6 只比上一行多 20% 的流量，吞吐立刻涨了 18%（6.16 是本表最漂亮的一行之一），而达标率从 100% 砸到 14%：TPOT 从 38ms 涨到 75ms，越过 60ms 的 SLO。再往上，吞吐几乎不动（6.33 → 6.37，+0.6%），P50 首 token 从 219ms 涨到 40.8 秒。

**同一套服务、同一份代码：吞吐 6.37 时 goodput 是 0.00，吞吐 5.21 时 goodput 是 5.21。** 吞吐更高的那个配置，一个达标的请求都没有。

模型自检：λ=10 时系统内请求数的时间加权均值是 388 条，`λ_eff × 平均停留 = 389` 条——Little 定律闭合，说明队列的增长不是 bug，是负载真的超了。

核心循环只有二十来行（完整脚本本机 2 秒跑完，纯标准库）：

```python
while len(done) < N_REQ:
    # 准入：KV cache 槽位决定同时在跑的序列数
    free = max_batch - len(decoding) - len(prefilling)
    while queue and free > 0:
        r = queue.pop(0); r.update(prompt_left=r['P'], o_left=r['O']); prefilling.append(r); free -= 1
    # 一个迭代：分块 prefill + 所有就绪序列各出 1 个 token
    budget, used, ready = PREFILL_BUDGET, 0, []
    for r in prefilling:
        take = min(r['prompt_left'], budget); r['prompt_left'] -= take
        budget -= take; used += take
        if r['prompt_left'] == 0: ready.append(r)
        if budget <= 0: break
    t_end = t + used / PREFILL_RATE + DEC_BASE + DEC_PER_SEQ * (len(decoding) + len(ready))
    for r in ready:
        prefilling.remove(r); r['ttft'] = t_end; decoding.append(r)
    alive = []
    for r in decoding:
        r['o_left'] -= 1
        if r['o_left'] == 0: r['finish'] = t_end; done.append(r)
        else: alive.append(r)
    decoding, t = alive, t_end
```

完整脚本在仓库：[demos/ai-concepts-91-goodput.py](https://github.com/shenlv7/blog/blob/main/demos/ai-concepts-91-goodput.py)。

## 💡 工程上怎么用这两个数

**报容量时报 goodput，并且带上 SLO。** 「5.2 req/s @ TTFT 1.5s / TPOT 60ms」是一句可核对的话；「1170 tok/s」不是。

**容量规划的系数按曲线陡度取，不按经验值。** 这张表里 goodput 在利用率 100%→120% 的区间里从峰值掉到 14%，几乎没有缓冲带。也就是说「留 20% 余量」在这套参数下刚好落在悬崖上，得留 30–40%，或者压缩排队：更小的 prompt、前缀缓存、更好的调度，都比加卡便宜。

**两个 SLO 分开监控，因为它们的病理不同。** TPOT 破线说明批或并发太大（`t_step` 的线性项），TTFT 破线说明排队在涨（容量不够）。把 goodput 当成唯一的健康指标，你会知道坏了，但不知道该调哪。

**上下文预算是并发预算。** 128 KB/token 这个系数意味着 4k 上下文最多 128 条并发、32k 只剩 16 条。长上下文的成本不是显存峰值，是它把批压小以后把 TPOT 顶上去，以及把 goodput 直接砍掉一个量级。

**别把模拟的数字当结论，把关系当结论。** 绝对吞吐换成真卡就变，但「批线性推高 TPOT」「利用率过 100% 后 TTFT 发散」「goodput 在饱和点断崖」这三条形状，在任何按带宽受限的 decode 栈上都成立——推理框架里那些默认值（`max_num_batched_tokens`、`max_num_seqs`、chunked prefill 的块大小）本质都在调这条曲线的位置。

下一篇顺着 `t_step` 里的线性项往下钻：既然 decode 的耗时基本全部花在读 KV 上，那么让每条序列少读一点——KV 量化、驱逐、共享——省下来的到底是什么，以及在什么条件下省不下来。
