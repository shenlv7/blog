---
title: "55 | 混合检索：把 BM25 和一个瞎猜的检索器融合，nDCG 从 0.72 掉到 0.20"
description: "600 篇合成语料、200 条查询、本机 CPU 纯 numpy 实测（无 GPU、无预训练 embedding、无真实业务数据）。三条路的 nDCG@10：BM25 0.7249、稠密 0.6115、RRF 融合 0.7846。但拆开看：编号式查询 BM25 命中率 100%，改写式查询 BM25 的 hit@10 是 0.000（查询词一个都不在金标文档里，分数结构性为零）；稠密路正好反过来，编号查询能被召回（hit@10 0.938）却只有 39% 排第一。融合整体最高，但 28.5% 的查询上它反而比最佳单路差，11% 差超过 0.1 nDCG，最差一条直接 -1.0。最狠的一条：把稠密路换成纯随机排名，RRF 掉到 0.2004——比干脆只用 BM25 差 72.4%，因为 RRF 给每一路的总票权是固定的，与这一路准不准无关。"
pubDate: 2026-10-18
tags: ["AI核心概念", "混合检索", "BM25", "稠密检索", "RRF", "排名融合", "向量检索", "RAG"]
difficulty: advanced
series: "ai-concepts"
seriesOrder: 55
slug: ai-concepts-55-hybrid-retrieval
---

# AI核心概念(55)：混合检索——融合赢在平均，输在单条

> 这一篇只回答一个很具体的问题：**BM25 和向量检索拼在一起，到底加了多少？什么时候加的是负分？**
>
> 我在本机 CPU 上用纯 numpy 跑了一遍（600 篇合成语料、200 条查询、零 GPU、零预训练模型）。三条路的总 nDCG@10：**BM25 0.7249、稠密 0.6115、RRF 融合 0.7846**——融合比最好的单路高 8.2%。
>
> 但这个 0.78 是平均值，平均值会盖住两件更要紧的事。第一，**28.5% 的查询上融合比最佳单路更差**，11% 差超过 0.1，最差的一条直接 -1.0（金标被挤出前十）。第二，把稠密那一路换成**纯随机排名**，融合从 0.7846 掉到 **0.2004**——比干脆只用 BM25 还低 72.4%。
>
> 所以这篇文章的重心不是"混合比单路好"，而是**为什么它好、它在什么条件下会背叛你**。答案藏在 RRF 的一行公式里：它对每一路的票权做归一化，跟这一路准不准毫无关系。
>
> 边界先说死：**全部是本机 CPU 上的纯 numpy 实测**，合成语料，稠密向量是我从语料自己算的「词-词共现 → PPMI → SVD」，**不是预训练 embedding 模型**。凡涉及真实系统（DPR、SPLADE、生产级混合检索）的结论，我只标出处，不冒充我测过。

## 🎯 三个词，先各自站好

**BM25**：稀疏检索的默认基线，源自 1994 年 TREC-3 的 Okapi 系统（Robertson、Walker、Jones、Hancock、Beaulieu），系统综述见 Robertson & Zaragoza 2009 年的 *The Probabilistic Relevance Framework: BM25 and Beyond*。它只做三件事，但每件都做得很讲究：**IDF**（越罕见的词权重越高）、**词频饱和**（同一个词出现 10 次不等于比 1 次重要 10 倍）、**长度归一**（长文档不该因为词多就占便宜）。一句话：**它数词，但数得比"数词"聪明。**

**稠密检索**：把查询和文档都编码成定长向量，用余弦或点积比相似度。真实系统用的是对比学习训出来的双塔编码器，代表作是 Karpukhin 等人 2020 年的 **DPR**（*Dense Passage Retrieval for Open-Domain Question Answering*，arXiv:2004.04906）。我这里没有编码器，用的是 **LSA**（Deerwester 等人 1990 年 *Indexing by Latent Semantic Analysis*）：词-词共现矩阵 → PPMI → 截断 SVD。它和 word2vec 是同一个祖宗——Levy & Goldberg 2014 年证明 SGNS 本质上在分解一个平移过的 PMI 矩阵。**但它没见过任何"问-答"对，所以只能靠二阶共现抓同义，抓不到真正的改写。**

**排名融合 / RRF**：Reciprocal Rank Fusion，Cormack、Clarke、Buettcher 在 SIGIR 2009 上提出，公式就一行：

```
score(d) = Σ_i  1 / (k + rank_i(d))        # k 通常取 60
```

它**只吃名次，不吃分数**。这是它最大的优点：不用把 BM25 的 0~15 分和余弦的 0.2~0.9 掰到同一把尺子上。替代方案是分数加权（CombSUM，Fox & Shaw 1994）——更灵活，但要多调一个 α。

## 🔬 实验设置，以及它为什么不算真实系统实测

| 项 | 设置 |
|---|---|
| 语料 | 600 篇；词表 359（含 220 个只出现一两次的长尾词，制造真实的重尾） |
| 文档构造 | 8 个主题簇 × 每簇 8 个近义词；每篇取主簇 3 词 + 副簇 2 词 + 16 个虚词 |
| 编号 | 80 个形如 `qx1000` 的编号，**全库 df=1**（实测 min=max=1） |
| 查询 | 200 条：K=编号式 80、P=改写式 80（与金标文档**零词面重叠**）、M=编号+改写词 40 |
| 相关标签 | 金标文档 3 分，同主题同簇文档 1 分；指标为分级相关的 nDCG@10 |
| 稠密路 | 词-词共现（窗口 ±5，按距离加权）→ PPMI → SVD 取 32 维；文档向量 = sqrt(tf) 加权词向量均值 |

32 维占 PPMI 平方谱能量的 **38.98%**（非零奇异值 252 个）。也就是说这一路是有损压缩，不是查表。

**不能从这个实验外推的东西**：合成语料没有真实语义；同簇文档之间差异极小（这是改写查询难看的根本原因）；LSA 远弱于预训练编码器；没有分词差异、没有多语言、没有 rerank 段。

## 📊 实验A：两条路的盲区，恰好互补

| 方法 | nDCG@10 | 编号 K | 改写 P | 混合 M |
|---|---|---|---|---|
| 随机排名（零假设） | 0.0153 | 0.006 | 0.027 | 0.010 |
| BM25 | 0.7249 | **1.000** | 0.312 | **1.000** |
| 稠密 32 维 | 0.6115 | 0.774 | **0.511** | 0.486 |
| RRF k=1 | 0.7828 | 1.000 | 0.475 | 0.963 |
| RRF k=10 | **0.7846** | 1.000 | 0.493 | 0.936 |
| RRF k=60（文献默认） | 0.7660 | 1.000 | 0.482 | 0.865 |
| RRF k=200 | 0.7605 | 1.000 | 0.479 | 0.846 |

零假设先钉住：随机排名 nDCG 0.0153，改写查询 0.027——所以后面的 0.31 / 0.51 不是噪声。**先量随机，再谈效果**，这一条是上一篇（水印基准线偏移）留下的教训。

再看金标文档的命中率，这才是"能不能用"的分水岭：

| 方法 | K hit@10 | P hit@10 | M hit@10 |
|---|---|---|---|
| BM25 | 1.000 | **0.000** | 1.000 |
| 稠密 32 维 | 0.938 | 0.075 | 0.725 |
| RRF k=10 | 1.000 | 0.025 | 1.000 |

**编号式查询，BM25 是 100%，而且几乎都排第一。** 这不是调参调出来的，是 IDF 的定义：编号 df=1，IDF = ln(1 + (600−1+0.5)/(1+0.5)) = **5.993**，而主题词 IDF 是 **2.578**，差 2.32 倍。一个词的稀有度本身就是最强的相关性信号。

**改写式查询，BM25 的 hit@10 是 0.000。** 0 不是"差"，是**结构性为零**：查询里的词一个都不在金标文档里，BM25 给它的分数就是 0，而库里有十篇以上文档分数大于 0。纯稀疏检索对改写不是弱，是**没有**。

**稠密路正好镜像。** 改写查询 hit@10 只有 0.075——它能捞到"同主题的文档"（nDCG 0.511 主要是 1 分档的相邻文档在撑着），但捞不到"那一篇"。这一条是玩具的天花板：我的语料里同簇文档几乎没差别，LSA 也只能分到这个粒度。**不要把这个 7.5% 当成 dense 的真实水平**，真实编码器在改写上强得多。

编号查询它倒是有 hit@10 0.938——能找到候选，但只有 39% 排第一。机制可以量：编号在语料里只出现一次，它的向量只能从这一次的 5 词窗口估出来，与"它所在文档"的余弦 **0.3297**，与随机文档 **0.1544**，差 0.18 而不是 0.9。**因为罕见词没有被学过，它只能拿"平均长相"去匹配。** 真实系统同理：BPE 能把 `QX-3147` 切成子词，但这个具体编号，模型没有为它分配过方向。

## 📊 实验B：融合赢在平均，输在单条

RRF 的 k 值得单独说：**文献默认的 60 在我这里不是最优**（0.7660），k=10 更好（0.7846）。k 越小越偏向头部名次，k 越大越平等。分数加权呢？

| α（BM25 权重） | 0 | 0.2 | 0.35 | 0.5 | 0.65 | 0.8 | 1.0 |
|---|---|---|---|---|---|---|---|
| 总 nDCG | 0.6115 | 0.7598 | 0.7890 | **0.7964** | 0.7919 | 0.7811 | 0.7249 |
| 编号 K | 0.774 | 0.991 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| 改写 P | **0.511** | 0.518 | 0.502 | 0.491 | 0.480 | 0.453 | 0.312 |

α=0.5 的凸组合比最好的 RRF 还高 0.0118。但**没有任何一个 α 在所有查询类型上都对**：K 类要到 α≥0.35 才满分，P 类在 α=0 最好、α=0.5 已经开始掉。这就是 RRF 存在的全部理由——**不想为每种查询分布去调 α，就用只吃名次的融合，代价是放弃一点上限。**

然后是这篇文章最该被记住的一张表：逐步对比"融合 vs 当次最佳单路"。

| 查询类型 | 融合后悔值均值 | 融合更差的占比 |
|---|---|---|
| 编号 K | +0.0000 | 0.0% |
| 改写 P | −0.0338 | **61.3%** |
| 混合 M | −0.1347 | 20.0% |
| 全部 | −0.0405 | 28.5% |

整体：更好 15.0%、持平 56.5%、**更差 28.5%**；11% 的查询差超过 0.1 nDCG；最差一条 **-1.0**——金标被融合挤出了前十。

**融合不会让你崩，但它会系统性地把你最好那一路的优势削平。** 如果你的查询里有 80% 是编号式，BM25 单路就是 1.000，混合检索在这一段上是在给自己找麻烦——它用一半的票权，去听一个在这类查询上只有 0.774 的意见。

## 📊 实验C：票权归一化 = 它默认"两路一样可信"

RRF 为什么会被垃圾分支带崩？把公式摊开算：

| k | 单路总票权 | 第 1 名占 | 前 10 名占 |
|---|---|---|---|
| 1 | 5.9766 | 8.4% | 33.8% |
| 60 | 2.3903 | 0.7% | 6.4% |
| 200 | 1.3844 | 0.4% | 3.5% |

关键在第二列：**每一路贡献的总票权是个固定数字，跟这一路准不准完全无关。** 一个纯随机排名，同样贡献 2.3903 的票。而 k=60 时第 1 名只值 0.7% 的总票权——**一个随机排名把某篇文档放在第 1 位，它拿到的那点票，足以顶掉 BM25 排第 2 的文档。**

实测把这件事说透了：

| 稠密那一路的样子 | 稠密单路 nDCG | RRF 融合 nDCG | 编号 K |
|---|---|---|---|
| 32 维（正常） | 0.6115 | **0.7846** | 1.000 |
| 8 维（压缩过狠） | 0.4866 | 0.7709 | 0.947 |
| 纯随机排名 | 0.0153 | **0.2004** | — |

随机分支那一行是全文最贵的一条：**融合 0.2004，比只用 BM25 的 0.7249 低 72.4%。** 8 维那一行更微妙——总 nDCG 还算体面（0.7709），但编号查询的 hit@1 从 1.000 掉到 0.947，**强项被分票了**。

所以结论不是"混合检索更好"，而是：**融合的前提是两路都带信息。任何一路退化成噪声，融合就是放大器，不是保险丝。**

## 🏗️ 生产里怎么落地

1. **并行召回，各取 50~100 条。** 融合需要纵深：只取 top-10 再融合，RRF 手里没有可交换的候选。Elasticsearch/OpenSearch、Qdrant、Weaviate、Vespa 都把混合检索和内建融合做成了配置项（ES 8 起有 RRF retriever，Qdrant 的 Query API 直接支持 `fusion=rrf`）。
2. **融合放在召回之后、rerank 之前。** RRF 只负责"选谁进候选池"，它不看具体内容。真正的语义判别交给交叉编码器 rerank（top-50 → top-10）。
3. **k=60 是经验值，不是定律。** 我的实验里 k=10 更好。要拿自己业务的数据扫一遍，成本很低。
4. **中文 BM25 有个隐藏坑。** Lucene 的 StandardTokenizer 对中文按单字切，BM25 会退化成"字袋"；要么用 bigram，要么换分词器。英文上的结论不能直接搬。
5. **成本别搞错方向。** 本机 600 篇：BM25 建索引 0.03s（倒排 11604 项），稠密建索引 0.73s（一次 SVD）；单查询暴力打分 BM25 0.154ms、稠密 0.045ms。**但这个对比不含编码器推理**——生产里稠密的钱花在"每篇文档编码一次、每个查询编码一次"和 ANN 索引的内存上，不在点积。

## 💻 可运行代码

完整版（含 α 扫描、后悔值、票权诊断、盲区实验）在仓库里，下面是能独立跑的最小版，输出就是上面那张分类型对比的缩小版：

```python
import math
from collections import defaultdict
import numpy as np

rng = np.random.default_rng(7)
TOPIC = {  # 8 个主题簇，每簇 8 个近义词
    "bat": "battery cell charge power runtime drain juice mah".split(),
    "cam": "camera lens sensor photo aperture zoom shutter pixel".split(),
    "ref": "refund reimburse money return credit compensate cashback rebate".split(),
    "shp": "shipping delivery courier parcel freight dispatch tracking transit".split(),
    "war": "warranty guarantee repair service coverage fix replacement maintain".split(),
    "scr": "screen display panel glass brightness oled pixelate flicker".split(),
    "aud": "speaker sound volume mic headphone bass mute echo".split(),
    "sto": "storage disk memory capacity gb filesystem backup corrupt".split(),
}
FILL = "the a and to of my it was for with this very really please issue problem".split()
NAMES = list(TOPIC)
docs, meta = [], []
for i in range(400):
    p = NAMES[i % 8]
    s = NAMES[rng.integers(8)]
    t = [TOPIC[c][j] for c, n in ((p, 3), (s, 2))
         for j in rng.choice(8, size=n, replace=False)]
    t += list(rng.choice(FILL, 14))
    rng.shuffle(t)
    docs.append(t)
    meta.append((p, s))
codes = []
for n, d in enumerate(rng.choice(400, size=50, replace=False)):   # 50 个全库唯一编号
    c = "qx%04d" % (1000 + n)
    docs[d].insert(int(rng.integers(len(docs[d]))), c)
    codes.append((c, int(d)))

vocab = {t: i for i, t in enumerate(sorted({t for d in docs for t in d}))}
N, V = len(docs), len(vocab)
tf, lens = [], np.zeros(N)
df = np.zeros(V)
for i, d in enumerate(docs):
    c = defaultdict(int)
    for t in d:
        c[vocab[t]] += 1
    tf.append(dict(c))
    df[list(c)] += 1
    lens[i] = len(d)
idf, avgdl = np.log(1 + (N - df + .5) / (df + .5)), lens.mean()

def bm25(q):                                    # 稀疏路
    s = np.zeros(N)
    for t in q:
        j = vocab.get(t)
        if j is None:
            continue
        for i, c in enumerate(tf):
            f = c.get(j, 0)
            if f:
                s[i] += idf[j] * f * 2.2 / (f + 1.2 * (.25 + .75 * lens[i] / avgdl))
    return s

C = np.zeros((V, V))                            # 词-词共现 → PPMI → SVD，自己学"稠密"向量
for d in docs:
    ids = [vocab[t] for t in d]
    for i, a in enumerate(ids):
        for j in range(max(0, i - 5), min(len(ids), i + 6)):
            if j != i:
                C[a, ids[j]] += 1 / abs(i - j)
p = C / C.sum()
with np.errstate(divide="ignore", invalid="ignore"):
    ppmi = np.maximum(np.log(p / (p.sum(1, keepdims=True) @ p.sum(0, keepdims=True))), 0)
ppmi[~np.isfinite(ppmi)] = 0
U, S, Vt = np.linalg.svd(ppmi, full_matrices=False)
wv = U[:, :32] * S[:32]
wv /= np.linalg.norm(wv, axis=1, keepdims=True) + 1e-12
dv = np.zeros((N, 32))
for i, d in enumerate(docs):
    v, w = np.zeros(32), 0.0
    for j, c in tf[i].items():                  # sqrt(tf) 加权：压低高频虚词的影响
        v += math.sqrt(c) * wv[j]
        w += math.sqrt(c)
    dv[i] = v / (np.linalg.norm(v) + 1e-12)

def dense(q):                                   # 稠密路
    rows = [wv[vocab[t]] for t in q if t in vocab]
    if not rows:
        return np.zeros(N)
    v = np.sum(rows, axis=0)
    return dv @ (v / (np.linalg.norm(v) + 1e-12))

def rrf(*score_list, k=60):                     # 排名融合：只吃名次，不吃分数
    out = np.zeros(N)
    for s in score_list:
        for r, i in enumerate(np.argsort(-s), 1):
            out[i] += 1 / (k + r)
    return out

def ndcg(order, rel, k=10):
    dcg = sum(g / math.log2(r + 1) for r, i in enumerate(order[:k], 1) if (g := rel.get(i, 0)))
    ideal = sum(g / math.log2(r + 1)
                for r, g in enumerate(sorted(rel.values(), reverse=True)[:k], 1))
    return dcg / ideal if ideal else 0

Q = []
for c, d in codes:                              # K：编号查询
    Q.append(("K", [c], {d: 3}))
for d in rng.permutation(N)[:60]:               # P：改写查询，与金标文档零词面重叠
    d = int(d)
    have, q = set(docs[d]), []
    for cl in meta[d]:
        q += [w for w in TOPIC[cl] if w not in have][:2]
    if len(q) >= 3:
        sib = {i: 1 for i in range(N)                    # 同主题相邻文档只算部分相关
               if {meta[i][0], meta[i][1]} == {meta[d][0], meta[d][1]} and i != d}
        Q.append(("P", q, {**sib, d: 3}))

print("%-6s %-7s %-9s %-7s %-7s" % ("type", "BM25", "Dense32", "RRF60", "随机"))
for t in ("K", "P"):
    rel_idx = [i for i, x in enumerate(Q) if x[0] == t]
    row = {"BM25": [], "Dense32": [], "RRF60": [], "随机": []}
    for i in rel_idx:
        _, q, rel = Q[i]
        for name, s in (("BM25", bm25(q)), ("Dense32", dense(q)),
                        ("RRF60", rrf(bm25(q), dense(q))), ("随机", rng.random(N))):
            row[name].append(ndcg(np.argsort(-s), rel))
    print("%-6s " % t + " ".join("%-9.4f" % np.mean(row[k]) for k in row))
print("查询 %d 条（K %d / P %d），语料 %d 篇，词表 %d"
      % (len(Q), sum(1 for x in Q if x[0] == "K"), sum(1 for x in Q if x[0] == "P"), N, V))
```

本机实跑输出（Python 3.11.15 / numpy 2.4.6，约 2 秒）：

```
type   BM25    Dense32   RRF60   随机
K      1.0000  0.5804    0.9257  0.0100
P      0.2327  0.4344    0.3519  0.0246
查询 110 条（K 50 / P 60），语料 400 篇，词表 130
```

按 50/60 的查询条数加权，三种方法的总平均是 **0.6127 / 0.5815 / 0.5008**（RRF / BM25 / 稠密）——和完整版的结论一致：**融合总平均最高，但在每一类查询上都不是第一。** 编号类 BM25 1.0000 对融合 0.9257；改写类稠密 0.4344 对融合 0.3519。

## 💡 三条能带走的

1. **稀疏和稠密不是替代关系，是两种不同的失效模式。** 稀疏在"这个词我没见过"上失效——改写、同义、跨语言；稠密在"这个词太罕见"上失效——编号、人名、错误码、版本号。混合检索的本质，是让这两种失效不要同时发生。
2. **融合是平均，平均削峰。** 28.5% 的查询上融合输给最佳单路。如果你的查询分布明显偏斜（一个内部工单系统，九成查询带单号），**直接路由**——先用规则或小分类器判断查询类型，再选一路——可能比融合更划算，而且更便宜。
3. **融合器对坏分支零容忍。** 上线前必须单独量每一路的 top-k 质量，并且**监控一路退化**：换编码器、改分词、索引没跟上，都会让一路悄悄变成噪声。那时候混合检索不是保险丝，是放大器——0.7249 掉到 0.2004 只需要一路变随机。

**这一篇的盲区**：合成语料没有真实语义，同簇文档差异太小（改写查询的绝对水平不可参考）；稠密路是 LSA 不是预训练编码器，32 维只占平方谱能量 38.98%；编号 df=1 是我构造的极值——真实库里同一编号常出现在多篇文档（同一商品多个页面），IDF 会降，BM25 的优势会缩小。要判断你自己的场景，只能拿你自己的查询日志跑一遍后悔值那张表。
