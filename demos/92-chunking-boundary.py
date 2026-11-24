"""92 · 分块边界：固定切块把答案切成两半，overlap 用索引体积换回来
离线模拟：TF-IDF 代替 embedding，只为测量「分块结构」本身造成的损失，非实测某模型的表现。
"""
import os
import numpy as np
from collections import Counter, defaultdict

rng = np.random.default_rng(int(os.environ.get("SEED", "20261124")))

FILLER = ["the", "module", "reports", "nominal", "throughput", "operators", "verified",
          "housing", "alignment", "during", "commissioning", "under", "load", "sustained",
          "thermal", "margin", "stayed", "above", "baseline", "field", "engineers", "noted",
          "vibration", "levels", "within", "spec", "shift", "startup", "sequence", "completed"]

def filler_sent():
    return list(rng.choice(FILLER, size=int(rng.integers(6, 11))))

# ---------- 语料：120 个模块，每个 5 条事实（attr -> value），每篇约 200 token ----------
N_DOC = 120
docs, facts = [], []
for d in range(N_DOC):
    attrs = [f"a{d*5+k}" for k in range(5)]
    vals = [f"v{d*5+k}" for k in range(5)]
    n_sent = 23
    slots = sorted(rng.choice(np.arange(1, n_sent), size=5, replace=False))
    slot_map = {s: k for k, s in enumerate(slots)}
    sents = []
    for i in range(n_sent):
        if i in slot_map:
            k = slot_map[i]
            toks = ["the", attrs[k], "threshold", "resolves", "to", vals[k], "under", "sustained", "load"]
            facts.append((d, len(sents), attrs[k], vals[k], toks))
            sents.append(toks)
        else:
            sents.append(filler_sent())
    docs.append(sents)

# 一份「交叉索引」文档：把 attr 名密集列表，不含任何 value
attrs_all = [f[2] for f in facts]
index_doc = [["cross", "reference", "table"] + attrs_all[i:i + 20] for i in range(0, len(attrs_all), 20)]
docs.append(index_doc)
IDX_DOC = len(docs) - 1

# ---------- 切块 ----------
def chunk_fixed(toks, size, overlap):
    return [(i, min(i + size, len(toks))) for i in range(0, len(toks), max(1, size - overlap))]

def chunk_aligned(sents, target):
    out, cur, start, pos = [], [], 0, 0
    for s in sents:
        if cur and len(cur) + len(s) > target:
            out.append((start, pos)); cur, start = [], pos
        cur += s; pos += len(s)
    if cur:
        out.append((start, pos))
    return out

def build(strategy):
    chunks = []
    for d, sents in enumerate(docs):
        toks = [t for s in sents for t in s]
        spans = chunk_fixed(toks, strategy[1], strategy[2]) if strategy[0] == "f" else chunk_aligned(sents, strategy[1])
        for a, b in spans:
            chunks.append((d, toks[a:b]))
    return chunks

def contains(sub, seq):
    n = len(sub)
    return any(seq[i:i + n] == sub for i in range(len(seq) - n + 1))

# 倒排索引 + 余弦（idf 由块级 df 算出）
def index(chunks):
    df = Counter()
    for _, toks in chunks:
        df.update(set(toks))
    N = len(chunks)
    idf = {t: float(np.log((N + 1) / (c + 1)) + 1.0) for t, c in df.items()}
    vecs, norms = [], []
    for _, toks in chunks:
        tf = Counter(toks)
        v = {t: c * idf.get(t, 1.0) for t, c in tf.items()}
        vecs.append(v); norms.append(float(np.linalg.norm(list(v.values()))))
    return idf, vecs, np.array(norms)

def search(q, chunks, idf, vecs, norms):
    qv = {t: idf[t] for t in q if t in idf}
    qn = float(np.linalg.norm(list(qv.values()))) if qv else 0.0
    if qn == 0:
        return list(range(len(chunks)))
    sims = np.zeros(len(chunks))
    for i, v in enumerate(vecs):
        if norms[i] == 0:
            continue
        dot = sum(w * v.get(t, 0.0) for t, w in qv.items())
        sims[i] = dot / (qn * norms[i])
    return list(np.argsort(-sims))

def dedup(rank, chunks, k):
    out, seen = [], set()
    for i in rank:
        if chunks[i][0] in seen:
            continue
        seen.add(chunks[i][0]); out.append(i)
        if len(out) == k:
            break
    return out

by_attr = {f[2]: f for f in facts}
PAIRS = [(f"a{d*5}", f"a{(d+1)*5}") for d in range(0, N_DOC - 1, 4)]   # 30 组跨篇双跳

print(f"{N_DOC} 篇文档 / {len(facts)} 条事实 / 索引文档 1 份 / 单跳 {len(facts)} 次，双跳 {len(PAIRS)} 次\n")
print(f"{'策略':<18}{'块数':>5}{'块均长':>7}{'索引token':>10}{'切断':>6}{'话题@5':>7}{'答案@5':>7}{'Top1':>6}"
      f"{'top5上下文':>9}{'双跳@3':>7}{'去重@3':>7}{'重复槽':>7}")

results = {}
for strategy, label in [(("f", 64, 0), "固定64/o0"), (("f", 64, 16), "固定64/o16"),
                        (("f", 64, 32), "固定64/o32"), (("f", 128, 0), "固定128/o0"),
                        (("f", 128, 64), "固定128/o64"), (("f", 256, 0), "固定256/o0"),
                        (("a", 96), "句界对齐~96")]:
    chunks = build(strategy)
    idf, vecs, norms = index(chunks)
    tok = sum(len(t) for _, t in chunks)

    broken = sum(not any(contains(f[4], t) for cd, t in chunks if cd == f[0]) for f in facts)
    r5 = t1 = idx_win = topic = ctx = 0
    for f in facts:
        rank = search(["what", "does", f[2], "resolve", "to"], chunks, idf, vecs, norms)
        r5 += any(contains(f[4], chunks[i][1]) for i in rank[:5])
        t1 += contains(f[4], chunks[rank[0]][1])
        idx_win += chunks[rank[0]][0] == IDX_DOC
        topic += any(f[2] in chunks[i][1] for i in rank[:5])
        ctx += sum(len(chunks[i][1]) for i in rank[:5])
    c3 = c3d = c5 = 0; dup = []
    for a1, a2 in PAIRS:
        rank = search(["which", "of", a1, a2, "is", "higher"], chunks, idf, vecs, norms)
        f1, f2 = by_attr[a1][4], by_attr[a2][4]
        def both(ids): return any(contains(f1, chunks[i][1]) for i in ids) and any(contains(f2, chunks[i][1]) for i in ids)
        c3 += both(rank[:3]); c5 += both(rank[:5]); c3d += both(dedup(rank, chunks, 3))
        dup.append(1 - len({chunks[i][0] for i in rank[:5]}) / 5)
    n = len(facts)
    print(f"{label:<18}{len(chunks):>5}{tok/len(chunks):>7.0f}{tok:>10}{broken/n:>6.1%}{topic/n:>7.3f}"
          f"{r5/n:>7.3f}{t1/n:>6.3f}{ctx/n:>9.0f}{c3/len(PAIRS):>7.3f}{c3d/len(PAIRS):>7.3f}{np.mean(dup):>7.2f}")
    results[label] = (chunks, idf, vecs, norms)

# 拆开看：固定64/o0 丢掉的那 18 条，是不是正好是被切断的那些
chunks, idf, vecs, norms = results["固定64/o0"]
lost_broken = lost_total = 0
for f in facts:
    rank = search(["what", "does", f[2], "resolve", "to"], chunks, idf, vecs, norms)
    intact = any(contains(f[4], t) for cd, t in chunks if cd == f[0])
    hit = any(contains(f[4], chunks[i][1]) for i in rank[:5])
    if not hit:
        lost_total += 1
        lost_broken += (not intact)
print(f"\n固定64/o0 的 {lost_total} 条未命中里，{lost_broken} 条是事实句被边界切断（{lost_broken/lost_total:.0%}）")
