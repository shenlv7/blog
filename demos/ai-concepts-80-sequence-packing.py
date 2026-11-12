#!/usr/bin/env python3
"""序列打包（sequence packing）的两本账：padding 浪费 与 跨文档注意力。

实验 A：给定一份真实感的长尾长度分布，算四种 batching 策略下
        "真实 token / 实际前向 token" 的利用率，以及总 token 数。
实验 B：把 8 篇文档拼成一条 2048 token 的序列，用真实的因果注意力
        权重（随机投影的 Q/K，numpy）量：不加文档边界掩码时，
        有多少注意力质量落在别的文档上。

不涉及任何训练，也不需要 GPU —— 所有数字都是可复算的算术/矩阵乘。
"""
import numpy as np

RNG = np.random.default_rng(7)

# ---------------- 实验 A：padding 浪费 ----------------
N_SAMPLES = 20000
BIN = 4096          # 打包后的定长窗口（训练序列长度）


def lengths(n):
    """长尾长度分布：中位数 280 token，σ=0.85，裁到 [40, 4096]。"""
    L = RNG.lognormal(mean=np.log(280), sigma=0.85, size=n)
    return np.clip(np.round(L), 40, BIN).astype(int)


L = lengths(N_SAMPLES)


def pad_cost(order, bs):
    """按给定顺序、每 bs 条一组，各自补齐到组内最长。"""
    order = np.asarray(order)
    total = 0
    for s in range(0, len(order), bs):
        chunk = L[order[s:s + bs]]
        total += chunk.max() * len(chunk)
    return total


def pack_cost(order):
    """顺序装进 BIN 大小的箱子，装满就换箱（每个箱按 BIN 计费）。"""
    total, used = 0, 0
    for i in order:
        if used + L[i] > BIN:
            total += BIN
            used = 0
        used += L[i]
    return total + (BIN if used else 0)


def pack_ffd(order):
    """First-Fit Decreasing：长样本先放，塞进第一个装得下的箱子。"""
    bins = []
    for i in order:
        ln = L[i]
        for b, room in enumerate(bins):
            if room >= ln:
                bins[b] -= ln
                break
        else:
            bins.append(BIN - ln)
    return BIN * len(bins)


real = int(L.sum())
rand = RNG.permutation(N_SAMPLES)
by_len = np.argsort(L)
desc = np.argsort(-L)
# 有限乱序：把长度排序的样本切成 200 条一桶，桶内打乱再排序 —— 近似真实实现
bucket = []
for s in range(0, N_SAMPLES, 200):
    seg = by_len[s:s + 200]
    bucket.append(seg[RNG.permutation(len(seg))])
bucketed = np.concatenate(bucket)
# 桶内乱序后，再按 micro-batch 内长度排序（真实框架里就是这样做的）
rows_bucket = []
for s in range(0, N_SAMPLES, 8):
    rows_bucket.extend(sorted(bucketed[s:s + 8], key=lambda i: L[i]))
bucketed = np.array(rows_bucket)

print("=== 实验 A：20000 条样本，真实 token 共 {:,} ===".format(real))
print(f"{'策略':<32}{'前向 token':>14}{'利用率':>9}{'是真实量':>10}")
rows = []
bs_list = [1, 2, 4, 8, 16, 32, 64]
for bs in bs_list:
    c = pad_cost(rand, bs)
    rows.append((f"朴素 batching, bs={bs}", c))
for bs in (8, 32, 64):
    rows.append((f"长度排序 batching, bs={bs}", pad_cost(by_len, bs)))
rows.append(("桶内乱序+组内排序, bs=8", pad_cost(bucketed, 8)))
rows.append((f"贪心打包, bin={BIN}", pack_cost(desc)))
rows.append((f"FFD 打包, bin={BIN}", pack_ffd(desc)))
for name, cost in rows:
    print(f"{name:<32}{cost:>14,}{real/cost:>9.1%}{cost/real:>10.2f}x")

# 分位数，说明长尾有多长
q = np.percentile(L, [50, 90, 99, 100]).astype(int)
print("\n长度分位 P50/P90/P99/max：", q, " 均值", round(L.mean(), 1))

# ---------------- 实验 B：跨文档注意力泄漏 ----------------
D_DOC = 8
DOC_LENS = np.array([64, 96, 128, 160, 224, 288, 416, 672])   # 合计 2048
TOTAL = int(DOC_LENS.sum())
D_MODEL, D_HEAD = 64, 32

# 每篇文档有自己的语义方向；token = 文档方向 + 噪声（带一点文档内结构）
doc_dir = RNG.normal(size=(D_DOC, D_MODEL))
tok = np.zeros((TOTAL, D_MODEL))
doc_of = np.zeros(TOTAL, dtype=int)
p = 0
for d, ln in enumerate(DOC_LENS):
    tok[p:p + ln] = doc_dir[d] + RNG.normal(scale=0.35, size=(ln, D_MODEL))
    doc_of[p:p + ln] = d
    p += ln

Wq = RNG.normal(scale=1 / np.sqrt(D_MODEL), size=(D_MODEL, D_HEAD))
Wk = RNG.normal(scale=1 / np.sqrt(D_MODEL), size=(D_MODEL, D_HEAD))
Q, K = tok @ Wq, tok @ Wk


def attn_rows(mask_doc):
    s = Q @ K.T / np.sqrt(D_HEAD)
    causal = np.tril(np.ones((TOTAL, TOTAL), dtype=bool))
    allowed = causal & (doc_of[:, None] == doc_of[None, :]) if mask_doc else causal
    s = np.where(allowed, s, -np.inf)
    s = s - s.max(axis=1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(axis=1, keepdims=True)


def leak_report(P):
    foreign = doc_of[:, None] != doc_of[None, :]
    per_pos = (P * foreign).sum(axis=1)          # 每个查询位置的外来注意力占比
    per_doc = [per_pos[doc_of == d].mean() for d in range(D_DOC)]
    first = [per_pos[np.where(doc_of == d)[0][0]] for d in range(D_DOC)]
    return per_doc, first


def visible_foreign():
    """纯计数：每个位置上，可见键中外来键的比例（无掩码时的确定性事实）。"""
    pos = np.arange(TOTAL)
    starts = np.concatenate([[0], np.cumsum(DOC_LENS)[:-1]])[doc_of]
    first_idx = np.array([np.where(doc_of == d)[0][0] for d in range(D_DOC)])
    return starts / (pos + 1), starts[first_idx] / (first_idx + 1)


vis_mean, vis_first = visible_foreign()
P_nomask = attn_rows(False)
P_mask = attn_rows(True)
doc_leak, first_leak = leak_report(P_nomask)
mask_leak, _ = leak_report(P_mask)

print(f"\n=== 实验 B：{D_DOC} 篇文档拼成 1 条 {TOTAL} token，因果注意力 ===")
print(f"{'文档':<5}{'长度':>6}{'起点':>7}{'可见键外来占比':>15}{'首token可见外来':>16}{'实测外来注意力':>14}")
start = 0
for d in range(D_DOC):
    print(f"{d:<5}{DOC_LENS[d]:>6}{start:>7}{np.mean(vis_mean[doc_of==d]):>14.1%}"
          f"{vis_first[d]:>15.1%}{doc_leak[d]:>13.1%}")
    start += DOC_LENS[d]

foreign = doc_of[:, None] != doc_of[None, :]
print(f"\n无边界掩码：可见键里 {np.mean(vis_mean):.1%} 属于别的文档；"
      f"实测这些键分走 {float((P_nomask*foreign).sum()/TOTAL):.1%} 的注意力质量")
print(f"加边界掩码：实测外来注意力质量 {float((P_mask*foreign).sum()/TOTAL):.2e}（应为 0）")
print(f"最后一篇的 token 平均 {np.mean(vis_mean[doc_of==7]):.1%} 的可见上下文是前面 7 篇；"
      f"它的首 token 是 {vis_first[7]:.1%}")

M = TOTAL * TOTAL
print(f"\n掩码本身的开销：把 {TOTAL}×{TOTAL} 的 bool 掩码真正建出来 = {M*1/2**20:.1f} MiB / 层，"
      f"打包成 bit = {M/8/2**20:.1f} MiB / 层；cu_seqlens 式变长核对数表只要 {D_DOC} 个 int32。")
