#!/usr/bin/env python3
"""向量检索里的元数据过滤：先过滤和后过滤差多少（纯 numpy，无向量库依赖）。

库：20000 条向量，64 维，100 个语义簇，每条带一个 tenant 字段（50 个租户，均匀分布 → 每个 2%）。
索引：IVF，100 个倒排列表（k-means 3 轮）。
查询：每条查询带自己的 tenant，真值 = 「同租户内」精确 top-10。

比较四种做法，指标是 recall@10 与扫描的点积次数：
  A 后过滤：全库 IVF（nprobe=8）取 top-10，再丢掉不匹配租户的
  B 后过滤 + 过采样：全库取 top-100 / top-500，再过滤
  C 前过滤 + 暴力：直接扫同租户的 ~400 条
  D 前过滤 + IVF：只扫命中的列表，凑满 10 条
"""
import numpy as np

N, D, NC, NTEN, NQ = 20000, 64, 100, 50, 200
NLIST, NPROBE, K = 100, 8, 10
NOISE, SEED = 1.1, 11

rng = np.random.default_rng(SEED)
centers = rng.normal(size=(NC, D)); centers /= np.linalg.norm(centers, axis=1, keepdims=True)
lab = rng.integers(0, NC, N)
X = centers[lab] + NOISE * rng.normal(size=(N, D))
X /= np.linalg.norm(X, axis=1, keepdims=True)
tenant = rng.integers(0, NTEN, N)

# ---- IVF：k-means 3 轮 ----
c = X[rng.choice(N, NLIST, replace=False)].copy()
for _ in range(3):
    assign = (X @ c.T).argmax(1)
    for j in range(NLIST):
        m = assign == j
        if m.any():
            c[j] = X[m].mean(0)
lists = [np.where(assign == j)[0] for j in range(NLIST)]

q_cl = rng.integers(0, NC, NQ)
Q = centers[q_cl] + NOISE * rng.normal(size=(NQ, D))
Q /= np.linalg.norm(Q, axis=1, keepdims=True)
q_tenant = rng.integers(0, NTEN, NQ)


def exact(q, t):
    idx = np.where(tenant == t)[0]
    sim = X[idx] @ q
    top = np.argsort(-sim)[:K]
    return idx[top], len(idx)


truth, cost_truth = [], 0
for i in range(NQ):
    tr, n = exact(Q[i], q_tenant[i]); truth.append(tr); cost_truth += n


def recall(found):
    return np.mean([len(set(f) & set(t)) / K for f, t in zip(found, truth)])


def post_filter(kk):
    found, cost = [], 0
    for i in range(NQ):
        q = Q[i]
        ls = np.argsort(-(q @ c.T))[:NPROBE]
        cand = np.concatenate([lists[j] for j in ls])
        sim = X[cand] @ q
        cost += len(cand)
        pick = cand[np.argsort(-sim)[:kk]]
        keep = pick[tenant[pick] == q_tenant[i]][:K]
        found.append(keep)
    return found, cost


print(f"库 {N} 条 / {NTEN} 租户（每租户 {100/NTEN:.0f}% = {N//NTEN} 条）/ {NQ} 条查询 / top-{K}")
print(f"{'做法':<22} {'recall@10':>10} {'扫描点积/查询':>14}")
print(f"{'C 前过滤+暴力（上界）':<22} {1.0:>10.4f} {cost_truth/NQ:>14.0f}")
for kk, name in ((10, "A 后过滤 top-10"), (100, "B 后过滤 过采样100"), (500, "B 后过滤 过采样500")):
    f, cost = post_filter(kk)
    print(f"{name:<22} {recall(f):>10.4f} {cost/NQ:>14.0f}")

# D 前过滤 + IVF：按列表顺序扫，命中租户才算，凑满 K 或扫完
found, cost = [], 0
for i in range(NQ):
    q = Q[i]
    ls = np.argsort(-(q @ c.T))[:NPROBE]
    got, scanned = [], 0
    for j in ls:
        ids = lists[j]; ids = ids[tenant[ids] == q_tenant[i]]
        if not len(ids):
            continue
        sim = X[ids] @ q; scanned += len(ids)
        order = ids[np.argsort(-sim)]
        got.extend(order.tolist())
        if len(got) >= K:
            break
    cost += scanned
    found.append(np.array(got[:K]))
print(f"{'D 前过滤+IVF(nprobe=8)':<22} {recall(found):>10.4f} {cost/NQ:>14.0f}")

# 过滤选择性扫描：租户数从 2 到 2000
print("\n租户数（过滤选择性）对后过滤召回的影响")
for nt in (10, 50, 200, 1000, 5000):
    t2 = rng.integers(0, nt, N); q2 = rng.integers(0, nt, NQ)
    truth2, costs = [], 0
    for i in range(NQ):
        idx = np.where(t2 == q2[i])[0]
        if len(idx) == 0:
            truth2.append(np.array([], int)); continue
        sim = X[idx] @ Q[i]; truth2.append(idx[np.argsort(-sim)[:K]]); costs += len(idx)
    f10, f100 = [], []
    for i in range(NQ):
        ls = np.argsort(-(Q[i] @ c.T))[:NPROBE]
        cand = np.concatenate([lists[j] for j in ls]); sim = X[cand] @ Q[i]
        order = cand[np.argsort(-sim)]
        f10.append(np.array([x for x in order[:10] if t2[x] == q2[i]]))
        f100.append(np.array([x for x in order[:100] if t2[x] == q2[i]][:K]))
    r = lambda F: np.mean([len(set(a) & set(b)) / max(1, len(b)) for a, b in zip(F, truth2)])
    print(f"  租户={nt:5d}（选择性 {100/nt:5.2f}%） 后过滤top10={r(f10):.4f}  "
          f"过采样100={r(f100):.4f}  前过滤暴力={costs/NQ:.0f} 点积/查询")
