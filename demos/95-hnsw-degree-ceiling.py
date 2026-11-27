#!/usr/bin/env python3
"""
HNSW：召回天花板由图的度数决定，还是由搜索预算决定？
纯 numpy + Python heapq 的真实实现：分层图构建、ef_search 扫描、连通块统计。
数据为合成重叠簇（非真实 embedding），文中已声明。
"""
import heapq
import json
import os
import time

import numpy as np

rng = np.random.default_rng(20261127)

N = int(os.environ.get("N", 5000))
D, NC = 64, 50
NQ, K = 400, 10

# ---- 数据：50 个重叠簇，簇内散度接近簇间距，最近邻不唯一 ----
centers = rng.normal(size=(NC, D))
centers /= np.linalg.norm(centers, axis=1, keepdims=True)
spread = rng.uniform(1.1, 1.5, size=(NC, 1))
labels = rng.integers(0, NC, size=N)
GAP = 0.55                                     # 簇心之间拉开的程度
X = GAP * centers[labels] + spread[labels] * rng.normal(size=(N, D)) / np.sqrt(D)
X /= np.linalg.norm(X, axis=1, keepdims=True)

qidx = rng.choice(N, NQ, replace=False)
Q = X[qidx] + 0.25 * rng.normal(size=(NQ, D)) / np.sqrt(D)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

t = time.time()
DM = (2.0 - 2.0 * (X @ X.T)).astype(np.float32)      # 全量两两距离（真实值）
QD = (2.0 - 2.0 * (Q @ X.T)).astype(np.float32)
GT = np.argsort(QD, axis=1)[:, :K]
n1 = np.sqrt(np.take_along_axis(QD, GT[:, :1], 1)).mean()
n10 = np.sqrt(np.take_along_axis(QD, GT[:, 9:10], 1)).mean()
print(f"N={N} D={D} 簇={NC} 查询={NQ} recall@{K} | 1-NN 距离 {n1:.3f} "
      f"10-NN 距离 {n10:.3f} | 距离矩阵 {DM.nbytes/1e6:.0f}MB {time.time()-t:.1f}s")


class HNSW:
    def __init__(self, M, efC):
        self.M, self.M0, self.efC = M, 2 * M, efC
        self.mL = 1.0 / np.log(max(M, 2))
        self.graph, self.level = [], []
        self.entry, self.max_level = None, -1
        self.dc = 0

    def search_layer(self, row, eps, ef, layer, count=True):
        visited = set(eps)
        res, cand = [], []
        for e in eps:
            d = row[e]
            res.append((-d, e))
            cand.append((d, e))
        heapq.heapify(res)
        heapq.heapify(cand)
        while cand:
            d, c = heapq.heappop(cand)
            if len(res) >= ef and d > -res[0][0]:
                break
            g = self.graph[c][layer]
            for e in g:
                if e in visited:
                    continue
                visited.add(e)
                if count:
                    self.dc += 1
                dd = row[e]
                if len(res) < ef or dd < -res[0][0]:
                    heapq.heappush(cand, (dd, e))
                    heapq.heappush(res, (-dd, e))
                    if len(res) > ef:
                        heapq.heappop(res)
        return sorted([(-d, e) for d, e in res])

    def select(self, cands, cap, q):
        chosen = []
        for d, i in cands:
            if len(chosen) >= cap:
                break
            if not chosen or (DM[chosen, i] > d).all():
                chosen.append(i)
        if len(chosen) < cap:
            s = set(chosen)
            for _, i in cands:
                if i not in s:
                    chosen.append(i)
                    s.add(i)
                if len(chosen) >= cap:
                    break
        return chosen

    def shrink(self, node, layer, cap):
        nb = self.graph[node][layer]
        if len(nb) <= cap:
            return
        order = np.argsort(DM[node, nb])
        cands = [(float(DM[node, nb[o]]), nb[o]) for o in order]
        self.graph[node][layer] = self.select(cands, cap, node)

    def insert(self, i, row=None):
        row = DM[i].tolist() if row is None else row
        lvl = int(-np.log(rng.random()) * self.mL)
        self.graph.append([[] for _ in range(lvl + 1)])
        self.level.append(lvl)
        if self.entry is None:
            self.entry, self.max_level = i, lvl
            return
        ep = [self.entry]
        for layer in range(self.max_level, lvl, -1):
            ep = [self.search_layer(row, ep, 1, layer)[0][1]]
        for layer in range(min(lvl, self.max_level), -1, -1):
            W = self.search_layer(row, ep, self.efC, layer)
            cap = self.M0 if layer == 0 else self.M
            nb = self.select(W, cap, i)
            self.graph[i][layer] = nb
            for j in nb:
                self.graph[j][layer].append(i)
                self.shrink(j, layer, cap)
            ep = [e for _, e in W]
        if lvl > self.max_level:
            self.entry, self.max_level = i, lvl

    def query(self, q, ef):
        row = QD[q].tolist()
        ep = [self.entry]
        for layer in range(self.max_level, 0, -1):
            ep = [self.search_layer(row, ep, 1, layer, count=False)[0][1]]
        return [e for _, e in self.search_layer(row, ep, ef, 0)[:K]]


def build(M, efC):
    g = HNSW(M, efC)
    for i in range(N):
        g.insert(i)
    return g


def evaluate(g, ef):
    hits = 0
    g.dc = 0
    for q in range(NQ):
        hits += len(set(g.query(q, ef)) & set(GT[q].tolist()))
    return hits / (NQ * K), g.dc / NQ


def components(g):
    par = list(range(N))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for i in range(N):
        for j in g.graph[i][0]:
            ra, rb = find(i), find(j)
            if ra != rb:
                par[ra] = rb
    return len({find(i) for i in range(N)})


if __name__ == "__main__":
    out, t0 = {}, time.time()
    EFS = (10, 32, 128, 512)
    rows = {}
    for M in (4, 8, 16, 32):
        tb = time.time()
        g = build(M, 64)
        cc = components(g)
        deg = np.array([len(g.graph[i][0]) for i in range(N)])
        print(f"\nM={M:2d} 构建 {time.time()-tb:5.1f}s | 层0 平均度 {deg.mean():5.1f} "
              f"| 孤立 {int((deg==0).sum())} | 连通块 {cc} | 最高层 {g.max_level}")
        rows[M] = {}
        for ef in EFS:
            r, per_q = evaluate(g, ef)
            rows[M][ef] = dict(recall=round(r, 4), dist_per_q=round(per_q, 1))
            print(f"   ef={ef:4d}  recall@{K}={r:.4f}  每查询距离计算={per_q:8.1f}"
                  f"   (穷举={N})")
        rows[M]["comps"] = cc
        rows[M]["mean_deg"] = round(float(deg.mean()), 1)
    out["grid"] = {str(k): v for k, v in rows.items()}
    out["elapsed_s"] = round(time.time() - t0, 1)
    print("\n" + json.dumps(out, ensure_ascii=False))
    print(f"\n总用时 {out['elapsed_s']}s")
