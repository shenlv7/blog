"""68 · Product Quantization: one byte per subvector.

numpy-only, single machine. Measures recall@10 for exact float32 vs PQ-ADC
vs PQ-ADC + exact rescoring, and a 1-bit binary quantizer at the same byte
budget. Not a GPU benchmark.
"""
import time
import numpy as np

RNG = np.random.default_rng(20261031)
D, N, NC = 128, 20000, 200
NQ, K = 300, 10
TRAIN = 5000
ITERS = 12
NOISE = 1.10        # within-cluster spread: higher = harder top-10

# ---------------------------------------------------------------- dataset
centers = RNG.normal(size=(NC, D))
assign = RNG.integers(0, NC, N)
dim_scale = np.exp(RNG.normal(0.0, 0.55, D))          # anisotropic dims
X = (centers[assign] + RNG.normal(0, NOISE, (N, D))) * dim_scale
q_idx = RNG.choice(N, NQ, replace=False)
Q = X[q_idx] + RNG.normal(0, 0.45, (NQ, D)) * dim_scale
X = X.astype(np.float32)
Q = Q.astype(np.float32)


def topset(a, k):
    return np.argsort(a, axis=1)[:, :k]


# ---------------------------------------------------------------- ground truth
def l2(Q, X):
    return (Q * Q).sum(1)[:, None] - 2 * Q @ X.T + (X * X).sum(1)[None, :]


t = time.time()
gt = topset(l2(Q, X), K)
t_exact = (time.time() - t) / NQ * 1e3
print(f"exact float32: {D*4} B/vec, top-{K} by brute force, {t_exact:.2f} ms/query")


def recall(found, gt):
    return float(np.mean([len(set(a) & set(b)) / K for a, b in zip(found, gt)]))


# ---------------------------------------------------------------- k-means
def kmeans(pts, k, iters=ITERS, rng=None):
    n = len(pts)
    C = pts[rng.choice(n, k, replace=False)].copy()
    for _ in range(iters):
        d = (pts * pts).sum(1)[:, None] - 2 * pts @ C.T + (C * C).sum(1)[None, :]
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = pts[m].mean(0)
    d = (pts * pts).sum(1)[:, None] - 2 * pts @ C.T + (C * C).sum(1)[None, :]
    return C, float(d.min(1).mean())


# ---------------------------------------------------------------- PQ
def pq_train(X, m, rng):
    sub = D // m
    codes = np.empty((len(X), m), dtype=np.uint8)
    cbs, distort = [], 0.0
    for j in range(m):
        pts = X[:TRAIN, j * sub:(j + 1) * sub]
        C, dist = kmeans(pts, 256, rng=rng)
        cbs.append(C)
        full = np.empty((len(X), sub), dtype=np.float32)
        for s in range(0, len(X), 5000):
            blk = X[s:s + 5000, j * sub:(j + 1) * sub]
            d = (blk * blk).sum(1)[:, None] - 2 * blk @ C.T + (C * C).sum(1)[None, :]
            codes[s:s + 5000, j] = d.argmin(1)
            full[s:s + 5000] = C[codes[s:s + 5000, j]]
        distort += float(((pts - C[codes[:TRAIN, j]]) ** 2).sum(1).mean())
    return cbs, codes, distort


def pq_search(Q, cbs, codes, pool_n=200):
    m = len(cbs)
    sub = D // m
    out = np.empty((len(Q), pool_n), dtype=np.int32)
    for i in range(len(Q)):
        lut = np.empty((m, 256), dtype=np.float32)
        for j, C in enumerate(cbs):
            q = Q[i, j * sub:(j + 1) * sub]
            lut[j] = (C * C).sum(1) - 2 * C @ q
        dist = lut[np.arange(m)[None, :], codes].sum(1)
        cand = np.argpartition(dist, pool_n)[:pool_n]
        out[i] = cand[np.argsort(dist[cand])]          # ADC order, sorted
    return out


def rerank(pool, Q, X):
    out = np.empty((len(Q), K), dtype=np.int32)
    for i in range(len(Q)):
        cand = pool[i]
        d = l2(Q[i:i + 1], X[cand])[0]
        out[i] = cand[np.argsort(d)[:K]]
    return out


print("\n== PQ: d=128, k=256 per subvector (1 byte each) ==")
print(f"{'m':>4} {'B/vec':>6} {'compress':>9} {'MSE/sub':>8} {'R@10':>7} "
      f"{'cov@200':>8} {'+rerank':>8} {'ms/q':>6}")
results = {}
for m in (8, 16, 32, 64):
    t = time.time()
    cbs, codes, dist = pq_train(X, m, np.random.default_rng(7))
    t_train = time.time() - t
    t = time.time()
    pool = pq_search(Q, cbs, codes)
    t_search = (time.time() - t) / NQ * 1e3
    r = recall(pool[:, :K], gt)
    cov = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(pool, gt)]))
    rr = recall(rerank(pool, Q, X), gt)
    results[m] = (r, rr, t_search)
    print(f"{m:>4} {m:>6} {D*4/m:>8.0f}x {dist/D:>8.3f} {r:>7.3f} "
          f"{cov:>8.3f} {rr:>8.3f} {t_search:>6.2f}   (train {t_train:.1f}s)")

# ---------------------------------------------------------------- binary (1 bit/dim)
print("\n== 1-bit binary quantizer, same 16-byte budget as m=16 PQ ==")
med = np.median(X, axis=0)
Bx = (X > med).astype(np.uint8) * 2 - 1
Bq = (Q > med).astype(np.uint8) * 2 - 1
t = time.time()
hamming = -Bq @ Bx.T.astype(np.int16)          # bigger = closer
b_pool = topset(-hamming.astype(np.float32), 200)
b_ms = (time.time() - t) / NQ * 1e3
b_cov = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(b_pool, gt)]))
print(f"R@10 {recall(b_pool[:, :K], gt):.3f}  cov@200 {b_cov:.3f}  "
      f"+rerank {recall(rerank(b_pool, Q, X), gt):.3f}  "
      f"{D/8:.0f} B/vec, {b_ms:.2f} ms/q")

# ---------------------------------------------------------------- pool size curve
print("\n== cov@200=1.000 is not free: how small can the pool be? (m=16, 16 B) ==")
cbs, codes, _ = pq_train(X, 16, np.random.default_rng(7))
pool = pq_search(Q, cbs, codes)
for p in (10, 20, 50, 100, 200):
    sub = pool[:, :p]
    cov = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(sub, gt)]))
    rr = recall(rerank(sub, Q, X), gt)
    print(f"  pool {p:>3}: cov {cov:.3f}  rerank R@10 {rr:.3f}")

# ---------------------------------------------------------------- variance-balanced grouping
print("\n== grouping matters: random vs variance-balanced (m=16) ==")
var = ((X[:TRAIN] - X[:TRAIN].mean(0)) ** 2).mean(0)
order = np.argsort(-var)
bal = [order[i::16] for i in range(16)]        # round-robin over sorted dims
Xp = np.concatenate([X[:, b] for b in bal], axis=1)
Qp = np.concatenate([Q[:, b] for b in bal], axis=1)
Xb, Qb = X, Q
X, Q = Xp, Qp
cbs, codes, dist = pq_train(X, 16, np.random.default_rng(7))
pool = pq_search(Q, cbs, codes)
print(f"balanced: MSE/sub {dist/D:.3f}  R@10 {recall(pool[:, :K], gt):.3f}  "
      f"+rerank {recall(rerank(pool, Qb, Xb), gt):.3f}")
