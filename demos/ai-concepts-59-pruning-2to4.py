#!/usr/bin/env python3
"""59 · 剪枝：50% 权重剪成 0，准确率只掉一点——但矩阵乘法一次都没变快

纯 numpy，无 GPU，无外部数据集（5x7 点阵数字 + 抖动，现场生成）。
实验内容：
  1. 训练一个 36-64-32-10 的 MLP 当被测模型
  2. 非结构化幅度剪枝：50/75/90/95%
  3. 2:4 结构化剪枝（每 4 个权重留 2 个最大）
  4. Wanda（幅度 × 输入激活范数）对照幅度剪枝
  5. 计时：dense / masked-dense / gather 稀疏乘
"""
import time
import numpy as np

rng = np.random.default_rng(0)

# ---------- 数据：5x7 点阵数字 ----------
GLYPHS = {
    0: ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    3: ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}


def base_vec(d):
    return np.array([[float(c) for c in row] for row in GLYPHS[d]])


def shift(img, dy, dx):
    out = np.zeros_like(img)
    h, w = img.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = img[ys, xs]
    return out


def make_set(n, flip=0.06):
    X = np.zeros((n, 36), np.float32)
    y = np.zeros(n, np.int64)
    for i in range(n):
        d = int(rng.integers(10))
        img = shift(base_vec(d), int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
        img = (img + (rng.random(img.shape) < flip)) % 2
        X[i, :35] = img.ravel()
        y[i] = d
    return X, y


Xtr, ytr = make_set(6000)
Xte, yte = make_set(800)
print(f"data: train={Xtr.shape} test={Xte.shape} 正样本像素占比={Xtr.mean():.3f}")

# ---------- 36-64-32-10 MLP（Adam）----------
DIMS = [36, 96, 48, 10]
params = []
for a, b in zip(DIMS[:-1], DIMS[1:]):
    params.append([rng.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32),
                   np.zeros(b, np.float32)])


def forward(X, ps, masks=None, keep_acts=False):
    acts = [X]
    h = X
    for i, (W, b) in enumerate(ps):
        if masks is not None:
            W = W * masks[i]
        z = h @ W + b
        if i < len(ps) - 1:
            h = np.maximum(z, 0)
            if keep_acts:
                acts.append(h)
        else:
            h = z
    return h, acts


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def accuracy(ps, X, y, masks=None):
    logits, _ = forward(X, ps, masks)
    return float((logits.argmax(1) == y).mean())


def train(ps, X, y, epochs=120, bs=64, lr=1.5e-3, wd=1e-4, seed=0):
    m = [[np.zeros_like(W), np.zeros_like(b)] for W, b in ps]
    v = [[np.zeros_like(W), np.zeros_like(b)] for W, b in ps]
    r = np.random.default_rng(seed)
    step = 0
    n = len(X)
    for ep in range(epochs):
        idx = r.permutation(n)
        for s in range(0, n, bs):
            bi = idx[s:s + bs]
            xb, yb = X[bi], y[bi]
            logits, acts = forward(xb, ps, keep_acts=True)
            p = softmax(logits)
            g = p.copy()
            g[np.arange(len(bi)), yb] -= 1.0
            g /= len(bi)
            grads = []
            for i in range(len(ps) - 1, -1, -1):
                W, b = ps[i]
                gW = acts[i].T @ g + 2 * wd * W
                gb = g.sum(0)
                grads.insert(0, (gW, gb))
                if i > 0:
                    g = (g @ W.T) * (acts[i] > 0)
            step += 1
            for i in range(len(ps)):
                for j in range(2):
                    gr = grads[i][j]
                    m[i][j] = 0.9 * m[i][j] + 0.1 * gr
                    v[i][j] = 0.999 * v[i][j] + 0.001 * gr * gr
                    mh = m[i][j] / (1 - 0.9 ** step)
                    vh = v[i][j] / (1 - 0.999 ** step)
                    ps[i][j] -= lr * mh / (np.sqrt(vh) + 1e-8)
    return ps


t0 = time.time()
params = train(params, Xtr, ytr)
base_acc = accuracy(params, Xtr, ytr), accuracy(params, Xte, yte)
print(f"训练完成 {time.time()-t0:.1f}s  train acc={base_acc[0]:.4f}  test acc={base_acc[1]:.4f}")

# ---------- 全局非结构化幅度剪枝 ----------
def mag_masks(params, sparsity, scores=None):
    scores = scores or [np.abs(W) for W, _ in params]
    allv = np.concatenate([s.ravel() for s in scores])
    thr = np.quantile(allv, sparsity)
    return [(s >= thr).astype(np.float32) if sparsity > 0 else np.ones_like(s)
            for s in scores]


print("\n非结构化幅度剪枝（全局阈值，不重训）")
print(f"{'稀疏度':>8}{'test acc':>12}{'各层实际稀疏率':>34}")
for s in [0.0, 0.5, 0.75, 0.9, 0.95]:
    masks = mag_masks(params, s)
    per = [1 - m.mean() for m in masks]
    print(f"{s:>8.2f}{accuracy(params, Xte, yte, masks):>12.4f}"
          f"{'  '.join(f'L{i}:{p:.2f}' for i, p in enumerate(per)):>34}")

# ---------- 2:4 结构化剪枝 ----------
def nm24(scores):
    o, i = scores.shape
    if i % 4:
        return np.ones_like(scores, dtype=np.float32)
    g = scores.reshape(o, i // 4, 4)
    keep = np.zeros_like(g)
    top2 = np.argsort(-g, axis=2)[:, :, :2]
    np.put_along_axis(keep, top2, 1.0, axis=2)
    return keep.reshape(o, i).astype(np.float32)


def nm24_masks(params, scores=None):
    scores = scores or [np.abs(W) for W, _ in params]
    return [nm24(s) for s in scores]


print("\n2:4 结构化剪枝（每 4 个权重留 2 个）")
m24 = nm24_masks(params)
print(f"{'':>10}{'test acc':>12}{'各层稀疏率':>30}")
print(f"{'2:4 mag':>10}{accuracy(params, Xte, yte, m24):>12.4f}"
      f"{'  '.join(f'L{i}:{1-m.mean():.2f}' for i, m in enumerate(m24)):>30}")
m50 = mag_masks(params, 0.5)
print(f"{'50% 非结构':>10}{accuracy(params, Xte, yte, m50):>12.4f}"
      f"{'  '.join(f'L{i}:{1-m.mean():.2f}' for i, m in enumerate(m50)):>30}")

# ---------- Wanda：幅度 × 输入激活范数 ----------
_, acts_tr = forward(Xtr, params, keep_acts=True)
wanda_scores = []
for i, (W, _) in enumerate(params):
    col_norm = np.linalg.norm(acts_tr[i], axis=0) / np.sqrt(len(acts_tr[i]))
    wanda_scores.append(np.abs(W) * col_norm[:, None])

print("\n同一稀疏度下 幅度 vs Wanda")
print(f"{'稀疏度':>8}{'幅度':>10}{'Wanda':>10}")
for s in [0.5, 0.75, 0.9]:
    a = accuracy(params, Xte, yte, mag_masks(params, s))
    b = accuracy(params, Xte, yte, mag_masks(params, s, wanda_scores))
    print(f"{s:>8.2f}{a:>10.4f}{b:>10.4f}")
# 2:4 下的对照
w24 = nm24_masks(params, wanda_scores)
print(f"{'2:4':>8}{accuracy(params, Xte, yte, m24):>10.4f}"
      f"{accuracy(params, Xte, yte, w24):>10.4f}")
diff = sum(int((a != b).sum()) for a, b in zip(m24, w24))
tot = sum(a.size for a in m24)
print(f"两套 2:4 掩码只在 {diff}/{tot} = {diff/tot*100:.1f}% 的权重上做不同取舍")

# ---------- 计时：省了算力，省没省时间 ----------
W1 = params[0][0]
A = Xte
mask1 = mag_masks(params, 0.9)[0]



def bench(fn, reps=60):
    fn()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts)) * 1e6


t_dense = bench(lambda: A @ W1)
t_masked = bench(lambda: A @ (W1 * mask1))


def sparse_skip(A, W, mask):
    """真正跳过零：逐输出列只取非零的输入维度做小矩阵乘"""
    B = A.shape[0]
    out = np.zeros((B, W.shape[1]), np.float32)
    for o in range(W.shape[1]):
        idx = np.nonzero(mask[:, o])[0]
        out[:, o] = A[:, idx] @ W[idx, o]
    return out


t_gather = bench(lambda: sparse_skip(A, W1, mask1))

print("\n计时（A: 800x36, W1: 36x96, 90% 稀疏, 中位数 of 60 次）")
print(f"dense 乘        {t_dense:8.1f} µs   乘加 {A.shape[0]*36*96/1e6:.2f} M")
print(f"masked dense 乘 {t_masked:8.1f} µs   乘加同上，只是多乘一遍掩码")
print(f"gather 稀疏乘   {t_gather:8.1f} µs   乘加 {A.shape[0]*mask1.sum()/1e6:.2f} M")
print(f"masked/dense = {t_masked/t_dense:.2f}x   gather/dense = {t_gather/t_dense:.2f}x")

# 存储：掩码之后矩阵还是满的
total = sum(W.size for W, _ in params)
print(f"\n参数总量 {total}，掩码后仍占 {total} 个 float32 = {total*4/1024:.1f} KiB（内存一点没省）")
print("2:4 的一半权重 + 2bit 元数据:", f"{total*4/1024/2:.1f} KiB + {total*2/8/1024:.1f} KiB")
