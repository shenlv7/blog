#!/usr/bin/env python3
"""AI 的进化之路 11 期 demo：逐层 RBM 预训练到底买到了什么。

数据：自造 8x8 二值「笔画图」——横、竖、斜三种笔画随机摆放，8% 像素翻转噪声。
   1. 逐层训练 RBM 栈 64-48-24-12，看每层 CD-1 重构误差。
   2. 用这套权重初始化 64-48-24-12-24-48-64 自编码器并微调，对比随机初始化
      同架构、以及「恒输出训练集均值图」的基线。
   3. 只给 100 个标签训 64-48-24-3 分类器：预训练初始化 vs 随机初始化。
只依赖 numpy，seed 固定。
"""
import numpy as np

rng = np.random.default_rng(2006)
D = 64
SIZES = [64, 48, 24, 12]
N_CLASS = 3


def make_images(n):
    X = np.zeros((n, 8, 8), dtype=np.float32)
    y = rng.integers(0, N_CLASS, n)
    for i in range(n):
        if y[i] == 0:
            X[i, rng.integers(1, 7), 1:7] = 1
        elif y[i] == 1:
            X[i, 1:7, rng.integers(1, 7)] = 1
        else:
            k = rng.integers(7, 12)
            for r in range(8):
                if 0 <= k - r < 8:
                    X[i, r, k - r] = 1
    flip = rng.random(X.shape) < 0.08
    X[flip] = 1 - X[flip]
    return X.reshape(n, D), y


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p)).astype(np.float32)


class RBM:
    """二值 RBM，CD-1 + 动量。可见单元偏置按数据边缘概率初始化。"""

    def __init__(self, nv, nh, lr=0.05, seed=0):
        g = np.random.default_rng(seed)
        self.W = g.normal(0, 0.01, (nv, nh)).astype(np.float32)
        self.bv, self.bh, self.lr = np.zeros(nv, np.float32), np.zeros(nh, np.float32), lr

    def ph(self, v):
        return sig(v @ self.W + self.bh)

    def pv(self, h):
        return sig(h @ self.W.T + self.bv)

    def cd1(self, v0):
        p0 = self.ph(v0)
        h0 = (rng.random(p0.shape) < p0).astype(np.float32)
        v1 = self.pv(h0)
        p1 = self.ph((rng.random(v1.shape) < v1).astype(np.float32))
        n = len(v0)
        dW = (v0.T @ p0 - v1.T @ p1) / n
        self.W += self.lr * dW
        self.bv += self.lr * (v0 - v1).mean(0)
        self.bh += self.lr * (p0 - p1).mean(0)
        return float(((v0 - v1) ** 2).mean())

    def fit(self, X, epochs, bs=64, tag=""):
        self.bv[:] = logit(X.mean(0))                 # 偏置先对齐数据边缘分布
        for ep in range(epochs):
            idx = rng.permutation(len(X))
            errs = [self.cd1(X[idx[i:i + bs]]) for i in range(0, len(X), bs)]
            if (ep + 1) % 20 == 0:
                print(f"    {tag} epoch {ep+1:3d}  CD-1 重构 MSE {np.mean(errs):.4f}")


def pretrain(X, epochs=60):
    Ws, acts = [], X
    for i, nh in enumerate(SIZES[1:]):
        print(f"  RBM {i+1}: {acts.shape[1]} -> {nh}")
        r = RBM(acts.shape[1], nh, seed=11 + i)
        r.fit(acts, epochs, tag=f"L{i+1}")
        Ws.append(r.W.copy())
        acts = r.ph(acts)                            # 隐单元激活概率成为下一层的输入
    return Ws


def train_ae(X, Ws=None, epochs=600, lr=2e-3, tag=""):
    """自编码器 + Adam。Ws 给出时作为初始化（解码器取转置）。"""
    L = len(SIZES) - 1
    if Ws is None:
        g = np.random.default_rng(99)
        Ws = [g.normal(0, 0.05, (a, b)).astype(np.float32) for a, b in zip(SIZES[:-1], SIZES[1:])]
    enc = [w.copy() for w in Ws]
    dec = [w.T.copy() for w in reversed(Ws)]
    params = enc + dec
    m = [np.zeros_like(p) for p in params]
    v2 = [np.zeros_like(p) for p in params]
    t, b1, b2, eps = 0, 0.9, 0.999, 1e-8
    n, bs = len(X), 64
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = X[idx[i:i + bs]]
            a = [b]
            for w in enc:
                a.append(sig(a[-1] @ w))
            for w in dec:
                a.append(sig(a[-1] @ w))
            d = [None] * (2 * L + 1)
            out = a[2 * L]
            d[2 * L] = (out - b) * out * (1 - out) * 2.0 / b.size
            gd = [None] * L
            for k in range(L - 1, -1, -1):
                gd[k] = a[L + k].T @ d[L + k + 1]
                d[L + k] = (d[L + k + 1] @ dec[k].T) * a[L + k] * (1 - a[L + k])
            ge = [None] * L
            for k in range(L - 1, -1, -1):
                ge[k] = a[k].T @ d[k + 1]
                if k > 0:
                    d[k] = (d[k + 1] @ enc[k].T) * a[k] * (1 - a[k])
            t += 1
            for j, gr in enumerate(ge + gd):
                m[j] = b1 * m[j] + (1 - b1) * gr
                v2[j] = b2 * v2[j] + (1 - b2) * gr * gr
                params[j] -= lr * (m[j] / (1 - b1 ** t)) / (np.sqrt(v2[j] / (1 - b2 ** t)) + eps)
        if (ep + 1) % 200 == 0:
            print(f"    {tag} epoch {ep+1:3d}  本批训练 MSE {((a[2*L] - b) ** 2).mean():.4f}")
    h = X
    for w in enc:
        h = sig(h @ w)
    for w in dec:
        h = sig(h @ w)
    return float(((h - X) ** 2).mean()), enc


def feats(X, enc):
    h = X
    for w in enc:
        h = sig(h @ w)
    return h


def train_clf(Xtr, ytr, Xte, yte, init=None, epochs=400, lr=3e-3):
    """64-48-24-3 三层分类器，softmax + Adam；init 为空则随机初始化。"""
    g = np.random.default_rng(3)
    if init is None:
        Ws = [g.normal(0, np.sqrt(2 / a), (a, b)).astype(np.float32)
              for a, b in zip(SIZES[:-1], SIZES[1:])]
    else:
        Ws = [w.copy() for w in init]
    Ws.append(np.zeros((SIZES[-1], N_CLASS), np.float32))
    params = Ws
    m = [np.zeros_like(p) for p in params]
    v2 = [np.zeros_like(p) for p in params]
    t, b1, b2 = 0, 0.9, 0.999
    Y = np.eye(N_CLASS, dtype=np.float32)[ytr]
    n, bs = len(Xtr), 32
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, bs):
            b = Xtr[idx[i:i + bs]]
            a = [b]
            for w in Ws[:-1]:
                a.append(sig(a[-1] @ w))
            z = a[-1] @ Ws[-1]
            p = np.exp(z - z.max(1, keepdims=True))
            p /= p.sum(1, keepdims=True)
            d = (p - Y[idx[i:i + bs]]) / len(b)
            grads = [None] * len(Ws)
            grads[-1] = a[-1].T @ d
            for k in range(len(Ws) - 2, -1, -1):
                d = (d @ Ws[k + 1].T) * a[k + 1] * (1 - a[k + 1])
                grads[k] = a[k].T @ d
            t += 1
            for j, gr in enumerate(grads):
                m[j] = b1 * m[j] + (1 - b1) * gr
                v2[j] = b2 * v2[j] + (1 - b2) * gr * gr
                params[j] -= lr * (m[j] / (1 - b1 ** t)) / (np.sqrt(v2[j] / (1 - b2 ** t)) + 1e-8)
    h = Xte
    for w in Ws[:-1]:
        h = sig(h @ w)
    return float((np.argmax(h @ Ws[-1], 1) == yte).mean())


print("=" * 60)
Xtr, ytr = make_images(3000)
Xte, yte = make_images(1000)
mean_mse = float(((np.tile(Xtr.mean(0), (len(Xte), 1)) - Xte) ** 2).mean())
print(f"训练 {len(Xtr)} 张 / 测试 {len(Xte)} 张，像素均值 {Xtr.mean():.3f}")
print(f"基线（恒输出训练集均值图）：测试 MSE = {mean_mse:.4f}")

print("\n[1] 逐层预训练 RBM 栈")
Ws = pretrain(Xtr, epochs=60)

print("\n[2] 自编码器重构（测试集 MSE）")
mse_pre, enc_pre = train_ae(Xtr, Ws, tag="pre")
print(f"   预训练 + 微调        ：{mse_pre:.4f}")
mse_rnd, enc_rnd = train_ae(Xtr, None, tag="rnd")
print(f"   随机初始化          ：{mse_rnd:.4f}")
print(f"   恒输出均值图（基线） ：{mean_mse:.4f}")

print("\n[3] 只给 100 个标签的 3 分类（测试 1000 张）")
idx = rng.permutation(len(Xtr))[:100]
acc_pre = train_clf(Xtr[idx], ytr[idx], Xte, yte, init=enc_pre)
acc_rnd = train_clf(Xtr[idx], ytr[idx], Xte, yte, init=None)
print(f"   预训练初始化：{acc_pre:.3f}")
print(f"   随机初始化  ：{acc_rnd:.3f}")
print("=" * 60)
