#!/usr/bin/env python3
"""BatchNorm 与 LayerNorm 的差别：一个用 batch 统计量，一个用样本统计量。

三组测量，全部在本机实跑（纯 numpy，CPU）：
1) 耦合：改动 batch 里第 0 个样本，看另外 7 个样本的输出变多少。
2) batch=1：BN 的方差是 0，输出与输入无关；LN 不受影响。
3) 训练：同一个任务、同一个学习率，bs ∈ {1,8,64} × {none,bn,ln}，各 5 个种子。
"""
import numpy as np

D_IN, H, C = 16, 32, 4
EPS = 1e-5


def bn_fwd(x, g, b):
    mu = x.mean(0)
    var = x.var(0)
    std = np.sqrt(var + EPS)
    xhat = (x - mu) / std
    return g * xhat + b, (xhat, std)


def bn_bwd(dy, cache, g):
    xhat, std = cache
    m = dy.shape[0]
    dg = (dy * xhat).sum(0)
    db = dy.sum(0)
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(0) - xhat * (dxhat * xhat).mean(0)) / std
    return dx, dg, db


def ln_fwd(x, g, b):
    mu = x.mean(1, keepdims=True)
    var = x.var(1, keepdims=True)
    std = np.sqrt(var + EPS)
    xhat = (x - mu) / std
    return g * xhat + b, (xhat, std)


def ln_bwd(dy, cache, g):
    xhat, std = cache
    n = dy.shape[1]
    dg = (dy * xhat).sum(0)
    db = dy.sum(0)
    dxhat = dy * g
    dx = (dxhat - dxhat.mean(1, keepdims=True)
          - xhat * (dxhat * xhat).mean(1, keepdims=True)) / std
    return dx, dg, db


def coupling_test():
    rng = np.random.default_rng(0)
    B, D = 8, 16
    x = rng.normal(0, 1, (B, D))
    g, b = np.ones(D), np.zeros(D)
    for name, f in (("BN", bn_fwd), ("LN", ln_fwd)):
        y0, _ = f(x, g, b)
        x2 = x.copy()
        x2[0] += 1.0
        y1, _ = f(x2, g, b)
        delta = np.abs(y1[1:] - y0[1:]).max()
        scale = y0[1:].std()
        print(f"{name}: 改动样本0 后，其余 {B-1} 个样本输出的最大变化 "
              f"{delta:.4f}（输出自身标准差 {scale:.4f}，比 {delta/scale:.3f}）")
    # batch=1：BN 的方差恒为 0，输出与输入无关；LN 保留样本内部的相对结构
    xs = [np.random.default_rng(1).normal(0, 1, (1, D)),
          np.random.default_rng(2).normal(0, 1, (1, D))]
    for name, f in (("BN", bn_fwd), ("LN", ln_fwd)):
        outs = [f(x, g, b)[0] for x in xs]
        print(f"{name}: batch=1、两个不同输入的输出差异 {np.abs(outs[0]-outs[1]).max():.4f}"
              f"（各自输出幅度 {np.abs(outs[0]).max():.4f}）")


def make_data(n, rng, centers):
    y = rng.integers(0, C, n)
    return centers[y] + rng.normal(0, 1.0, (n, D_IN)), y


def train(norm, bs, seed, steps=600, lr=5e-3):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 2.5, (C, D_IN))       # train/val 共用同一组类中心
    Xtr, ytr = make_data(800, rng, centers)
    Xva, yva = make_data(400, rng, centers)
    W1 = rng.normal(0, np.sqrt(2 / D_IN), (D_IN, H))
    W2 = rng.normal(0, np.sqrt(2 / H), (H, C))
    b1 = np.zeros(H); b2 = np.zeros(C)
    gam = np.ones(H); bet = np.zeros(H)
    ps = [W1, b1, W2, b2, gam, bet]
    ms = [np.zeros_like(p) for p in ps]; vs = [np.zeros_like(p) for p in ps]
    nrm = {"bn": (bn_fwd, bn_bwd), "ln": (ln_fwd, ln_bwd)}.get(norm)

    def fwd(x):
        z1 = x @ W1 + b1
        cache = None
        if nrm:
            z1, cache = nrm[0](z1, gam, bet)
        h = np.tanh(z1)
        return h, h @ W2 + b2, cache, z1

    def acc(x, t):
        _, z, _, _ = fwd(x)
        return (z.argmax(1) == t).mean()

    t = 0
    for _ in range(steps):
        sel = rng.integers(0, len(Xtr), bs)
        x, tgt = Xtr[sel], ytr[sel]
        h, z, cache, z1 = fwd(x)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        dz = p; dz[np.arange(bs), tgt] -= 1; dz /= bs
        gW2 = h.T @ dz; gb2 = dz.sum(0)
        dh = (dz @ W2.T) * (1 - h ** 2)
        if nrm:
            dz1, gg, gb = nrm[1](dh, cache, gam)
        else:
            dz1, gg, gb = dh, np.zeros(H), np.zeros(H)
        gW1 = x.T @ dz1; gb1 = dz1.sum(0)
        grads = [gW1, gb1, gW2, gb2, gg, gb]
        t += 1
        for i, (p_, gr) in enumerate(zip(ps, grads)):
            ms[i] = 0.9 * ms[i] + 0.1 * gr
            vs[i] = 0.999 * vs[i] + 0.001 * gr * gr
            p_ -= lr * (ms[i] / (1 - 0.9 ** t)) / (np.sqrt(vs[i] / (1 - 0.999 ** t)) + 1e-8)
    return acc(Xva, yva)


def grad_probe():
    rng = np.random.default_rng(3)
    g, b = np.ones(32), np.zeros(32)
    dy = rng.normal(0, 1, (1, 32))
    for name, f, bw in (("BN", bn_fwd, bn_bwd), ("LN", ln_fwd, ln_bwd)):
        y, cache = f(rng.normal(0, 1, (1, 32)), g, b)
        dx, dg, db = bw(dy, cache, g)
        print(f"{name}: batch=1、上游梯度 ||dy||={np.linalg.norm(dy):.3f} 时，"
              f"||dx||={np.linalg.norm(dx):.3e}，||dγ||={np.linalg.norm(dg):.3e}")


if __name__ == "__main__":
    print("== 1. batch 内耦合 ==")
    coupling_test()
    print("\n== 2. batch=1 时谁还能传梯度 ==")
    grad_probe()
    print("\n== 3. 训练：验证准确率（5 种子均值 ± 标准差，随机猜 0.25）==")
    for norm in ("none", "bn", "ln"):
        for bs in (1, 8, 64):
            a = np.array([train(norm, bs, s) for s in range(5)])
            print(f"{norm:>4} bs={bs:>2}: {a.mean():.3f} ± {a.std():.3f}  {np.round(a, 3)}")
