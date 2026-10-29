#!/usr/bin/env python3
"""66 · DP-SGD：裁剪敏感度 → 噪声 → 隐私预算 ε → 准确率（numpy 单机实测）

不是 GPU / LLM 实测。合成二分类数据 + 逻辑回归，目的是把 DP-SGD 的
四个数字串起来：每样本梯度范数（定 C）、每步噪声（σC/B）、
ε（RDP 子采样会计师）、测试准确率。

用法：python3 66-dp-sgd.py
"""
import numpy as np
from math import comb, log, exp, inf

rng = np.random.default_rng(20261029)

N_BASE, N_OUT, N_TEST, D = 2000, 8, 4000, 40
w_true = rng.normal(0, 0.9, D)

Xb = rng.normal(0, 1, (N_BASE, D))
yb = (rng.random(N_BASE) < 1 / (1 + np.exp(-(Xb @ w_true)))).astype(float)
Xo = rng.normal(0, 1, (N_OUT, D)) * 6.0                     # 极端样本：范数 6 倍
yo = (rng.random(N_OUT) < 1 / (1 + np.exp(-(Xo @ w_true)))).astype(float)
Xte = rng.normal(0, 1, (N_TEST, D))
yte = (rng.random(N_TEST) < 1 / (1 + np.exp(-(Xte @ w_true)))).astype(float)

Xtr = np.vstack([Xb, Xo])
ytr = np.concatenate([yb, yo])
N = len(ytr)
LR = 0.5


def grad(theta, Xa, ya):
    """logistic 逐样本梯度：(p − y)·[x, 1]"""
    p = 1.0 / (1.0 + np.exp(-(Xa @ theta[:D] + theta[D])))
    r = (p - ya)[:, None]
    return np.concatenate([r * Xa, r], axis=1)


def train(sigma=0.0, B=100, epochs=60, clip=True, C=None, seed=0):
    r = np.random.default_rng(seed)
    theta = np.zeros(D + 1)
    steps = epochs * N // B
    for _ in range(steps):
        idx = r.integers(0, N, B)
        G = grad(theta, Xtr[idx], ytr[idx])
        if clip:
            nrm = np.linalg.norm(G, axis=1, keepdims=True)
            G = G * np.minimum(1.0, C / (nrm + 1e-12))
        g = G.mean(0)                       # 裁剪后均值：L2 敏感度 = C/B
        if sigma > 0:
            g = g + r.normal(0.0, sigma * C / B, size=g.shape)
        theta -= LR * g
    return theta, steps


def bce(theta, Xa, ya):
    z = Xa @ theta[:D] + theta[D]
    return np.logaddexp(0.0, z) - ya * z


def acc(theta, Xa, ya):
    return float(((Xa @ theta[:D] + theta[D] > 0) == (ya > 0.5)).mean())


def auc(pos, neg):
    """Mann-Whitney AUC（分数越大越像成员），并列取平均秩。"""
    v = np.concatenate([pos, neg])
    order = np.argsort(v, kind="mergesort")
    sv = v[order]
    rr = np.empty(len(v))
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and sv[j + 1] == sv[i]:
            j += 1
        rr[i:j + 1] = (i + j) / 2.0 + 1.0
        i = j + 1
    r = np.empty(len(v))
    r[order] = rr
    rp = r[:len(pos)].sum()
    return float((rp - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def rdp_subsampled(q, sigma, alpha):
    """子采样高斯机制的 RDP（Mironov et al. 2019，整数阶展开）。"""
    lg = [log(comb(alpha, k)) + k * log(q) + (alpha - k) * log(1 - q)
          + (k * k - k) / (2 * sigma * sigma) for k in range(alpha + 1)]
    m = max(lg)
    return (m + log(sum(exp(x - m) for x in lg))) / (alpha - 1)


def eps_delta(q, sigma, steps, delta=1e-6):
    return min(rdp_subsampled(q, sigma, a) * steps + log(1 / delta) / (a - 1)
               for a in range(2, 129))


# ---------- 1. 裁剪阈值：预热一轮，取逐样本梯度范数中位数 ----------
th0, _ = train(epochs=1, clip=False)
norms = np.linalg.norm(grad(th0, Xtr, ytr), axis=1)
C = float(np.median(norms))
print(f"train={N}（含 {N_OUT} 条 6 倍范数极端样本） test={N_TEST} dim={D} lr={LR}")
print(f"逐样本梯度范数：中位数 {C:.3f} / p99 {np.percentile(norms, 99):.2f} / "
      f"最大 {norms.max():.2f}（最大 = 中位数的 {norms.max() / C:.1f} 倍）\n")

cases = [
    ("无DP",              dict(sigma=0.0, B=100, epochs=60, clip=False)),
    ("仅裁剪 C",           dict(sigma=0.0, B=100, epochs=60, clip=True, C=C)),
    ("DP σ=1",            dict(sigma=1.0, B=100, epochs=60, clip=True, C=C)),
    ("DP σ=2",            dict(sigma=2.0, B=100, epochs=60, clip=True, C=C)),
    ("DP σ=4",            dict(sigma=4.0, B=100, epochs=60, clip=True, C=C)),
    ("DP σ=8",            dict(sigma=8.0, B=100, epochs=60, clip=True, C=C)),
    ("DP σ=1 · 1轮",       dict(sigma=1.0, B=100, epochs=1, clip=True, C=C)),
    ("DP σ=1 · B=500",    dict(sigma=1.0, B=500, epochs=60, clip=True, C=C)),
]

hdr = (f"{'设置':<14}{'步数':>6}{'ε(δ=1e-6)':>11}{'测试acc':>8}"
       f"{'成员loss':>9}{'非成员loss':>10}{'MI_AUC':>8}")
print(hdr)
print("-" * 76)
print("[零假设] 未训练模型上的 loss 攻击 AUC = "
      f"{auc(-bce(np.zeros(D + 1), Xtr, ytr), -bce(np.zeros(D + 1), Xte, yte)):.3f}")
out = {}
for name, kw in cases:
    th, steps = train(**kw)
    ltr, lte = bce(th, Xtr, ytr), bce(th, Xte, yte)
    eps = inf if kw["sigma"] == 0 else eps_delta(kw["B"] / N, kw["sigma"], steps)
    mi = auc(-ltr, -lte)
    out[name] = (eps, steps, acc(th, Xte, yte), float(ltr.mean()), float(lte.mean()), mi)
    eps_s = "∞" if eps == inf else f"{eps:.2f}"
    print(f"{name:<14}{steps:>6}{eps_s:>11}{out[name][2]:>8.3f}"
          f"{out[name][3]:>9.3f}{out[name][4]:>10.3f}{mi:>8.3f}")

# ---------- 2. 等预算对比：同样的 ε，多步小噪声 vs 少步大噪声 ----------
print("\n把 ε 花在「多步」还是「大噪声」上：")
for name in ("DP σ=1 · 1轮", "DP σ=2", "DP σ=1 · B=500"):
    e, s, a, lt, le, mi = out[name]
    print(f"  {name:<14} 步数 {s:>4}  ε={e:.2f}  acc={a:.3f}")
