#!/usr/bin/env python3
"""第 75 篇：奖励模型过优化（Goodhart）最小可复现实验。

设定（模拟，不是真调线上 RM）：
  - 答案空间 3000 条，每条 3 个可观测特征：c(答对) / L(长度,0..1) / s(谄媚)
  - 真实有用度 U = c - 0.8*relu(L-0.5) - 0.15*s
  - 标注者打分 A = c + 0.9*L + 0.4*s + N(0,0.3)（人偏好长答案与附和）
  - 在 A 的成对比较上用 Bradley-Terry 拟合奖励模型 r = w·x（牛顿法，3 个参数）
  - 策略 π_θ ∝ exp(θ·x)，对 E_π[r] 做精确梯度上升（不采样，曲线无噪声）
  - 对照组：只保留 |ΔL|<=0.05 的比较对，重拟合同一模型
本机 numpy 实测，秒级。
"""
import numpy as np

rng = np.random.default_rng(7)
N = 3000
c = rng.integers(0, 2, N).astype(float)
L = rng.random(N)
s = (rng.random(N) < 0.3).astype(float)
X = np.stack([c, L, s], 1)
U = c - 0.8 * np.clip(L - 0.5, 0, None) - 0.15 * s
A = X @ np.array([1.0, 0.9, 0.4])


def fit_rm(pairs, iters=25):
    """Bradley-Terry: P(i≻j)=sigmoid(r_i-r_j)，牛顿法拟 w（3 维）。"""
    i, j = pairs
    d = X[i] - X[j]
    w = np.zeros(3)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(d @ w)))
        g = d.T @ (p - 1.0) / len(i)
        H = (d * (p * (1 - p))[:, None]).T @ d / len(i)
        w -= np.linalg.solve(H + 1e-3 * np.eye(3), g)
    return w


def sample_pairs(n=30000, narrow=False):
    a = rng.integers(0, N, 4 * n)
    if narrow:
        # 在同一长度带里配对：b 取 a 附近 ±0.05 内长度最接近的候选
        b = a.copy()
        for _ in range(1):
            cand = rng.integers(0, N, 4 * n)
            ok = np.abs(L[cand] - L[a]) <= 0.05
            b = np.where(ok, cand, b)
        keep = np.abs(L[b] - L[a]) <= 0.05
        a, b = a[keep][:n], b[keep][:n]
    else:
        b = rng.integers(0, N, 4 * n)
    # 标注者噪声：同一条答案在两两比较里打分相同，噪点落在对子上
    win = A[a] + rng.normal(0, 0.3, len(a)) > A[b] + rng.normal(0, 0.3, len(a))
    return a[win], b[win]


w_biased = fit_rm(sample_pairs(narrow=False))
w_balanced = fit_rm(sample_pairs(narrow=True))
print("RM 权重 [答对, 长度, 谄媚]   真实有用度对应 [1, ≤0, -0.15]")
print("  常规成对数据 :", np.round(w_biased, 3))
print("  长度配对平衡 :", np.round(w_balanced, 3))


def optimize(w, steps=160, lr=1.5, tag=""):
    theta = np.zeros(3)
    hist = []
    for t in range(steps + 1):
        z = X @ theta
        p = np.exp(z - z.max())
        p /= p.sum()
        Er, Eu, El = float(p @ (X @ w)), float(p @ U), float(p @ L)
        kl = float(p @ np.log(p * N))
        hist.append((t, El, Er, Eu, kl))
        if t in (0, 1, 2, 3, 5, 10, 20, 40, 80, 160):
            print(f"  step {t:3d}  长度 {El:.3f}  奖励模型 {Er:+.3f}  真实 {Eu:+.3f}  KL {kl:.3f}")
        g = (X - p @ X).T @ (p * (X @ w))
        theta += lr * g
    return hist


for name, w in (("常规 RM", w_biased), ("长度平衡 RM", w_balanced)):
    print(f"\n对 {name} 做奖励最大化：")
    h = optimize(w)
    best = max(h, key=lambda r: r[3])
    print(f"  → 真实有用度峰值 {best[3]:+.3f}（step {best[0]}），"
          f"终点 {h[-1][3]:+.3f}，奖励模型端点 {h[-1][2]:+.3f}")

print("\n参考：单条答案真实有用度最大 =", round(float(U.max()), 3),
      "| 长度拉满+谄媚时的真实有用度 =", round(float(1 - 0.8 * 0.5 - 0.15), 3))
