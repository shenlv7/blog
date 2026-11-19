#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai-evolution-09 · 1990s：从「造大脑」退到「解优化」
=====================================================
复现正文引用的几个量：

1. 同一数据集、同一条感知机规则，样本顺序不同 -> 超平面不同（无唯一解）。
2. 同一数据集、同一个软间隔 SVM 对偶问题，换 6 次样本顺序 -> 解一致（凸问题的唯一最优解）。
3. XOR 型四点：线性核分不开，二次核分开，而后者从没显式构造过 6 维特征。
4. 核与显式特征点积数值等价。
5. 扫 C：间隔大小与违反间隔的点数怎么换。

对偶问题用加速投影梯度（FISTA + 盒约束/等式约束投影）解，收敛看 KKT 残差。
只依赖 numpy，全部 CPU 计算，数据规模 16~26 个点。
"""

import numpy as np


# ------------------------------------------------------------ 核函数
def k_linear(A, B):
    return A @ B.T


def k_poly2(A, B):
    """k(u,v) = (1 + u·v)^2，等价于 6 维显式特征的点积。"""
    return (1.0 + A @ B.T) ** 2


def k_rbf(A, B, gamma=1.0):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-gamma * d2)


# ------------------------------------------------------------ 对偶求解
def project(v, y, C):
    """把 v 投到 {0 <= a <= C, a'y = 0}；对乘子做二分，单调性保证收敛。"""
    lo, hi = -10.0, 10.0
    for _ in range(18):
        mid = 0.5 * (lo + hi)
        s = np.sum(y * np.clip(v - mid * y, 0.0, C))
        if s > 0:
            lo = mid
        else:
            hi = mid
    return np.clip(v - 0.5 * (lo + hi) * y, 0.0, C)


def svm_fit(K, y, C=1.0, iters=2500):
    """解 min 1/2 a'Qa - sum(a) s.t. 0<=a<=C, a'y=0，Q_ij = y_i y_j K_ij"""
    Q = (y[:, None] * y[None, :]) * K
    L = np.linalg.eigvalsh(Q).max()
    a = np.zeros(len(y))
    z = a.copy()
    t = 1.0
    for _ in range(iters):
        a_new = project(z - (Q @ z - 1.0) / L, y, C)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        z = a_new + ((t - 1.0) / t_new) * (a_new - a)
        a, t = a_new, t_new
    # 用落在 (0, C) 上的支撑向量定 b
    f_free = (a * y) @ K
    free = (a > 1e-7) & (a < C - 1e-7)
    b = float(np.mean(y[free] - f_free[free])) if free.any() else 0.0
    return a, b


def kkt_residual(K, y, a, b, C):
    """所有点的最大违约量：alpha=0 要 y f >= 1，alpha=C 要 y f <= 1，自由点要 =1"""
    f = (a * y) @ K + b
    yf = y * f
    r = 0.0
    for ai, v in zip(a, yf):
        if ai <= 1e-7:
            r = max(r, 1.0 - v)
        elif ai >= C - 1e-7:
            r = max(r, v - 1.0)
        else:
            r = max(r, abs(v - 1.0))
    return r


def accuracy(K, y, a, b):
    f = (a * y) @ K + b
    pred = np.where(f >= 0, 1.0, -1.0)
    return float((pred == y).mean())


# ------------------------------------------------------------ 感知机
def perceptron(X, y, epochs=300):
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        errs = 0
        for xi, yi in zip(X, y):
            if yi * (w @ xi + b) <= 0:
                w += yi * xi
                b += yi
                errs += 1
        if errs == 0:
            break
    return w, b


# ------------------------------------------------------------ 数据
rng = np.random.default_rng(9)
c1 = rng.normal([-1.0, -1.0], 0.45, size=(12, 2))
c2 = rng.normal([1.0, 1.0], 0.45, size=(12, 2))
X = np.vstack([c1, c2])
y = np.array([-1.0] * 12 + [1.0] * 12)

print("=" * 70)
print("1) 可分的两团点：感知机的解取决于样本顺序")
print("=" * 70)
margins, angles = [], []
for seed in range(6):
    idx = np.random.default_rng(seed).permutation(len(y))
    w, b = perceptron(X[idx], y[idx])
    mg = 2.0 / np.linalg.norm(w)
    margins.append(mg)
    angles.append(np.degrees(np.arctan2(w[1], w[0])))
    print(f"  顺序 {seed}: w = [{w[0]: .3f}, {w[1]: .3f}], b = {b: .3f}, "
          f"间隔 = {mg:.3f}, 法向角 = {angles[-1]: .1f}°")
print(f"  -> 6 次都 100% 分开训练集；间隔最大 {max(margins):.3f}、最小 {min(margins):.3f}，"
      f"相差 {max(margins)/min(margins):.2f} 倍；法向角跨度 {max(angles)-min(angles):.1f}°")

print()
print("=" * 70)
print("2) 同一数据集：软间隔 SVM 的解与样本顺序无关")
print("=" * 70)
K = k_linear(X, X)
a0, b0 = svm_fit(K, y, C=10.0)
print(f"  原顺序: ||w|| = {np.linalg.norm((a0*y) @ X):.6f}, b = {b0:+.6f}, "
      f"支撑向量 {int((a0 > 1e-7).sum())} 个, 间隔 = {2/np.linalg.norm((a0*y) @ X):.3f}, "
      f"KKT 残差 = {kkt_residual(K, y, a0, b0, 10.0):.1e}")
sols = []
for seed in range(5):
    idx = np.random.default_rng(seed).permutation(len(y))
    a_s, b_s = svm_fit(K[np.ix_(idx, idx)], y[idx], C=10.0)
    aa = np.zeros(len(y))
    aa[idx] = a_s
    sols.append(np.append((aa * y) @ X, b_s))
    print(f"  顺序 {seed}: ||w|| = {np.linalg.norm((aa*y) @ X):.6f}, b = {b_s:+.6f}, "
          f"支撑向量 {int((aa > 1e-7).sum())} 个, 间隔 = {2/np.linalg.norm((aa*y) @ X):.3f}, "
          f"KKT 残差 = {kkt_residual(K, y, aa, b_s, 10.0):.1e}")
sols = np.array(sols)
print(f"  -> 5 次解与原顺序解的最大绝对偏差 = {np.abs(sols - np.append((a0*y) @ X, b0)).max():.2e}")
print(f"  -> SVM 间隔 {2/np.linalg.norm((a0*y) @ X):.3f}，对比感知机 6 个解里最宽的 {max(margins):.3f}")

print()
print("=" * 70)
print("3) XOR 型四点：线性核 50%，二次核 100%")
print("=" * 70)
Xx = np.array([[-1.0, -0.9], [-0.9, 1.0], [0.9, -1.0], [1.0, 0.95]])
yx = np.array([-1.0, 1.0, 1.0, -1.0])
for name, kf, C in [("linear", k_linear, 10.0), ("poly2", k_poly2, 10.0), ("rbf", k_rbf, 10.0)]:
    Kx = kf(Xx, Xx)
    ax, bx = svm_fit(Kx, yx, C=C, iters=6000)
    f = (ax * yx) @ Kx + bx
    print(f"  {name:6s}: 决策值 = {np.round(f, 3)}, 训练准确率 = {accuracy(Kx, yx, ax, bx):.2f}, "
          f"支撑向量 {int((ax > 1e-7).sum())}/4")

print()
print("=" * 70)
print("4) 核技巧核对：(1 + u·v)^2 与 6 维显式特征点积")
print("=" * 70)


def phi(u):
    x1, x2 = u
    return np.array([1.0, np.sqrt(2) * x1, np.sqrt(2) * x2, x1 ** 2, x2 ** 2, np.sqrt(2) * x1 * x2])


u, v = np.array([0.7, -1.3]), np.array([-0.4, 0.9])
lhs, rhs = (1.0 + u @ v) ** 2, phi(u) @ phi(v)
print(f"  k(u,v)        = {lhs:.15f}")
print(f"  phi(u)·phi(v) = {rhs:.15f}")
print(f"  |差|          = {abs(lhs - rhs):.2e}   (输入 2 维 -> 特征 6 维)")
Phi = np.array([phi(x) for x in X])
print(f"  24 个点的核矩阵与显式特征点积的最大偏差 = {np.abs(k_poly2(X, X) - Phi @ Phi.T).max():.2e}")

print()
print("=" * 70)
print("5) 软间隔参数 C：间隔与违反间隔的点数")
print("=" * 70)
Xn = np.vstack([X, [[0.15, 0.05], [0.05, 0.20]]])
yn = np.append(y, [1.0, -1.0])
Kn = k_linear(Xn, Xn)
for C in [0.01, 0.1, 1.0, 100.0]:
    a, b = svm_fit(Kn, yn, C=C)
    w = (a * yn) @ Xn
    sv = int((a > 1e-7).sum())
    at_bound = int((a > C - 1e-7).sum())
    print(f"  C = {C:6.2f}: 训练准确率 {accuracy(Kn, yn, a, b):.2f}, "
          f"||w|| = {np.linalg.norm(w):6.3f}, 间隔 = {2/np.linalg.norm(w):6.3f}, "
          f"支撑向量 {sv:2d} 个, 其中 a=C {at_bound:2d} 个")

print()
print("注：本脚本为 CPU/numpy 实现，数据规模 16~26 点。")
print("    USPS 9709 训练 / 2007 测试上的实验不在本脚本范围内。")
