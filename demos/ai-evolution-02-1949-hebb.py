#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
赫布规则（Hebb 1949）的四个可复现实验
1. 线性赫布规则：方向收敛到第一主成分，模长指数发散
2. 归一化 = 幂迭代；Oja 规则把模长钉在 1
3. 细胞组合：纯赫布能接出块结构，但会被"放电频率"劫持
4. 同一批权重当联想存储器：损坏率一高就退回随机
全部为数值模拟，不是生物实测。
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(1949)


def unit(v):
    return v / np.linalg.norm(v)


def ang_deg(a, b):
    c = abs(float(np.dot(unit(a), unit(b))))
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


# ---------------------------------------------------------------- 实验 1
print("=" * 62)
print("实验 1  线性赫布规则 dw = eta * y * x   （y = w·x）")
print("=" * 62)
C = np.array([[3.0, 2.4], [2.4, 2.5]])
L = np.linalg.cholesky(C)
lam, V = np.linalg.eigh(C)
pc1 = V[:, -1]
print("输入协方差 C =", C.tolist())
print("特征值 =", lam, " 第一主成分 =", pc1)

eta, batch = 0.01, 256
w = np.array([0.6, 0.8])
print(f"\n{'step':>5} {'||w||':>12} {'与PC1夹角':>10}")
for t in range(201):
    if t in (0, 25, 50, 100, 150, 200):
        print(f"{t:>5} {np.linalg.norm(w):>12.4f} {ang_deg(w, pc1):>9.3f}°")
    xb = (L @ rng.standard_normal((2, batch))).T      # (batch, 2)
    g = xb.T @ (xb @ w) / batch                       # 批平均的赫布梯度
    w = w + eta * g
print(f"预测每步放大 1+eta*lambda1 = {1 + eta * lam[-1]:.4f}，"
      f"200 步后约 {(1 + eta * lam[-1]) ** 200:.0f} 倍")

# ---------------------------------------------------------------- 实验 2
print()
print("=" * 62)
print("实验 2  归一化就是幂迭代；Oja 规则给模长一个上界")
print("=" * 62)
w_n = np.array([1.0, 0.0])
w_o = np.array([1.0, 0.0])
eta, batch = 0.01, 64
print(f"{'step':>5} {'归一化夹角':>12} {'Oja夹角':>10} {'Oja ||w||':>10}")
for t in range(401):
    if t in (0, 10, 25, 50, 100, 200, 400):
        print(f"{t:>5} {ang_deg(w_n, pc1):>11.4f}° {ang_deg(w_o, pc1):>9.4f}° "
              f"{np.linalg.norm(w_o):>10.4f}")
    xb = (L @ rng.standard_normal((2, batch))).T
    yv = xb @ w_n
    w_n = unit(w_n + eta * (xb.T @ yv) / batch)        # 纯赫布 + 每步归一化
    yo = xb @ w_o
    grad_o = (yo[:, None] * (xb - yo[:, None] * w_o)).sum(axis=0) / batch
    w_o = w_o + eta * grad_o
print(f"Oja 终点 ||w|| = {np.linalg.norm(w_o):.4f}（自己停在 1）")

# ---------------------------------------------------------------- 实验 3
print()
print("=" * 62)
print("实验 3  细胞组合：80 个神经元、4 个共激活组（每组 20）")
print("=" * 62)
N, G, SIZE, P_IN, P_OUT, eta = 80, 4, 80, 0.85, 0.06, 0.02
membership = np.arange(N) // 20
same = membership[:, None] == membership[None, :]
off = ~np.eye(N, dtype=bool)


def run(steps, decay):
    W = np.zeros((N, N))
    rate = np.zeros(N)
    for _ in range(steps):
        g = rng.integers(G)
        p = np.full(N, P_OUT)
        p[membership == g] = P_IN
        x = (rng.random(N) < p).astype(float)
        rate += x
        W = (1.0 - decay) * W + eta * np.outer(x, x)   # decay=0 就是纯赫布
    Wm = np.abs(W).copy()
    np.fill_diagonal(Wm, -1)
    top = np.argsort(Wm, axis=1)[:, -19:]
    hit = (membership[top] == membership[:, None]).mean()
    wi = np.abs(W)[off & same].mean()
    be = np.abs(W)[off & ~same].mean()
    return W, hit, wi, be


print(f"{'步数':>6} {'衰减':>6} {'||W||_F':>10} {'组内均值':>9} "
      f"{'组间均值':>9} {'对比度':>7} {'前19伙伴同组':>12}")
for steps, decay in ((500, 0.0), (1000, 0.0), (2000, 0.0), (4000, 0.0),
                     (4000, 0.001), (4000, 0.01)):
    W, hit, wi, be = run(steps, decay)
    print(f"{steps:>6} {decay:>6.3f} {np.linalg.norm(W):>10.1f} {wi:>9.4f} "
          f"{be:>9.4f} {wi / be:>7.2f} {hit:>11.1%}")

W_hi, hit_hi, wi_hi, _ = run(4000, 0.0)
W_lo, hit_lo, wi_lo, _ = run(4000, 0.01)
print(f"\n纯赫布跑 4000 步的组内权重是跑 1000 步的 "
      f"{wi_hi / run(1000, 0.0)[2]:.2f} 倍——同一批连接，强度只跟学习时长有关；")
print(f"加 0.01 衰减后稳态组内权重 {wi_lo:.3f}，仍保留块结构 {hit_lo:.0%}。")

# ---------------------------------------------------------------- 实验 4
print()
print("=" * 62)
print("实验 4  同一批权重当联想存储器：从残缺线索恢复模式")
print("=" * 62)
N, P = 100, 5
pats = rng.choice([-1.0, 1.0], size=(P, N))
W = pats.T @ pats
np.fill_diagonal(W, 0.0)                 # 不自激
print(f"N={N} 个神经元，存入 {P} 个模式，"
      f"Hebb 型存储器的理论容量约 0.138*N = {0.138 * N:.1f}")
print(f"{'损坏率':>7} {'逐位准确率':>11} {'完全恢复':>10}")
for rate in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
    acc, exact, trials = 0.0, 0, 40
    for _ in range(trials):
        k = rng.integers(P)
        cue = pats[k].copy()
        cue[rng.random(N) < rate] *= -1
        s = cue.copy()
        for _ in range(6):
            s = np.sign(W @ s)
            s[s == 0] = 1.0
        acc += (s == pats[k]).mean()
        exact += int((s == pats[k]).all())
    print(f"{rate:>7.0%} {acc / trials:>10.1%} {exact:>7d}/{trials}")

print()
print("三条边界：赫布规则给出的是相关，不是因果；没有误差信号；")
print("权重没有上界，必须靠外部规则（衰减、归一化、竞争）才稳得住。")
