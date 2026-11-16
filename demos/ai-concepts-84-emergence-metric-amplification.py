#!/usr/bin/env python3
"""概念卡片 84 · 「涌现」是度量放大：精确匹配 = 每 token 正确率的 L 次幂。

全程离散采样模拟（numpy，CPU），不是任何真实模型的评测。

A. 同一个模型族（每 token 正确率随 log10N 平滑上升），只改答案长度 L=1/2/4/8，
   同时算 token 级准确率与精确匹配（EM），比较 logit 斜率与 5%→95% 过渡宽度。
B. 只用 100 条测试样本去测小模型：真实 EM 已经 0.81% 时，观测到「0 分」的概率。
"""
import math

import numpy as np

rng = np.random.default_rng(20261116)
K, X0 = 1.2, 9.6          # 模型族：logit(p_token) = K*(log10N - X0)
XS = np.round(np.arange(6.0, 14.01, 0.1), 1)
N = 5000


def p_token(x):
    return 1.0 / (1.0 + math.exp(-K * (x - X0)))


def measure(x, L, n=N):
    hits = rng.binomial(L, p_token(x), size=n)     # 每条样本答对几个 token
    return hits.mean() / L, (hits == L).mean()    # token_acc, exact_match


def logit(v):
    v = np.clip(v, 1e-6, 1 - 1e-6)
    return np.log(v / (1 - v))


def slope(xs, ys):
    """过渡段内的最小二乘斜率：只取 0.05 < metric < 0.95 的点。"""
    xs, ys, vs = np.array(xs), logit(np.array(ys)), np.array(ys)
    m = (vs > 0.05) & (vs < 0.95)
    return float(np.polyfit(xs[m], ys[m], 1)[0]), int(m.sum())


def width(xs, ys, lo=0.05, hi=0.95):
    xs, ys = np.array(xs), np.array(ys)
    def cross(t):
        idx = np.where((ys[:-1] - t) * (ys[1:] - t) <= 0)[0]
        if not len(idx):
            return None
        i = idx[0]
        return xs[i] + (t - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    a, b = cross(lo), cross(hi)
    return None if a is None or b is None else float(b - a)


print("=== A. 同一模型族、同一份权重曲线，只换任务长度 L ===")
print(f"模型族 logit(p_token) = 1.2*(log10N - 9.6)，x 扫 6.0→14.0，每条设置采样 n={N}\n")
print(f"{'L':>2} {'tok斜率':>8} {'EM斜率':>8} {'斜率比':>7} {'tok宽':>7} {'EM宽':>7} {'宽度比':>7}")
table = {}
for L in (1, 2, 4, 8):
    ts, es = [], []
    for x in XS:
        t, e = measure(float(x), L)
        ts.append(t)
        es.append(e)
    st, _ = slope(XS, ts)
    se, _ = slope(XS, es)
    wt, we = width(XS, ts), width(XS, es)
    table[L] = (ts, es)
    print(f"{L:>2} {st:8.3f} {se:8.3f} {se/st:7.2f} "
          f"{wt:7.3f} {we:7.3f} {wt/we:7.2f}")

print("\nL=4 的具体读数（「4 位算术」的形状）：")
ts, es = table[4]
print(f"{'log10N':>7} {'p_token':>8} {'token_acc':>10} {'exact_match':>12}")
for x in (8.0, 9.0, 9.4, 9.6, 9.8, 10.0, 10.5, 11.0):
    i = int(np.where(np.isclose(XS, x))[0][0])
    print(f"{x:7.1f} {p_token(x):8.4f} {ts[i]:10.4f} {es[i]:12.4f}")

print("\n=== A2. 同一个能力、不同任务长度：表面得分差多少 ===")
print(f"{'L':>2} {'EM=1% 时 p':>11} {'EM=50% 时 p':>12} {'p=0.56 时 EM':>13} {'p=0.70 时 EM':>13}")
for L in (1, 2, 4, 8, 16):
    print(f"{L:>2} {0.01**(1/L)*100:10.1f}% {0.5**(1/L)*100:11.1f}% "
          f"{0.56**L*100:12.2f}% {0.70**L*100:12.2f}%")

print("\n=== A3. logit 尺度的放大倍数 L(1-p)/(1-p^L) ===")
for p in (0.1, 0.3, 0.5, 0.7, 0.9):
    r = [L * (1 - p) / (1 - p ** L) for L in (2, 4, 8)]
    print(f"p={p:.1f}  L=2:{r[0]:5.2f}  L=4:{r[1]:5.2f}  L=8:{r[2]:5.2f}")

print("\n=== B. 100 条样本、真实 EM 已经不为 0 时，观测到「0 分」的概率 ===")
for p in (0.30, 0.40, 0.50):
    true_em = p ** 4
    theory = (1 - true_em) ** 100
    zero = np.mean([(rng.binomial(4, p, size=100) == 4).mean() == 0 for _ in range(4000)])
    print(f"p_token={p:.2f}  真实EM={true_em*100:5.2f}%  "
          f"理论P(0分)={theory*100:5.1f}%  4000 次实测={zero*100:5.1f}%")
