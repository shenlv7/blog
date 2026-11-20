#!/usr/bin/env python3
"""88 — Chinchilla 参数化缩放律：把论文 Eq.(10) 的系数代回约束优化。

L(N,D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28        (Hoffmann et al. 2022, Eq. 10)
约束    C = 6 N D                                   (Kaplan et al. 2020, 稠密 Transformer)

输出：
  1. 闭式解 vs 网格搜索（验证解析式没推错）
  2. 各算力预算下的 N* / D* / tokens-per-param / 损失
  3. 与论文附录 Table A3（Approach 2 / 3）的对照
  4. 指数敏感性：alpha/beta 动 ±0.02，token/param 变多少
无 GPU，纯浮点运算；不是训练实测，是论文拟合式的重新代入。
"""
import math

E, A, B, ALPHA, BETA = 1.69, 406.4, 410.7, 0.34, 0.28


def loss(N, D, e=E, a=A, b=B, al=ALPHA, be=BETA):
    return e + a / N ** al + b / D ** be


def closed_form(C, al=ALPHA, be=BETA, a=A, b=B):
    """拉格朗日解：min L s.t. 6ND=C  =>  N* = (alphaA/betaB)^(1/(alpha+beta)) * (C/6)^(beta/(alpha+beta))"""
    s = al + be
    G = (al * a / (be * b)) ** (1.0 / s)
    N = G * (C / 6.0) ** (be / s)
    return N, C / (6.0 * N)


def grid_opt(C, lo=6.0, hi=13.0, steps=700):
    best = None
    for i in range(steps + 1):
        N = 10 ** (lo + (hi - lo) * i / steps)
        D = C / (6.0 * N)
        L = loss(N, D)
        if best is None or L < best[0]:
            best = (L, N, D)
    return best


def human(x, unit=""):
    for div, suf in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if x >= div:
            return f"{x/div:6.2f}{suf}{unit}"
    return f"{x:6.2f}{unit}"


print("=" * 78)
print("1) 闭式解 vs 网格搜索（同一预算下网格最小值的 N 应与解析 N* 重合）")
print("=" * 78)
print(f"{'算力 C (FLOPs)':>14} {'N* 解析':>10} {'N 网格':>10} {'差':>7} {'L 解析':>8} {'L 网格':>8}")
for C in (1e19, 1e21, 1e23, 5.76e23, 1e25):
    Nc, Dc = closed_form(C)
    Lg, Ng, _ = grid_opt(C)
    gap = abs(Ng / Nc - 1) * 100
    print(f"{C:14.2e} {human(Nc):>10} {human(Ng):>10} {gap:6.2f}% {loss(Nc, Dc):8.4f} {Lg:8.4f}")

print()
print("=" * 78)
print("2) 各预算的最优解（论文 Eq.10 系数，C=6ND 约束）")
print("=" * 78)
print(f"{'C (FLOPs)':>12} {'N*':>9} {'D*':>9} {'D/N':>8} {'L*':>7} {'训练天数@1e21 FLOP/s':>10}")
for C in (1e19, 1e20, 1e21, 1e22, 1e23, 5.76e23, 1e24, 1e25):
    N, D = closed_form(C)
    days = C / 1e21 / 86400
    print(f"{C:12.1e} {human(N):>9} {human(D):>9} {D/N:8.1f} {loss(N, D):7.4f} {days:10.1f}")

print()
print("=" * 78)
print("3) 与论文附录 Table A3 的对照（Approach 2 / Approach 3 的投影行）")
print("=" * 78)
print("论文 Table A3 原文：")
print("  Approach 2  67B  ->  6.88e23 FLOPs, 1.7T tokens  (25 token/param)")
print("  Approach 3  67B  ->  1.71e24 FLOPs, 4.1T tokens  (61 token/param)")
print("  Approach 1 67B  ->  1.5T tokens (Table 3)        (22 token/param)")
print("  Chinchilla 实际  70B x 1.4T = 5.88e23 FLOPs      (20 token/param)")
print()
for C, tag in ((6.88e23, "Approach 2 预算"), (1.71e24, "Approach 3 预算")):
    N, D = closed_form(C)
    print(f"  把 Eq.10 系数代入 {tag} C={C:.2e}:  N*={human(N)}  D*={human(D)}  "
          f"{D/N:.0f} token/param  L*={loss(N, D):.4f}")
NC, DC = 70e9, 1.4e12
print(f"\n  Chinchilla 实配 70B x 1.4T 的损失 = {loss(NC, DC):.4f}")
N2, D2 = closed_form(6 * NC * DC)
print(f"  同预算下 Eq.10 认为自己更优的点 = {human(N2)} x {human(D2)}  L={loss(N2, D2):.4f}"
      f"  (比实配低 {loss(NC, DC) - loss(N2, D2):.4f})")

print()
print("=" * 78)
print("4) 指数敏感性：D/N 是 (alpha,beta) 的函数，不是常数")
print("=" * 78)
print(f"{'alpha':>6} {'beta':>6} {'N 指数 a':>9} {'C=1e21 的 D/N':>14} {'C=5.76e23 的 D/N':>17}")
for al in (0.32, 0.34, 0.36):
    for be in (0.26, 0.28, 0.30):
        N1, D1 = closed_form(1e21, al=al, be=be)
        N2_, D2_ = closed_form(5.76e23, al=al, be=be)
        print(f"{al:6.2f} {be:6.2f} {be/(al+be):9.3f} {D1/N1:14.1f} {D2_/N2_:17.1f}")
