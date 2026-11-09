#!/usr/bin/env python3
"""1969 · Minsky & Papert《Perceptrons》的最小可运行实验

四个实验，只用 NumPy：
  A. 原始输入上的单层单元学 XOR：训练集最高 3/4，永远到不了 4/4
  B. 换成 order-k 的与项特征：k < n 时最小二乘解退化成常数，k = n 才精确命中
  C. 该表示的系数比恒为 2^(n-1)（书中 §3.2 的结论），且对多出来的一位零预测（恰好 50%）
  D. 连通性归约的最小实例：奇数行填满时，图形连通 <=> 每个偶数行非空
"""
import itertools
import numpy as np


# ---------------------------------------------------------------- 数据
def parity_data(n):
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)
    y = (X.sum(axis=1) % 2).astype(np.float64)
    return X, y


def conj_features(X, k):
    """order-k 特征：所有 |S| <= k 的与项 AND_{i in S} x_i。"""
    n = X.shape[1]
    return np.stack([X[:, S].prod(axis=1)
                     for s in range(1, k + 1)
                     for S in itertools.combinations(range(n), s)], axis=1)


# ---------------------------------------------------------------- A
def perceptron_rule(X, y, epochs=150, lr=1.0, seed=0):
    """Rosenblatt 式错误修正：w += lr*(y-yhat)*x，只在线性可分时收敛。"""
    r = np.random.default_rng(seed)
    w = r.normal(0, 0.01, X.shape[1])
    b = 0.0
    best = 0.0
    drift = []
    for _ in range(epochs):
        w_prev = w.copy()
        for i in r.permutation(len(X)):
            yhat = 1.0 if X[i] @ w + b > 0 else 0.0
            w += lr * (y[i] - yhat) * X[i]
            b += lr * (y[i] - yhat)
        best = max(best, np.mean((X @ w + b > 0).astype(float) == y))
        drift.append(np.linalg.norm(w - w_prev))
    return w, b, best, float(np.mean(drift[-10:]))


def exp_a():
    X, y = parity_data(2)
    bests, finals, drifts = [], [], []
    for s in range(20):
        w, b, best, dr = perceptron_rule(X, y, seed=s)
        bests.append(best)
        finals.append(np.mean((X @ w + b > 0).astype(float) == y))
        drifts.append(dr)
    print("[A] XOR 上的错误修正规则，20 个随机初值 / 150 轮")
    print(f"    训练集历史上最高准确率: max={max(bests):.2f}  mean={np.mean(bests):.2f}"
          f"  (4 个样本里最多对 3 个)")
    print(f"    末轮准确率            : mean={np.mean(finals):.2f}  最小={min(finals):.2f}")
    print(f"    末 10 轮权重位移      : mean={np.mean(drifts):.4f}  (不趋零 = 不收敛)")
    ya = (X.sum(axis=1) == 2).astype(float)
    w, b, _, _ = perceptron_rule(X, ya, seed=0)
    print(f"    对照 AND（线性可分）  : 末轮准确率={np.mean((X @ w + b > 0).astype(float) == ya):.2f}")


# ---------------------------------------------------------------- B
def exp_b(n=8):
    X, y = parity_data(n)
    print(f"[B] {n} 位 parity：只放开 order<=k 的与项，最小二乘闭式解（无优化器，结果确定）")
    print("    k    特征数   秩    训练准确率   残差/样本数")
    for k in range(1, n + 1):
        F = conj_features(X, k)
        A = np.hstack([F, np.ones((len(F), 1))])
        w = np.linalg.lstsq(A, y, rcond=None)[0]
        pred = (A @ w > 0.5).astype(float)
        resid = np.mean((y - A @ w) ** 2)
        print(f"    {k:<4} {F.shape[1]:<8} {np.linalg.matrix_rank(A):<5} "
              f"{np.mean(pred == y):<12.3f} {resid:.3f}")


# ---------------------------------------------------------------- C
def exp_c():
    print("[C] 精确解：w_S = (-2)^(|S|-1)，阈值 0.5（这一族权重由 |S| 唯一决定）")
    print("    n   特征数   系数比 max/min   训练准确率   多一位后的准确率")
    for n in (2, 3, 4, 6, 8, 10, 12):
        X, y = parity_data(n)
        F = conj_features(X, n)
        sizes = np.array([len(S) for s in range(1, n + 1)
                          for S in itertools.combinations(range(n), s)])
        w = (-2.0) ** (sizes - 1)
        acc = np.mean(((F @ w) > 0.5).astype(float) == y)
        ratio = np.abs(w).max() / np.abs(w).min()
        Xn = np.array(list(itertools.product([0, 1], repeat=n + 1)), dtype=float)
        yn = (Xn.sum(axis=1) % 2).astype(float)
        Fn = conj_features(Xn[:, :n], n)          # 新的一位不出现在任何特征里
        acc_ext = np.mean(((Fn @ w) > 0.5).astype(float) == yn)
        print(f"    {n:<3} {F.shape[1]:<8} {ratio:<16.0f} {acc:.3f}        {acc_ext:.3f}")


# ---------------------------------------------------------------- D
def connected(cells):
    if not cells:
        return False
    start = next(iter(cells))
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cells)


def exp_d():
    """定理 5.1 证明里那一步的最小版：2m+1 行，奇数行整行填满（m=2, 5 行 4 列）。
       此时图形连通 <=> 每个偶数行至少有一个格子，即 m 个箱子的 one-in-a-box。"""
    rows, cols = 5, 4
    even_rows, odd_rows = [1, 3], [0, 2, 4]
    total = mismatch = both = 0
    for bits in itertools.product([0, 1], repeat=len(even_rows) * cols):
        cells = {(r, c) for r in odd_rows for c in range(cols)}
        for idx, bit in enumerate(bits):
            if bit:
                cells.add((even_rows[idx // cols], idx % cols))
        lab = connected(cells)
        oiab = all(any((r, c) in cells for c in range(cols)) for r in even_rows)
        total += 1
        both += oiab
        mismatch += (lab != oiab)
    print(f"[D] {rows}x{cols}，奇数行填满，偶数行任意（共 {total} 个图形）")
    print(f"    「图形连通」与「每个偶数行非空」不一致的图形数: {mismatch}")
    print(f"    两者都判为连通的图形数: {both}")
    print("    即：连通性在这族图形上等价于 m 个箱子的 one-in-a-box，而该谓词的 order = m")


if __name__ == "__main__":
    exp_a()
    print()
    exp_b(8)
    print()
    exp_c()
    print()
    exp_d()
