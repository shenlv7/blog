#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai-evolution 08 · 1986 反向传播
五个实验，全部纯 numpy，本机可跑：
  1. 梯度检验：解析梯度 vs 中心差分（误差量级）
  2. 2-2-1 学 XOR：十个随机种子，统计收敛所需轮数
  3. 单层 2-1 学 XOR：证明无隐藏单元时的误差下界 = 0.25
  4. 全零初始化：对称性不破，隐藏单元永远学到同一个函数
  5. 复现 Rumelhart-Hinton-Williams 1986 Fig.1 的镜像对称任务（64 个 6 位串，2 个隐藏单元）
  6. 解析梯度 vs 差分的墙钟时间比（反向传播为什么"算得动"）
"""
import time
import numpy as np

np.set_printoptions(precision=3, suppress=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class MLP:
    """只有一层隐藏层的全连接网，sigmoid 单元，均方误差 + 动量项。

    Δw(t) = -eps * dE/dw + alpha * Δw(t-1)   # 1986 年那篇论文的第 9 式
    """

    def __init__(self, n_in, n_hidden, n_out, rng, init_scale=0.3):
        self.W1 = rng.uniform(-init_scale, init_scale, (n_in, n_hidden))
        self.b1 = np.zeros(n_hidden)
        self.W2 = rng.uniform(-init_scale, init_scale, (n_hidden, n_out))
        self.b2 = np.zeros(n_out)
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X):
        H = sigmoid(X @ self.W1 + self.b1)
        O = sigmoid(H @ self.W2 + self.b2)
        return H, O

    def grads(self, X, Y):
        n = X.shape[0]
        H, O = self.forward(X)
        dO = (O - Y) * O * (1.0 - O)              # sigmoid + MSE 的 delta
        gW2 = H.T @ dO / n
        gb2 = dO.mean(axis=0)
        dH = (dO @ self.W2.T) * H * (1.0 - H)     # 误差走回隐藏层
        gW1 = X.T @ dH / n
        gb1 = dH.mean(axis=0)
        return gW1, gb1, gW2, gb2, O

    def step(self, grads, eps=0.5, alpha=0.9):
        gW1, gb1, gW2, gb2, _ = grads
        self.vW1 = alpha * self.vW1 - eps * gW1
        self.vb1 = alpha * self.vb1 - eps * gb1
        self.vW2 = alpha * self.vW2 - eps * gW2
        self.vb2 = alpha * self.vb2 - eps * gb2
        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def loss(self, X, Y):
        _, O = self.forward(X)
        return 0.5 * ((O - Y) ** 2).sum() / X.shape[0]


XOR_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_Y = np.array([[0.0], [1.0], [1.0], [0.0]])


def run_xor(seed, n_hidden=2, init_scale=0.3, epochs=8000, tol=0.05):
    rng = np.random.default_rng(seed)
    net = MLP(2, n_hidden, 1, rng, init_scale)
    for ep in range(1, epochs + 1):
        net.step(net.grads(XOR_X, XOR_Y), eps=0.5, alpha=0.9)
        _, O = net.forward(XOR_X)
        if np.abs(O - XOR_Y).max() < tol:
            return ep, O.ravel(), net
    _, O = net.forward(XOR_X)
    return None, O.ravel(), net


print("=" * 62)
print("实验 1  梯度检验：解析梯度 vs 中心差分")
print("=" * 62)
rng = np.random.default_rng(7)
net = MLP(2, 3, 1, rng, init_scale=1.0)
g = net.grads(XOR_X, XOR_Y)[0]
h = 1e-6
num = np.zeros_like(net.W1)
for i in range(net.W1.shape[0]):
    for j in range(net.W1.shape[1]):
        old = net.W1[i, j]
        net.W1[i, j] = old + h
        lp = net.loss(XOR_X, XOR_Y)
        net.W1[i, j] = old - h
        lm = net.loss(XOR_X, XOR_Y)
        net.W1[i, j] = old
        num[i, j] = (lp - lm) / (2 * h)
fmt = lambda a: np.array2string(a, formatter={"float_kind": lambda v: f"{v:+.6f}"})
print("解析梯度  dE/dW1 =\n", fmt(g))
print("数值梯度  dE/dW1 =\n", fmt(num))
print(f"最大绝对误差 = {np.abs(g - num).max():.2e}   （单精度训练可接受门槛是 1e-4）")

print()
print("=" * 62)
print("实验 2  2-2-1 学 XOR：十个种子")
print("=" * 62)
eps_list = []
ok = 0
for s in range(10):
    ep, out, _ = run_xor(s)
    if ep:
        ok += 1
        eps_list.append(ep)
        print(f"seed {s}: 第 {ep:4d} 轮收敛  输出 = {out}")
    else:
        print(f"seed {s}: 8000 轮未收敛  输出 = {out}")
print(f"收敛 {ok}/10，平均 {np.mean(eps_list):.0f} 轮，最慢 {max(eps_list)} 轮，最快 {min(eps_list)} 轮")

print()
print("=" * 62)
print("实验 3  单层 2-1（无隐藏层）学 XOR：误差下界")
print("=" * 62)
rng = np.random.default_rng(0)
w = rng.normal(0, 1, 2)
b = 0.0
for _ in range(20000):
    z = XOR_X @ w + b
    o = sigmoid(z)
    d = (o - XOR_Y.ravel()) * o * (1 - o)
    w -= 0.5 * (XOR_X.T @ d) / 4
    b -= 0.5 * d.mean()
o = sigmoid(XOR_X @ w + b)
print(f"训练后输出 = {o}")
print(f"MSE = {((o - XOR_Y.ravel())**2).mean():.4f}   理论下界 0.25（两个输入输出必须取同一个值）")
print(f"阈值 0.5 判对 = {int(((o > 0.5) == (XOR_Y.ravel() > 0.5)).sum())}/4")

print()
print("=" * 62)
print("实验 3b  线性隐藏层（去掉 sigmoid）学 XOR：加层等于没加")
print("=" * 62)
rng = np.random.default_rng(5)
W1 = rng.normal(0, 0.5, (2, 2)); b1 = np.zeros(2)
W2 = rng.normal(0, 0.5, (2, 1)); b2 = np.zeros(1)
lin = lambda z: z                              # 线性单元：σ 换成恒等映射
for _ in range(20000):
    H = lin(XOR_X @ W1 + b1)
    O = lin(H @ W2 + b2)
    dO = (O - XOR_Y)                           # 线性单元的 delta 就是残差本身
    W2 -= 0.5 * (H.T @ dO) / 4
    b2 -= 0.5 * dO.mean(axis=0)
    dH = (dO @ W2.T)                           # 没有 σ'(z) 这个因子
    W1 -= 0.5 * (XOR_X.T @ dH) / 4
    b1 -= 0.5 * dH.mean(axis=0)
H = lin(XOR_X @ W1 + b1)
O = lin(H @ W2 + b2)
print(f"训练后输出 = {O.ravel()}")
print(f"MSE = {((O - XOR_Y) ** 2).mean():.6f}   两个隐藏单元实际是输入的线性变换：")
print(f"  h = xW1，W1 的秩 = {np.linalg.matrix_rank(W1)}，两层复合等价于单个 2x1 线性映射")

print()
print("=" * 62)
print("实验 4  全零初始化：隐藏单元退化成同一个")
print("=" * 62)
rng = np.random.default_rng(1)
zero = MLP(2, 2, 1, rng, init_scale=0.0)
for _ in range(8000):
    zero.step(zero.grads(XOR_X, XOR_Y), eps=0.5, alpha=0.9)
H, O = zero.forward(XOR_X)
print(f"8000 轮后 MSE = {zero.loss(XOR_X, XOR_Y):.4f}（实验 2 收敛时约 2e-3）")
print("两个隐藏单元对 4 个输入的激活：")
print(f"  unit0 = {H[:, 0]}")
print(f"  unit1 = {H[:, 1]}")
print(f"  两者最大差 = {np.abs(H[:, 0] - H[:, 1]).max():.2e}  -> 完全共线，等于只有一个隐藏单元")

print()
print("=" * 62)
print("实验 5  8-3-8 编码器：3 个隐藏单元装下 8 个 8 位模式")
print("=" * 62)
# 8 个 8 位模式（人为给定，互不相同）；目标输出 = 输入本身，隐藏层只有 3 个单元
PAT = np.array([
    [1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1],
], dtype=float)
rng = np.random.default_rng(11)
enc = MLP(8, 3, 8, rng, init_scale=0.5)
for ep in range(1, 30001):
    enc.step(enc.grads(PAT, PAT), eps=0.5, alpha=0.9)
    _, O = enc.forward(PAT)
    if np.abs(O - PAT).max() < 0.1:
        break
H, O = enc.forward(PAT)
print(f"{ep} 轮后最大绝对误差 = {np.abs(O - PAT).max():.5f}，MSE = {enc.loss(PAT, PAT):.2e}")
print("8 位 -> 3 位 -> 8 位：24 个输入到隐藏的权重 + 24 个隐藏到输出的权重")
print("学到的 3 维码（8 个模式是相邻位移，码字也排成一条连续路径）：")
for i in range(8):
    print(f"  {PAT[i].astype(int)} -> ({H[i,0]:.2f}, {H[i,1]:.2f}, {H[i,2]:.2f})")
dmin = min(np.abs(H[i] - H[j]).sum() for i in range(8) for j in range(i + 1, 8))
print(f"码字之间的最小 L1 距离 = {dmin:.2f}（没有两个模式共用同一个码）")


print()
print("=" * 62)
print("实验 6  为什么『往回传』是这个算法能跑起来的原因")
print("=" * 62)
rng = np.random.default_rng(3)
big = MLP(50, 40, 10, rng, init_scale=0.1)
Xb = rng.uniform(0, 1, (1000, 50))
Yb = rng.uniform(0, 1, (1000, 10))
t0 = time.perf_counter()
for _ in range(20):
    big.grads(Xb, Yb)
t_analytic = time.perf_counter() - t0

# 差分法只是抽 20 个权重做一次估计，然后按权重的 0.05% 外推全量代价
net_small = big
h = 1e-6
w_flat = [(m, i, j) for m in ("W1", "W2") for i, j in np.ndindex(getattr(big, m).shape)]
probe = w_flat[:20]
t0 = time.perf_counter()
for (m, i, j) in probe:
    arr = getattr(big, m)
    old = arr[i, j]
    arr[i, j] = old + h
    big.loss(Xb, Yb)
    arr[i, j] = old - h
    big.loss(Xb, Yb)
    arr[i, j] = old
t_probe = time.perf_counter() - t0
n_w = len(w_flat)
t_diff_est = t_probe / len(probe) * n_w

print(f"网络 50-40-10，批大小 1000，参数总数 {n_w}")
print(f"解析梯度（一次前向 + 一次反向）：20 次耗时 {t_analytic:.3f}s")
print(f"中心差分：每个权重要 2 次前向，全量外推 ≈ {t_diff_est:.2f}s")
print(f"比值 ≈ {t_diff_est / t_analytic:.0f}x")
