"""Muon vs AdamW: orthogonalized momentum on hidden layers.

纯 numpy,无 GPU。验证三件事:
1) Newton-Schulz 五次迭代能把任意矩阵的奇异值压平到 ~1
2) Muon 每层更新 RMS 恒定(≈0.2),Adam 的层间 update RMS 相差数倍
3) 小 MLP 上收敛速度
"""
import numpy as np

rng = np.random.default_rng(0)


# ---------- Newton-Schulz 正交化 (Keller Jordan / Moonlight 的五次迭代) ----------
def zeropower_via_ns(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.astype(np.float32).copy()
    X /= (np.linalg.norm(X) + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


# ---------- 模型: 3 层 MLP, 手工反向传播 ----------
D_IN, D_H, N_CLS, BATCH = 64, 96, 10, 128
CENTERS = rng.normal(0, 1.0, (N_CLS, D_IN)).astype(np.float32)  # 固定簇心, 否则任务不可学


def init_params():
    return [
        rng.normal(0, np.sqrt(2 / D_IN), (D_H, D_IN)).astype(np.float32),
        rng.normal(0, np.sqrt(2 / D_H), (D_H, D_H)).astype(np.float32),
        rng.normal(0, np.sqrt(1 / D_H), (N_CLS, D_H)).astype(np.float32),
        np.zeros(D_H, np.float32),
        np.zeros(D_H, np.float32),
        np.zeros(N_CLS, np.float32),
    ]


def make_batch(n):
    # 10 类固定簇心 + 噪声: 任务可学但需要非线性边界
    y = rng.integers(0, N_CLS, n)
    x = CENTERS[y] + rng.normal(0, 2.2, (n, D_IN)).astype(np.float32)
    return x, y


def forward(p, x):
    W1, W2, W3, b1, b2, b3 = p
    h1 = np.maximum(x @ W1.T + b1, 0)
    h2 = np.maximum(h1 @ W2.T + b2, 0)
    logits = h2 @ W3.T + b3
    z = logits - logits.max(1, keepdims=True)
    prob = np.exp(z) / np.exp(z).sum(1, keepdims=True)
    return prob, (x, h1, h2, W2, W3)


def loss_and_grad(p, x, y):
    prob, (x0, h1, h2, W2, W3) = forward(p, x)
    n = len(y)
    loss = -np.log(prob[np.arange(n), y] + 1e-9).mean()
    d = prob.copy()
    d[np.arange(n), y] -= 1
    d /= n
    gW3 = d.T @ h2
    gb3 = d.sum(0)
    dh2 = d @ W3
    dh2[h2 <= 0] = 0
    gW2 = dh2.T @ h1
    gb2 = dh2.sum(0)
    dh1 = dh2 @ W2
    dh1[h1 <= 0] = 0
    gW1 = dh1.T @ x0
    gb1 = dh1.sum(0)
    return loss, [gW1, gW2, gW3, gb1, gb2, gb3]


# ---------- 两个优化器 ----------
class AdamW:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.95, wd=0.01):
        self.lr, self.b1, self.b2, self.wd = lr, b1, b2, wd
        self.m = [np.zeros_like(q) for q in params]
        self.v = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        rms, us = [], []
        for i, (q, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            u = mh / (np.sqrt(vh) + 1e-8)
            q -= self.lr * (u + self.wd * q)
            us.append(self.lr * u)
            rms.append(float(np.sqrt((self.lr * u) ** 2).mean()))
        self.last_u = us
        return rms


class Muon:
    """2D 隐藏层权重用正交化动量, 其余(输出层/偏置)退回 AdamW 风格。"""

    def __init__(self, params, lr=2e-2, mom=0.95, wd=0.01):
        self.lr, self.mom, self.wd = lr, mom, wd
        self.mom_buf = [np.zeros_like(q) for q in params]
        self.aux = [np.zeros_like(q) for q in params]
        self.aux2 = [np.zeros_like(q) for q in params]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        rms, us = [], []
        for i, (q, g) in enumerate(zip(params, grads)):
            self.mom_buf[i] = self.mom * self.mom_buf[i] + g
            if q.ndim == 2 and q.shape[0] > 1 and q.shape[1] > 1 and i < 2:  # 仅隐藏层
                o = zeropower_via_ns(self.mom_buf[i])
                scale = 0.2 * np.sqrt(max(o.shape))
                u = o * scale
            else:  # 输出层与偏置: Adam
                self.aux[i] = 0.9 * self.aux[i] + 0.1 * g
                self.aux2[i] = 0.95 * self.aux2[i] + 0.05 * g * g
                u = (self.aux[i] / (1 - 0.9 ** self.t)) / (
                    np.sqrt(self.aux2[i] / (1 - 0.95 ** self.t)) + 1e-8)
            q -= self.lr * (u + self.wd * q)
            us.append(self.lr * u)
            rms.append(float(np.sqrt((self.lr * u) ** 2).mean()))
        self.last_u = us
        return rms


def stable_rank(M):
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv.sum() ** 2 / (sv ** 2).sum())


def train(opt_cls, steps=1200, **kw):
    p = init_params()
    opt = opt_cls(p, **kw)
    rms_at_100, u_at_100, hist = None, None, []
    for s in range(steps):
        x, y = make_batch(BATCH)
        loss, g = loss_and_grad(p, x, y)
        rms = opt.step(p, g)
        if s == 100:
            rms_at_100, u_at_100 = rms, [q.copy() for q in opt.last_u[:2]]
        if (s + 1) % 200 == 0:
            xv, yv = make_batch(2000)
            vl = -np.log(forward(p, xv)[0][np.arange(2000), yv] + 1e-9).mean()
            hist.append((s + 1, float(loss), float(vl)))
    return p, rms_at_100, u_at_100, hist


# ---------- 1) 正交化本身 ----------
print("== Newton-Schulz: 奇异值谱 ==")
G = rng.normal(0, 1, (96, 64)).astype(np.float32) @ np.diag(
    np.linspace(0.05, 4.0, 64)).astype(np.float32)  # 故意做成病态矩阵
before = np.linalg.svd(G, compute_uv=False)
after = np.linalg.svd(zeropower_via_ns(G), compute_uv=False)
print(f"输入  : min={before.min():.3f} max={before.max():.3f} "
      f"条件数={before.max()/before.min():.1f}")
print(f"输出  : min={after.min():.3f} max={after.max():.3f} "
      f"条件数={after.max()/after.min():.2f}")
Gs = zeropower_via_ns(G)
print(f"更新 RMS(未缩放)= {np.sqrt((Gs**2).mean()):.4f} "
      f"(理论 sqrt(1/max(A,B))={np.sqrt(1/96):.4f})")
print(f"stable rank 输入 {stable_rank(G):.1f} -> 输出 {stable_rank(Gs):.1f} / 64")

# ---------- 2)(3) MLP 训练对比 (各自扫 lr, 取验证集最好) ----------
for name, cls, lrs in [("AdamW", AdamW, [1e-3, 3e-3, 1e-2]),
                       ("Muon ", Muon, [5e-3, 1e-2, 2e-2, 4e-2])]:
    runs = []
    for lr in lrs:
        res = train(cls, lr=lr)
        runs.append((res, lr))
    (p, rms, us, hist), lr = min(runs, key=lambda t: t[0][3][-1][2])
    print(f"\n== {name} best lr={lr:g}, 每层更新 RMS @step100 ==")
    for lab, r in zip(["W1", "W2", "W3(Adam)", "b1", "b2", "b3"], rms):
        print(f"  {lab:8s} {r:.4f}")
    for lab, M in zip(["W1", "W2"], us):
        sv = np.linalg.svd(M, compute_uv=False)
        print(f"  {lab} 更新矩阵 stable rank = {stable_rank(M):.1f} / {min(M.shape)} "
              f"| top1 奇异值占比 = {sv[0]**2/(sv**2).sum():.1%}")
    print(f"  曲线 (step, train, val): {[(s, round(a,3), round(b,3)) for s,a,b in hist]}")
    print(f"  全部 lr 的终值 val: {[(lr, round(r[3][-1][2],3)) for r, lr in runs]}")

# ---------- 4) 病态线性回归: 更新谱形状直接决定收敛 ----------
def cond_experiment(opt_cls, lr, steps=200):
    U, _ = np.linalg.qr(rng.normal(0, 1, (64, 64)))
    V, _ = np.linalg.qr(rng.normal(0, 1, (64, 64)))
    S = np.logspace(0, 3, 64)                      # 条件数 1000
    A = (U * S) @ V.T
    A = (A / np.sqrt((A ** 2).mean())).astype(np.float32)  # 归一化到 RMS=1
    W_star = rng.normal(0, 0.1, (64, 64)).astype(np.float32)
    B = A @ W_star
    W = np.zeros((64, 64), np.float32)
    opt = opt_cls([W], lr=lr)
    hist = []
    init = float(((W @ A - B) ** 2).mean())
    for s in range(steps):
        R = W @ A - B
        G = (R @ A.T) / B.size
        opt.step([W], [G])
        if (s + 1) % 50 == 0:
            hist.append(round(float((R ** 2).mean()), 8))
    return float((R ** 2).mean()), hist, init


print("\n== 病态线性回归 (cond(A)=1000), epoch 子集损失 ==")
best = {}
INIT = None
for name, cls, lrs in [("AdamW", AdamW, [1e-3, 1e-2, 1e-1]),
                       ("Muon ", Muon, [1e-3, 3e-3, 1e-2, 3e-2])]:
    runs = [(cond_experiment(cls, lr), lr) for lr in lrs]
    (loss, hist, init), lr = min(runs, key=lambda t: t[0][0])
    best[name] = loss
    INIT = init
    print(f"{name} best lr={lr:g}: 初始 MSE={init:.3e} -> 终值 MSE={loss:.3e}  曲线={hist}")
print(f"\n最优 MSE 比值 Adam/Muon = {best['AdamW']/best['Muon ']:.1f}x")
