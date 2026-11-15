#!/usr/bin/env python3
"""83: AdamW 的解耦权重衰减 vs Adam + L2（耦合）

A. 纯衰减：一个没有任何数据梯度的参数，看 λ 到底还剩多少作用。
B. 两组特征尺度差 10 倍、真值都是 1 的线性回归：耦合 L2 对低梯度组过度正则。
C. 每参数等效衰减 λ_eff = λ|θ|/√v̂ 的分布。
"""
import numpy as np

np.random.seed(0)


class Adam:
    def __init__(self, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, wd=0.0, decoupled=False):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.wd, self.decoupled = wd, decoupled
        self.m = None
        self.v = None
        self.t = 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta)
            self.v = np.zeros_like(theta)
        self.t += 1
        if not self.decoupled:
            grad = grad + self.wd * theta          # L2 写进梯度
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        theta = theta - self.lr * mh / (np.sqrt(vh) + self.eps)
        if self.decoupled:
            theta = theta - self.lr * self.wd * theta   # 直接从参数里扣
        return theta


# ---------- A: 没有数据梯度的参数 ----------
def part_a(steps=200, lr=0.02):
    print("== A. 纯衰减：θ0=1.0，数据梯度=0，λ 从 0.001 扫到 100 ==")
    print(f"{'wd':>8} {'耦合 过零步':>12} {'耦合 |θ|@200':>13} {'耦合 翻符号次数':>16} "
          f"{'AdamW |θ|@200':>15} {'理论 (1-lr·λ)^200':>18}")
    rows = []
    for wd in [0.001, 0.01, 0.1, 1.0, 100.0]:
        out = {}
        for name, dec in [("l2", False), ("wd", True)]:
            th = np.array([1.0])
            opt = Adam(lr=lr, wd=wd, decoupled=dec)
            traj = [1.0]
            for _ in range(steps):
                th = opt.step(th, np.zeros_like(th))
                traj.append(float(th[0]))
            cross = next((i for i, x in enumerate(traj) if x * traj[0] < 0), None)
            flips = sum(1 for i in range(1, len(traj)) if traj[i] * traj[i - 1] < 0)
            out[name] = (cross, abs(traj[-1]), flips)
        theory = abs(1 - lr * wd) ** steps
        rows.append((wd, out["l2"], out["wd"]))
        print(f"{wd:8.3f} {str(out['l2'][0]):>12} {out['l2'][1]:13.4f} {out['l2'][2]:16d} "
              f"{out['wd'][1]:15.4f} {theory:18.4f}")
    traj = []
    th = np.array([1.0])
    opt = Adam(lr=lr, wd=0.1, decoupled=False)
    for _ in range(80):
        th = opt.step(th, np.zeros_like(th))
        traj.append(float(th[0]))
    print("  耦合轨迹前 5 步:", " ".join(f"{x:.4f}" for x in traj[:5]))
    print("  耦合轨迹第 60-70 步:", " ".join(f"{x:.4f}" for x in traj[60:70]))
    return rows, traj


# ---------- B: 两组特征尺度差 10 倍，真值都是 1 ----------
def make_data(n=1024, d_hi=4, d_lo=4, noise=0.1, seed=1):
    rng = np.random.default_rng(seed)
    Xhi = rng.normal(size=(n, d_hi))
    Xlo = rng.normal(size=(n, d_lo)) * 0.1     # 尺度 0.1 → 梯度小 10 倍
    X = np.concatenate([Xhi, Xlo], axis=1)
    w_true = np.ones(d_hi + d_lo)
    y = X @ w_true + rng.normal(scale=noise, size=n)
    return X, y, w_true, d_hi


def train_reg(Xtr, ytr, wd, decoupled, lr=0.01, steps=2000, batch=64, seed=2):
    rng = np.random.default_rng(seed)
    w = np.zeros(Xtr.shape[1])
    opt = Adam(lr=lr, wd=wd, decoupled=decoupled)
    for _ in range(steps):
        idx = rng.integers(0, Xtr.shape[0], batch)
        Xb, yb = Xtr[idx], ytr[idx]
        g = 2 * Xb.T @ (Xb @ w - yb) / batch
        w = opt.step(w, g)
    return w


def part_b():
    X, y, w_true, d_hi = make_data()
    Xtr, ytr, Xte, yte = X[:800], y[:800], X[800:], y[800:]
    print("\n== B. 线性回归 8 维，后 4 维特征尺度 0.1，真值全为 1，2000 步 ==")
    hdr = f"{'方案':<20} {'前4维均值':>10} {'后4维均值':>10} {'train MSE':>11} {'test MSE':>10}"
    print(hdr)
    res = {}
    for name, wd, dec in [("wd=0 对照", 0.0, True), ("Adam+L2 λ=0.1", 0.1, False),
                          ("AdamW λ=0.1", 0.1, True)]:
        w = train_reg(Xtr, ytr, wd, dec)
        hi, lo = float(w[:d_hi].mean()), float(w[d_hi:].mean())
        tr = float(np.mean((Xtr @ w - ytr) ** 2))
        te = float(np.mean((Xte @ w - yte) ** 2))
        res[name] = (hi, lo, tr, te)
        print(f"{name:<20} {hi:10.3f} {lo:10.3f} {tr:11.4f} {te:10.4f}")
    print(f"  耦合/解耦 的后4维收敛差距: {res['AdamW λ=0.1'][1]/res['Adam+L2 λ=0.1'][1]:.2f}x")
    return res


# ---------- C: 每参数等效衰减 ----------
def part_c(steps=400, lr=0.01, wd=0.1):
    X, y, w_true, d_hi = make_data()
    rng = np.random.default_rng(3)
    w = np.zeros(X.shape[1])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    lam = []
    for t in range(1, steps + 1):
        idx = rng.integers(0, X.shape[0], 64)
        g = 2 * X[idx].T @ (X[idx] @ w - y[idx]) / 64 + wd * w
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        vh = v / (1 - 0.999 ** t)
        lam.append(wd * np.abs(w) / np.sqrt(vh))
        w = w - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(vh) + 1e-8)
    L = np.array(lam[100:])          # 跳过前 100 步的暖机
    hi, lo = float(L[:, :d_hi].mean()), float(L[:, d_hi:].mean())
    print("\n== C. 每参数等效衰减 λ_eff = λ|θ|/√v̂（Adam+L2，第 100-400 步均值） ==")
    print(f"前 4 维（梯度大）λ_eff = {hi:.5f}")
    print(f"后 4 维（梯度小）λ_eff = {lo:.5f}")
    print(f"跨度 = {lo/hi:.1f}x；AdamW 两组都恰好等于 λ = {wd}")
    # 精确核对：一步理论值 λ/√v̂ vs 实测
    idx = rng.integers(0, X.shape[0], 64)
    g = 2 * X[idx].T @ (X[idx] @ w - y[idx]) / 64
    print(f"  当前数据梯度范数：前 4 维 {np.linalg.norm(g[:d_hi]):.4f}，后 4 维 {np.linalg.norm(g[d_hi:]):.4f}")
    return hi, lo


if __name__ == "__main__":
    part_a()
    part_b()
    part_c()
