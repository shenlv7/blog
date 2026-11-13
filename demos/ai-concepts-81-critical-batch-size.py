"""critical batch size / gradient noise scale -- pure numpy, CPU only.

A) B_simple = tr(Sigma)/||G||^2 measured at 5 checkpoints of one SGD run
B) direct noise curve: ||g_B - G||^2 / ||G||^2 for B = 1..1024 (200 draws each)
C) train at 9 batch sizes with lr ~ B, record steps and samples to reach eval loss

cost proxy = samples seen (steps * B). this is a CPU toy, not a GPU benchmark.
"""
import numpy as np
import time

rng = np.random.default_rng(7)
n, d, H = 8192, 16, 64
X = rng.normal(size=(n, d))
w_true = rng.normal(size=d) / np.sqrt(d)
Y = np.tanh(2.0 * (X @ w_true)) + 0.05 * rng.normal(size=n)
Xv, Yv = X[:1024], Y[:1024]


def init(seed=0):
    r = np.random.default_rng(seed)
    return [r.normal(size=(d, H)) / np.sqrt(d), np.zeros(H),
            r.normal(size=H) / np.sqrt(H), np.zeros(1)]


def grad(P, Xb, Yb):
    h = np.tanh(Xb @ P[0] + P[1])
    r = h @ P[2] + P[3] - Yb
    dp = r / len(Yb)
    dz = np.outer(dp, P[2]) * (1 - h ** 2)
    return [Xb.T @ dz, dz.sum(0), h.T @ dp, np.array([dp.sum()])]


def ev(P, Xb=Xv, Yb=Yv):
    return 0.5 * np.mean((np.tanh(Xb @ P[0] + P[1]) @ P[2] + P[3] - Yb) ** 2)


def per_sample(P, Xb, Yb):
    h = np.tanh(Xb @ P[0] + P[1])
    r = h @ P[2] + P[3] - Yb
    dz = (r[:, None] * P[2][None, :]) * (1 - h ** 2)
    gW1 = np.einsum('ni,nj->nij', Xb, dz).reshape(len(Yb), -1)
    return np.concatenate([gW1, dz, r[:, None] * h, r[:, None]], axis=1)


def b_simple(P):
    idx = rng.choice(n, 4096, replace=False)
    S = per_sample(P, X[idx], Y[idx])
    G = S.mean(0)
    return ((S - G) ** 2).sum(1).mean() / (G @ G), G


def train(bs, lr, max_steps=20000, target=0.003):
    P = init(0)
    mom = [np.zeros_like(w) for w in P]
    r = np.random.default_rng(1)
    for step in range(1, max_steps + 1):
        ii = r.integers(0, n, bs)
        g = grad(P, X[ii], Y[ii])
        for k in range(4):
            mom[k] = 0.9 * mom[k] + g[k]
            P[k] -= lr * mom[k]
        if step % 10 == 0 and ev(P) <= target:
            return step, step * bs, ev(P)
    return None, max_steps * bs, ev(P)


# ---- A: B_simple along the run ----
print("== A. B_simple 在一次训练中的漂移 (bs=32, lr=0.01) ==")
print(f"{'step':>6} {'eval_loss':>10} {'||G||^2':>10} {'tr(Sigma)':>11} {'B_simple':>9}")
P = init(0)
mom = [np.zeros_like(w) for w in P]
r = np.random.default_rng(1)
marks = {0, 100, 300, 700, 2000}
ckpt = {}
for step in range(0, 2001):
    if step in marks:
        bs_val, G = b_simple(P)
        idx = rng.choice(n, 4096, replace=False)
        S = per_sample(P, X[idx], Y[idx])
        tr = ((S - S.mean(0)) ** 2).sum(1).mean()
        print(f"{step:6d} {ev(P):10.5f} {G@G:10.5f} {tr:11.4f} {bs_val:9.1f}")
        if step == 300:
            ckpt = [w.copy() for w in P]
    ii = r.integers(0, n, 32)
    g = grad(P, X[ii], Y[ii])
    for k in range(4):
        mom[k] = 0.9 * mom[k] + g[k]
        P[k] -= 0.01 * mom[k]

# ---- B: direct noise curve at the step-300 checkpoint ----
bs300, G300 = b_simple(ckpt)
print(f"\n== B. 噪声曲线 @ step300 checkpoint (B_simple={bs300:.1f}) ==")
print(f"{'batch':>6} {'rel_noise':>11} {'rel_noise*B':>12} {'pred B_simple':>13}")
full = grad(ckpt, X, Y)
Gf = np.concatenate([g.ravel() for g in full])
for B in [1, 4, 16, 64, 256, 1024]:
    err = 0.0
    for _ in range(200):
        ii = rng.integers(0, n, B)
        gb = np.concatenate([g.ravel() for g in grad(ckpt, X[ii], Y[ii])])
        err += ((gb - Gf) ** 2).sum()
    rel = (err / 200) / (Gf @ Gf)
    print(f"{B:6d} {rel:11.4f} {rel*B:12.2f} {rel*B:13.1f}")

# ---- C: train at nine batch sizes, lr scaled linearly ----
print("\n== C. 九个 batch，lr = 0.01 * B/32，跑到 eval loss 0.003 ==")
print(f"{'batch':>6} {'lr':>7} {'steps':>7} {'samples':>9} {'vs B=32':>9} {'ms/step':>8}")
BS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
ref = None
for B in BS:
    lr = 0.01 * B / 32
    t0 = time.time()
    st, seen, e = train(B, lr)
    ms = (time.time() - t0) * 1000 / max(st or 20000, 1)
    if B == 32:
        ref = seen
    ratio = f"{seen/ref:5.2f}x" if ref else "-"
    print(f"{B:6d} {lr:7.3f} {str(st):>7} {seen:9d} {ratio:>9} {ms:8.2f}")
