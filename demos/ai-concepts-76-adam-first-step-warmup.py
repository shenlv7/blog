"""概念卡片 76 · Adam 的第一步：为什么必须先 warmup

2-64-1 tanh MLP，two-moons 二分类（10% 标签翻转），257 个参数，纯 numpy 单核 CPU，
1.5 秒跑完。量的全是几何量（位移范数、方向余弦、单步 loss 变化），不是最终精度：
三步以内的量是确定性的，200 步以上的 loss 方差大于效应本身。

对应文章：/blog/posts/ai-concepts-76-adam-first-step-warmup/
"""
import numpy as np

D, H = 2, 64
NPARAM = 2 * 64 + 64 + 64 + 1
rng = np.random.default_rng(7)
N = 16000
SHAPES = [(D, H), (H,), (H,), (1,)]


def make_moons(n, noise=0.12, flip=0.0, seed=0):
    r = np.random.default_rng(seed)
    t = r.uniform(0, np.pi, n // 2)
    X = np.concatenate([np.stack([np.cos(t), np.sin(t) - 0.5], 1),
                        np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)])
    y = np.concatenate([np.zeros(n // 2), np.ones(n // 2)])
    i = r.permutation(n)
    X, y = X[i] + r.normal(0, noise, (n, 2)), y[i]
    if flip:
        y = np.where(r.random(n) < flip, 1 - y, y)
    return (X - X.mean(0)) / X.std(0), y.astype(float)


Xtr, ytr = make_moons(N, flip=0.10, seed=1)
Xev, yev = make_moons(4000, seed=2)


def init(seed=0):
    r = np.random.default_rng(seed)
    return [r.normal(0, 1.0, (D, H)), np.zeros(H),
            r.normal(0, 1 / np.sqrt(H), H), np.zeros(1)]


def flat(p):
    return np.concatenate([a.ravel() for a in p])


def unflat(v):
    out, i = [], 0
    for s in SHAPES:
        k = int(np.prod(s))
        out.append(v[i:i + k].reshape(s))
        i += k
    return out


def grads(p, X, y):
    h = np.tanh(X @ p[0] + p[1])
    pr = 1 / (1 + np.exp(-(h @ p[2] + p[3][0])))
    g = (pr - y) / len(X)
    gz = (g[:, None] * (1 - h ** 2)) * p[2][None, :]
    return np.concatenate([(X.T @ gz).ravel(), gz.sum(0), h.T @ g, [g.sum()]])


def loss_of(p, X, y):
    h = np.tanh(X @ p[0] + p[1])
    pr = np.clip(1 / (1 + np.exp(-(h @ p[2] + p[3][0]))), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(pr) + (1 - y) * np.log(1 - pr)))


class Adam:
    """标准 Adam（β1=.9, β2=.999）。step() 返回本步更新向量 u，即参数位移。"""
    def __init__(self, p):
        self.m = np.zeros_like(flat(p))
        self.v = np.zeros_like(flat(p))
        self.t = 0

    def step(self, g, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        self.m = b1 * self.m + (1 - b1) * g
        self.v = b2 * self.v + (1 - b2) * g ** 2
        mhat = self.m / (1 - b1 ** self.t)
        vhat = self.v / (1 - b2 ** self.t)
        return lr * mhat / (np.sqrt(vhat) + eps)


norm0 = float(np.linalg.norm(flat(init(0))))
g64 = grads(init(0), Xtr[:64], ytr[:64])
L0 = loss_of(init(0), Xev, yev)
print(f'参数 d={NPARAM}，初始参数范数 |θ₀|={norm0:.2f}，初始 eval loss={L0:.4f}')

print('\n【一】第 1 步的位移：Adam 恒等于 lr·√d，SGD 只是 lr·|g|')
print('    lr      SGD |Δθ|   Adam |Δθ|   Adam/SGD   Adam|Δθ|/|θ₀|')
for lr in (0.01, 0.1, 0.3):
    ns = np.linalg.norm(lr * g64)
    na = np.linalg.norm(Adam(init(0)).step(g64, lr))
    print(f'  {lr:6.2f}   {ns:9.5f}   {na:9.4f}  {na / ns:8.1f}x      {na / norm0 * 100:6.1f}%')

print(f'\n【二】位移与梯度大小、batch 大小都无关（lr=0.3，理论 lr·√d={0.3 * np.sqrt(NPARAM):.4f}）')
for s in (1e-3, 1.0, 1e3):
    print(f'  梯度 ×{s:<7g} → {np.linalg.norm(Adam(init(0)).step(g64 * s, 0.3)):.4f}')
for B in (8, 64, 2048):
    print(f'  batch={B:<6d} → {np.linalg.norm(Adam(init(0)).step(grads(init(0), Xtr[:B], ytr[:B]), 0.3)):.4f}')

print('\n【三】走完这一步之后，eval loss 变成多少')
for lr in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
    p = init(0)
    p1 = unflat(flat(p) - Adam(p).step(g64, lr))
    l = loss_of(p1, Xev, yev)
    print(f'  lr={lr:6.3f}  |Δθ|={lr * np.sqrt(NPARAM):8.4f}  eval {l:.4f}  ({l - L0:+.4f})')
for W in (50, 200):
    p = init(0)
    p1 = unflat(flat(p) - Adam(p).step(g64, 0.3 / W))
    print(f'  warmup W={W:3d}（首步等效 lr={0.3 / W:.4f}）→ eval {loss_of(p1, Xev, yev):.4f}')

print('\n【四】首步方向可信度：与 8192 样本全量梯度首步的余弦相似度')
pref = init(0)
uref = Adam(pref).step(grads(pref, Xtr[:8192], ytr[:8192]), 0.3)
uref /= np.linalg.norm(uref)
for B in (8, 64, 512, 2048):
    cs = []
    for _ in range(20):
        p = init(0)
        i = rng.integers(0, N, B)
        u = Adam(p).step(grads(p, Xtr[i], ytr[i]), 0.3)
        cs.append(float(u @ uref / np.linalg.norm(u)))
    print(f'  batch={B:<5d}  cos={np.mean(cs):.3f} ± {np.std(cs) / np.sqrt(20):.3f}')
