"""ai-concepts-86 demo: InfoNCE 的批内负样本、假负样本与"有效批大小"。

合成语料（240 个主题 / 1440 文档 / 1920 查询）+ 一个两层 MLP 双塔编码器，
纯 CPU 约 4~5 分钟跑完（6 个 arm × 2000 步）。这是机制演示，不是真实 embedding 模型训练。
"""
import numpy as np

K, PER_DOC, PER_Q = 240, 6, 8      # 240 主题 / 1440 文档 / 1920 查询
D_IN, H, D_OUT = 64, 64, 24
SIG, TAU = 1.6, 0.07               # 主题内噪声 / 温度

def build(c, n, seed, sig):
    r = np.random.default_rng(seed); X, y = [], []
    for t in range(K):
        X.append(c[t] + sig * r.normal(size=(n, D_IN)) / np.sqrt(D_IN)); y += [t] * n
    X = np.concatenate(X); X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X, np.array(y)

C = np.random.default_rng(3).normal(size=(K, D_IN))
C /= np.linalg.norm(C, axis=1, keepdims=True)
D, Dy = build(C, PER_DOC, 1, SIG)   # 语料
Q, Qy = build(C, PER_Q, 2, SIG)     # 训练查询
Qt, Qty = build(C, 4, 3, SIG)       # 留出查询

def enc(P, X):
    h = np.maximum(X @ P[0] + P[1], 0.0)
    z = h @ P[2] + P[3]
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

def recall(P, k=1):
    s = enc(P, Qt) @ enc(P, D).T
    top = np.argsort(-s, axis=1)[:, :k]
    return float(np.mean([(Dy[top[i]] == Qty[i]).any() for i in range(len(Qt))]))

def fn_mass(P, B=512, seed=99):
    """同一批 B 条里，与查询同主题（本该是正样本）的文档吃掉了多少分母权重"""
    r = np.random.default_rng(seed)
    idx = r.choice(len(Q), B, replace=False)
    didx = Qy[idx] * PER_DOC + r.integers(0, PER_DOC, size=B)
    S = enc(P, Q[idx]) @ enc(P, D[didx]).T / TAU
    Pm = np.exp(S - S.max(1, keepdims=True)); Pm /= Pm.sum(1, keepdims=True)
    bad = (Qy[idx][:, None] == Dy[didx][None, :]) & ~np.eye(B, dtype=bool)
    return float(Pm[bad].sum() / len(idx))

HOT = 20  # 20 个热门主题占七成流量

def sample_idx(r, B, skew):
    if not skew:
        return r.choice(len(Q), B, replace=False)
    t = np.where(r.random(B) < 0.7, r.integers(0, HOT, B), r.integers(0, K, B))
    return (t * PER_Q + r.integers(0, PER_Q, B)).astype(int)

def train(B, steps=2000, lr=2e-3, mask_same=False, skew=False, seed=1):
    r = np.random.default_rng(seed)
    P = [r.normal(size=(D_IN, H)) * np.sqrt(2 / D_IN), np.zeros(H),
         r.normal(size=(H, D_OUT)) * np.sqrt(2 / H), np.zeros(D_OUT)]
    mom = [np.zeros_like(p) for p in P]; vel = [np.zeros_like(p) for p in P]
    fn_pairs = 0
    for step in range(1, steps + 1):
        idx = sample_idx(r, B, skew)
        didx = Qy[idx] * PER_DOC + r.integers(0, PER_DOC, size=B)
        q, d = Q[idx], D[didx]
        hq = np.maximum(q @ P[0] + P[1], 0.0); hd = np.maximum(d @ P[0] + P[1], 0.0)
        zq, zd = enc(P, q), enc(P, d)
        S = zq @ zd.T / TAU
        # 诊断量：批内和查询同主题、却不是它自己正样本的条数
        fn_pairs += int(((Qy[idx][:, None] == Dy[didx][None, :]) & ~np.eye(B, dtype=bool)).sum())
        if mask_same:      # oracle 上界：把假负样本从分母里摘掉（保留对角线上的正样本）
            bad = (Qy[idx][:, None] == Dy[didx][None, :]) & ~np.eye(B, dtype=bool)
            S = np.where(bad, -1e4, S)
        Pm = np.exp(S - S.max(1, keepdims=True)); Pm /= Pm.sum(1, keepdims=True)
        loss = -np.log(np.diag(Pm) + 1e-12).mean()
        G = Pm.copy(); G[np.arange(B), np.arange(B)] -= 1.0; G /= B
        dzq, dzd = (G @ zd) / TAU, (G.T @ zq) / TAU
        dzq -= zq * (dzq * zq).sum(1, keepdims=True)      # 反向穿过 L2 归一化
        dzd -= zd * (dzd * zd).sum(1, keepdims=True)
        dhq, dhd = (dzq @ P[2].T) * (hq > 0), (dzd @ P[2].T) * (hd > 0)
        gW1 = q.T @ dhq + d.T @ dhd
        gb1 = dhq.sum(0) + dhd.sum(0)
        gW2 = hq.T @ dzq + hd.T @ dzd
        gb2 = dzq.sum(0) + dzd.sum(0)
        for i, g in enumerate([gW1, gb1, gW2, gb2]):
            mom[i] = 0.9 * mom[i] + 0.1 * g; vel[i] = 0.999 * vel[i] + 0.001 * g * g
            P[i] -= lr * (mom[i] / (1 - 0.9 ** step)) / (np.sqrt(vel[i] / (1 - 0.999 ** step)) + 1e-8)
    return P, fn_pairs / steps, loss

if __name__ == "__main__":
    print(f"{'arm':26s} {'负样本/步':>9s} {'同主题/步':>9s} {'loss':>6s} {'R@1':>6s} {'R@10':>6s}")
    arms = [("B=32", dict(B=32)), ("B=128", dict(B=128)), ("B=512", dict(B=512)),
            ("B=512 屏蔽假负样本", dict(B=512, mask_same=True)),
            ("B=512 热点集中", dict(B=512, skew=True)),
            ("B=512 热点集中+屏蔽", dict(B=512, skew=True, mask_same=True))]
    keep = None
    for name, kw in arms:
        P, fn, loss = train(**kw)
        if kw["B"] == 512 and not kw.get("mask_same") and not kw.get("skew"):
            keep = P
        print(f"{name:26s} {kw['B']-1:9d} {fn:9.2f} {loss:6.3f} {recall(P,1):6.3f} {recall(P,10):6.3f}", flush=True)
    print(f"\n在 B=512 训练出的模型上，假负样本吃掉同一批 512 条分母权重的 {fn_mass(keep)*100:.1f}%")
