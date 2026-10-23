"""60 · Matryoshka 表示学习（MRL）最小实验
线性编码器 + 对比损失（InfoNCE），numpy 手写梯度。
对照组：只在 64 维全空间上训练；实验组：在 {8,16,32,64} 四个前缀上都算同一个损失。
"""
import numpy as np

rng = np.random.default_rng(0)

# ---------- 数据：4096 篇"文档"，三级结构 + 稀疏内容特征 ----------
N_TOPIC, N_SUB, N_DOC = 8, 8, 64
N = N_TOPIC * N_SUB * N_DOC              # 4096 篇
N_CONTENT, ACTIVE = 64, 6                # 共享词表 64 维，每篇 6 个特征
D = N_TOPIC + N_TOPIC * N_SUB + N_CONTENT  # 8 + 64 + 64 = 136
DIM = 64                                 # 目标向量维度

topic = np.repeat(np.arange(N_TOPIC), N_SUB * N_DOC)
sub = np.tile(np.repeat(np.arange(N_SUB), N_DOC), N_TOPIC)
local = np.tile(np.arange(N_DOC), N_TOPIC * N_SUB)
rows = np.arange(N)
group = topic * N_SUB + sub


def draw_dirichlet(k, alpha, r):
    return r.dirichlet(np.full(k, alpha))


# 主题偏好 → 子话题偏好（子话题 = 一半主题偏好 + 一半自身偏好，制造层级）
topic_pref = np.array([draw_dirichlet(N_CONTENT, 0.25, rng) for _ in range(N_TOPIC)])
sub_pref = np.array([0.5 * topic_pref[t] + 0.5 * draw_dirichlet(N_CONTENT, 0.25, rng)
                     for t in range(N_TOPIC) for _ in range(N_SUB)])

X = np.zeros((N, D))
for i in range(N):
    X[i, topic[i]] = 1.0
    X[i, N_TOPIC + group[i]] = 1.0
    p = sub_pref[group[i]] / sub_pref[group[i]].sum()
    feat = rng.choice(N_CONTENT, ACTIVE, replace=False, p=p)
    X[i, N_TOPIC + N_TOPIC * N_SUB + feat] = 1.0


def noisy_view(Xv, drop=0.25, seed=None):
    """模拟用户 query：内容特征随机丢一部分（用词不同），层级标签保留。"""
    r = np.random.default_rng(seed)
    keep = (r.random(Xv.shape) > drop).astype(float)
    return Xv * keep


# ---------- 前向 / 反向 ----------
def prefix_norm(Z, p):
    z = Z[:, :p]
    n = np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
    return z / n, n


def pair_loss_grad(E, Xq, Xd, p, tau):
    Zq, Zd = Xq @ E.T, Xd @ E.T
    eq, nq = prefix_norm(Zq, p)
    ed, nd = prefix_norm(Zd, p)
    S = eq @ ed.T / tau
    S = S - S.max(1, keepdims=True)
    P = np.exp(S)
    P /= P.sum(1, keepdims=True)
    b = len(Xq)
    loss = -np.mean(np.log(P[rows[:b], rows[:b]] + 1e-12))
    dS = P
    dS[rows[:b], rows[:b]] -= 1.0
    dS /= b * tau
    deq, ded = dS @ ed, dS.T @ eq
    dzq = (deq - eq * (deq * eq).sum(1, keepdims=True)) / nq
    dzd = (ded - ed * (ded * ed).sum(1, keepdims=True)) / nd
    dZq, dZd = np.zeros_like(Zq), np.zeros_like(Zd)
    dZq[:, :p], dZd[:, :p] = dzq, dzd
    return loss, dZq.T @ Xq + dZd.T @ Xd


def train(mode, steps=900, prefixes=(8, 16, 32, 64), tau=0.08, lr=0.03, bs=192):
    E = rng.normal(0, 0.1, size=(DIM, D))
    m, v = np.zeros_like(E), np.zeros_like(E)
    ps = prefixes if mode == "mrl" else (DIM,)
    for t in range(1, steps + 1):
        pick = rng.choice(N, bs, replace=False)
        Xd = X[pick]
        Xq = noisy_view(Xd, 0.25, seed=int(rng.integers(1 << 30)))
        g = np.zeros_like(E)
        for p in ps:
            _, gp = pair_loss_grad(E, Xq, Xd, p, tau)
            g += gp / len(ps)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        E -= lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return E


def evaluate(E, p, drop=0.25, seed=7):
    Xq = noisy_view(X, drop, seed=seed)
    eq, _ = prefix_norm(Xq @ E.T, p)
    ed, _ = prefix_norm(X @ E.T, p)
    S = eq @ ed.T
    order = np.argsort(-S, axis=1)
    rank = np.argmax(order == rows[:, None], axis=1)
    return float(np.mean(rank < 1)), float(np.mean(rank < 50)), float(np.median(rank) + 1)


def cascade(E, p, K, seed=7):
    """8 维粗筛 top-K，再用全额向量重排（生产里常见的两段式）。"""
    Xq = noisy_view(X, 0.25, seed=seed)
    Zq, Zd = Xq @ E.T, X @ E.T
    eq, _ = prefix_norm(Zq, p)
    ed, _ = prefix_norm(Zd, p)
    cand = np.argsort(-(eq @ ed.T), axis=1)[:, :K]
    fq = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-9)
    fd = Zd / (np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-9)
    Sfull = np.take_along_axis(fq @ fd.T, cand, axis=1)
    hit_cand = (cand == rows[:, None]).any(1)
    best = cand[rows, np.argmax(Sfull, axis=1)]
    return float(np.mean(best == rows)), float(np.mean(hit_cand))


def fit_pca(E, out_dim):
    """后处理对照：把已训好的 64 维向量做 PCA 压到 out_dim 维（不动模型权重）。"""
    Z = X @ E.T
    mu = Z.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(Z - mu, full_matrices=False)
    return mu, Vt[:out_dim].T


def evaluate_pca(E, mu, P, seed=7):
    Xq = noisy_view(X, 0.25, seed=seed)
    Zq, Zd = (Xq @ E.T - mu) @ P, (X @ E.T - mu) @ P
    eq = Zq / (np.linalg.norm(Zq, axis=1, keepdims=True) + 1e-9)
    ed = Zd / (np.linalg.norm(Zd, axis=1, keepdims=True) + 1e-9)
    order = np.argsort(-(eq @ ed.T), axis=1)
    rank = np.argmax(order == rows[:, None], axis=1)
    return float(np.mean(rank < 1)), float(np.mean(rank < 10)), float(np.median(rank) + 1)


if __name__ == "__main__":
    res, enc = {}, {}
    for mode in ("standard", "mrl"):
        E = train(mode)
        enc[mode] = E
        res[mode] = {p: evaluate(E, p) for p in (8, 16, 32, 48, 64)}
        if mode == "standard":
            for od in (8, 16, 32):
                mu, P = fit_pca(E, od)
                res.setdefault("standard+pca", {})[od] = evaluate_pca(E, mu, P)
        print(f"\n=== {mode} ===")
        print(f"{'截断到':>6} {'R@1':>7} {'R@50':>7} {'中位名次':>8}")
        for p, (r1, r50, med) in res[mode].items():
            print(f"{p:>6} {r1:>7.3f} {r50:>7.3f} {med:>8.0f}")

    print("\n--- 后处理 PCA 对照（standard 模型不动，只换投影） ---")
    for od in (8, 16, 32):
        r1, r50, med = res["standard+pca"][od]
        print(f"PCA→{od:>2}: R@1={r1:.3f} R@50={r50:.3f} 中位名次={med:.0f}")

    print("\n--- 两段式：8 维粗筛 top-K 再用全额向量重排 ---")
    for mode in ("standard", "mrl"):
        for K in (20, 50):
            r1, hit = cascade(enc[mode], 8, K)
            print(f"{mode:9s} K={K:>3}: 候选含正样本 {hit:.3f} → 重排后 R@1={r1:.3f}")

    print("\n--- 相对满维 R@1 的保留率 ---")
    print(f"{'截断到':>6} {'standard':>10} {'mrl':>10}")
    for p in (8, 16, 32):
        print(f"{p:>6} {res['standard'][p][0] / res['standard'][64][0]:>10.2f} "
              f"{res['mrl'][p][0] / res['mrl'][64][0]:>10.2f}")

    print("\n--- 存储成本（100 万条 × float32） ---")
    for d in (1536, 1024, 512, 256, 128, 64, 32):
        print(f"{d:>5} 维: {d * 4 * 1e6 / 1e9:>6.2f} GB")
