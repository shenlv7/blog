"""1957 感知机的原始学习规则 —— 计数强化，而不是权重修正。

结构按 Rosenblatt 1957 年报告 85-460-1 (Project PARA) 的描述搭：
  S 系统：20x20 光电池（本篇用布尔阵列代替）
  A 系统：固定随机 ±1 连线 + 阈值 θ；每个 A 单元有一个随历史累积的"值" v
  R 系统：两个互斥的响应单元，读"被该刺激激活的 A 单元的值"的均值，
          与阈值 θ_R 比较，互相抑制，只有一个能亮

1957 年报告的原话大意：每次"刺激与响应同时出现"，连接这个刺激到这个响应的
那部分 A 单元的"聚集的值"就增加一点。也就是 v[A, r] += 1 —— 只有加法，没有减法。

三种读法：
  ① 计数，直接比均值                    （最贴近报告字面）
  ② 计数 + 每个 R 单元自己的 θ_R         （报告里的阈值，按训练集校准）
  ③ 错误修正 w += x（真类）/ w -= x（错类）（1960 年代以后的标准做法）

任务：把方形和十字分开，每个刺激随机平移到视场里的任意位置。

用法：/usr/bin/python3 ai-evolution-05-1957-perceptron.py
"""
import numpy as np

SIZE = 20                      # 光电池阵列边长（Mark I 为 20x20 = 400 个）
N_S = SIZE * SIZE
SHAPES = ["square", "circle", "triangle", "cross"]
A_CLS, B_CLS = 0, 3           # 本篇只区分 square 与 cross


def draw(name, s):
    g = np.zeros((s, s), dtype=bool)
    c = (s - 1) / 2.0
    if name == "square":
        g[0, :] = g[-1, :] = True
        g[:, 0] = g[:, -1] = True
    elif name == "circle":
        for i in range(s):
            for j in range(s):
                rr = np.hypot(i - c, j - c)
                g[i, j] = (c - 1.2) <= rr <= c
    elif name == "triangle":
        g[-1, :] = True
        for i in range(s):
            g[i, int(round(c - (c - i) * c / (s - 1)))] = True
            g[i, int(round(c + (c - i) * c / (s - 1)))] = True
    elif name == "cross":
        m = s // 2
        g[m - 1:m + 2, :] = True
        g[:, m - 1:m + 2] = True
    return g


def make_sample(name, s, rng, lit=None):
    """把形状随机平移到视场里；lit 给定时把亮像素数精确对齐。"""
    g = draw(name, s)
    if lit is not None:
        idx = np.flatnonzero(g)
        if len(idx) >= lit:
            keep = rng.choice(idx, size=lit, replace=False)
        else:
            keep = np.concatenate([idx, rng.choice(np.flatnonzero(~g), lit - len(idx), replace=False)])
        g2 = np.zeros_like(g)
        g2.flat[keep] = True
        g = g2
    r0 = rng.integers(0, SIZE - s + 1)
    c0 = rng.integers(0, SIZE - s + 1)
    field = np.zeros((SIZE, SIZE), dtype=bool)
    field[r0:r0 + s, c0:c0 + s] = g
    return field.reshape(-1).astype(np.float64)


def dataset(n_per_class, rng, matched, sizes=None):
    """只生成两类：方形（标签 0）与十字（标签 1）。"""
    X, y = [], []
    for k, nm in ((A_CLS, SHAPES[A_CLS]), (B_CLS, SHAPES[B_CLS])):
        for _ in range(n_per_class):
            if matched:
                s, lit = 11, 36
            else:
                s, lit = int(sizes[k] + rng.integers(-1, 2)), None
            X.append(make_sample(nm, s, rng, lit))
            y.append(0 if k == A_CLS else 1)
    return np.array(X), np.array(y)


def build_wiring(n_a, patch=(4, 5), rng=None):
    """每个 A 单元看视场里的一个小窗口（4x5=20 个 S 点），权重 ±1。

    局域窗口是关键：从整个视场随机取 20 个点，窗口里凑不出可用的特征。
    Mark I 的 S-A 插线板也是按局域图形插的。
    """
    W = np.zeros((n_a, N_S))
    ph, pw = patch
    for i in range(n_a):
        r = rng.integers(0, SIZE - ph + 1)
        c = rng.integers(0, SIZE - pw + 1)
        sub = np.zeros((SIZE, SIZE))
        sub[r:r + ph, c:c + pw] = rng.choice([-1.0, 1.0], size=patch)
        W[i] = sub.reshape(-1)
    return W


def a_layer(X, W, percentile=80, seed=3):
    """固定阈值 θ：取该单元在训练集上响应的 80 分位（激活率约 15-20%）。

    偶数个 ±1 求和会大量落在 0 上，直接取分位会把阈值卡在 0；
    加 1e-6 的抖动只为了让分位点落在唯一值上，不改变任何可见量级。
    """
    pre = X @ W.T
    jitter = np.random.default_rng(seed).normal(0, 1e-6, pre.shape)
    theta = np.percentile(pre + jitter, percentile, axis=0)
    return pre >= theta, theta


class CountPerceptron:
    """1957 规则：刺激与响应同时出现，被激活的 A 单元对该响应的值 +1。"""

    def __init__(self, n_a, n_r=2):
        self.v = np.zeros((n_a, n_r))
        self.theta_r = None

    def score(self, A):
        act = A.sum(axis=1, keepdims=True)
        return (A @ self.v) / np.maximum(act, 1.0)

    def learn(self, A, y):
        for a_row, r in zip(A, y):
            self.v[a_row, r] += 1.0

    def fit_thresholds(self, A, y):
        sc = self.score(A)
        self.theta_r = np.array([sc[y == r][:, r].mean() for r in range(self.v.shape[1])])

    def predict(self, A):
        return self.score(A).argmax(axis=1)

    def predict_with_threshold(self, A):
        return (self.score(A) - self.theta_r).argmax(axis=1)


class ErrorCorrectionPerceptron:
    """现代规则：错了才动权重，真类 +x，错类 -x。"""

    def __init__(self, n_a, n_r=2, rng=None):
        self.w = np.zeros((n_a, n_r))
        self.rng = rng or np.random.default_rng(7)

    def predict(self, A):
        return (A @ self.w).argmax(axis=1)

    def learn(self, A, y, epochs=30):
        idx = np.arange(len(y))
        for _ in range(epochs):
            self.rng.shuffle(idx)
            for i in idx:
                p = int((A[i] @ self.w).argmax())
                if p != y[i]:
                    self.w[:, y[i]] += A[i]
                    self.w[:, p] -= A[i]


def acc(pred, y):
    return float((pred == y).mean())


def task(n_a, matched, sizes, seed):
    tr_x, tr_y = dataset(80, np.random.default_rng(seed), matched, sizes)
    te_x, te_y = dataset(80, np.random.default_rng(seed + 1), matched, sizes)
    W = build_wiring(n_a, rng=np.random.default_rng(1000 + n_a))
    Atr, theta = a_layer(tr_x, W)
    Ate = (te_x @ W.T) >= theta
    return Atr, tr_y, Ate, te_y, W


def sweep(title, matched, sizes=None, seed=1):
    print(f"\n== {title} ==")
    print(f"训练 160 / 测试 160 个刺激（两类各半，位置随机）；随机基线 0.500")
    print(f"{'A单元':>6} {'①计数':>7} {'②计数+θ_R':>10} {'③错误修正':>9} {'激活率':>7}")
    for n_a in (16, 64, 256, 512):
        Atr, ytr, Ate, yte, _ = task(n_a, matched, sizes, seed)
        cp = CountPerceptron(n_a)
        cp.learn(Atr, ytr)
        cp.fit_thresholds(Atr, ytr)
        ep = ErrorCorrectionPerceptron(n_a)
        ep.learn(Atr, ytr)
        print(f"{n_a:>6} {acc(cp.predict(Ate), yte):>7.3f} "
              f"{acc(cp.predict_with_threshold(Ate), yte):>10.3f} "
              f"{acc(ep.predict(Ate), yte):>9.3f} {Atr.mean():>7.2f}")


def confusion(pred, y, tag):
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())
    print(f"  {tag:<14} 方→方 {tn:>3}  方→十 {fp:>3}  十→方 {fn:>3}  十→十 {tp:>3}")


def curve(matched, sizes, seed=3):
    print("\n== 学习曲线（A 单元 512，B 组尺度不受控；横轴=每类看过的样本数） ==")
    print(f"{'样本/类':>7} {'①计数':>7} {'②计数+θ_R':>10} {'③错误修正':>9}")
    Atr, ytr, Ate, yte, _ = task(512, matched, sizes, seed)
    n = len(ytr) // 2
    for k in (1, 2, 5, 10, 20, 40, 80):
        idx = np.r_[np.arange(k), n + np.arange(k)]
        cp = CountPerceptron(512)
        cp.learn(Atr[idx], ytr[idx])
        cp.fit_thresholds(Atr[idx], ytr[idx])
        ep = ErrorCorrectionPerceptron(512, rng=np.random.default_rng(7))
        ep.learn(Atr[idx], ytr[idx], epochs=30)
        print(f"{k:>7} {acc(cp.predict(Ate), yte):>7.3f} "
              f"{acc(cp.predict_with_threshold(Ate), yte):>10.3f} "
              f"{acc(ep.predict(Ate), yte):>9.3f}")
    return Atr, ytr, Ate, yte


if __name__ == "__main__":
    sweep("A 组：两种形状的亮像素数完全对齐（各 36 个）", matched=True)
    sweep("B 组：尺度不受控（方 13、十字 9，各自 ±1 抖动）",
          matched=False, sizes=[13, 11, 11, 9], seed=3)
    Atr, ytr, Ate, yte = curve(matched=False, sizes=[13, 11, 11, 9])
    cp = CountPerceptron(512)
    cp.learn(Atr, ytr)
    cp.fit_thresholds(Atr, ytr)
    ep = ErrorCorrectionPerceptron(512)
    ep.learn(Atr, ytr)
    print("\nB 组 512 单元的混淆矩阵")
    confusion(cp.predict(Ate), yte, "①计数")
    confusion(cp.predict_with_threshold(Ate), yte, "②计数+θ_R")
    confusion(ep.predict(Ate), yte, "③错误修正")
