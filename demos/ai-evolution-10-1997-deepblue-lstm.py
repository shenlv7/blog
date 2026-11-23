"""ai-evolution-10 · 1997
实验 1：误差信号穿过 T 步之后还剩多少（tanh 循环 vs 常数误差传送带 CEC）
实验 2：延迟回忆任务上，vanilla RNN 与 LSTM 的真实学习曲线（手工 BPTT，纯 numpy）
跑法：python3 ai-evolution-10-1997-deepblue-lstm.py
"""
import numpy as np

rng = np.random.default_rng(1997)


# ---------------------------------------------------------------- 实验 1
def rnn_jacobian_norm(T, rho, n=100, hidden=32):
    """‖∂h_T/∂h_0‖ for h_t = tanh(W h_{t-1})，W 的谱半径固定为 rho"""
    out = []
    for _ in range(n):
        W = rng.normal(size=(hidden, hidden)) / np.sqrt(hidden)
        s = np.max(np.abs(np.linalg.eigvals(W)))
        W *= rho / s
        J = np.eye(hidden)
        h = rng.normal(size=hidden) * 0.5
        for _ in range(T):
            h = np.tanh(W @ h)
            J = np.diag(1.0 - h ** 2) @ W @ J
        out.append(np.linalg.norm(J, 2))
    return float(np.median(out))


def cec_norm(T):
    """c_t = c_{t-1}（输入门关闭、遗忘门=1）：∂c_T/∂c_0 精确等于 1"""
    J = 1.0
    for _ in range(T):
        J *= 1.0
    return J


# ---------------------------------------------------------------- 实验 2
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def make_task(n, T, rng):
    bits = rng.integers(0, 2, size=n).astype(float)
    X = np.zeros((n, T, 1))
    X[:, 0, 0] = bits                 # 第 1 步给出要记住的比特，之后全是 0
    return X, bits


class VanillaRNN:
    def __init__(self, H, seed=0):
        r = np.random.default_rng(seed)
        self.H = H
        self.p = {
            "Wh": r.normal(size=(H, H)) * 0.1,
            "Wx": r.normal(size=(H, 1)) * 0.1,
            "bh": np.zeros(H),
            "Wy": r.normal(size=(1, H)) * 0.1,
            "by": np.zeros(1),
        }

    def forward(self, X):
        p = self.p
        n, T = X.shape[0], X.shape[1]
        H = self.H
        hs = np.zeros((n, T + 1, H))
        for t in range(T):
            hs[:, t + 1] = np.tanh(hs[:, t] @ p["Wh"].T + X[:, t] @ p["Wx"].T + p["bh"])
        return hs[:, T] @ p["Wy"].T + p["by"], hs

    def loss_grad(self, X, y):
        p = self.p
        n, T = X.shape[0], X.shape[1]
        H = self.H
        logit, hs = self.forward(X)
        logit = logit.ravel()
        pr = sigmoid(logit)
        loss = float(np.mean(np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))))
        dlogit = (pr - y) / n
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wy"] = np.einsum("n,nH->H", dlogit, hs[:, T])[None, :]
        g["by"] = np.array([dlogit.sum()])
        dh = np.outer(dlogit, p["Wy"][0])
        for t in range(T, 0, -1):
            da = dh * (1 - hs[:, t] ** 2)
            g["Wh"] += da.T @ hs[:, t - 1]
            g["Wx"] += da.T @ X[:, t - 1]
            g["bh"] += da.sum(0)
            dh = da @ p["Wh"]
        return loss, g

    def acc(self, X, y):
        logit, _ = self.forward(X)
        return float(np.mean((logit.ravel() > 0) == (y > 0.5)))


class LSTM1997:
    """原版 LSTM：输入门 + 输出门 + 常数误差传送带，没有遗忘门（2000 年才加）"""

    def __init__(self, H, seed=0):
        r = np.random.default_rng(seed)
        self.H = H
        f = lambda *s: r.normal(size=s) * 0.1
        self.p = {
            "Wx": f(4, H, 1), "Wh": f(4, H, H), "b": np.zeros((4, H)),
            "Wy": f(1, H), "by": np.zeros(1),
        }
        self.p["b"][0, :] = 1.0          # 输入门偏置前移：训练一开始就记住

    def forward(self, X):
        p = self.p
        n, T = X.shape[0], X.shape[1]
        H = self.H
        c = np.zeros((n, H))
        h = np.zeros((n, H))
        cache = []
        for t in range(T):
            z = X[:, t] @ p["Wx"].reshape(4 * H, 1).T + h @ p["Wh"].reshape(4 * H, H).T + p["b"].reshape(-1)
            i = sigmoid(z[:, :H])
            g = np.tanh(z[:, H:2 * H])
            o = sigmoid(z[:, 3 * H:])
            c_new = c + i * g
            h_new = o * np.tanh(c_new)
            cache.append((h.copy(), c.copy(), i, g, o, c_new, h_new))
            h, c = h_new, c_new
        logit = h @ p["Wy"].T + p["by"]
        return logit, cache, (h, c)

    def loss_grad(self, X, y):
        p = self.p
        n, T = X.shape[0], X.shape[1]
        H = self.H
        logit, cache, _ = self.forward(X)
        logit = logit.ravel()
        pr = sigmoid(logit)
        loss = float(np.mean(np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))))
        dlogit = (pr - y) / n
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wy"] = np.einsum("n,nH->H", dlogit, cache[-1][6])[None, :]
        g["by"] = np.array([dlogit.sum()])
        dh = np.outer(dlogit, p["Wy"][0])
        dc = np.zeros((n, H))
        for t in range(T - 1, -1, -1):
            h_prev, c_prev, i, gg, o, c_new, _ = cache[t]
            tc = np.tanh(c_new)
            dc = dc + dh * o * (1 - tc ** 2)          # 传送带：这一步的 dc 直接加到上一步
            do = dh * tc * o * (1 - o)
            di = dc * gg * i * (1 - i)
            dg = dc * i * (1 - gg ** 2)
            dz = np.concatenate([di, dg, np.zeros_like(di), do], axis=1)
            x_t = X[:, t]
            g["Wx"] += np.stack([dz[:, k * H:(k + 1) * H].T @ x_t for k in range(4)])
            g["Wh"] += np.stack([dz[:, k * H:(k + 1) * H].T @ h_prev for k in range(4)])
            g["b"] += np.stack([dz[:, k * H:(k + 1) * H].sum(0) for k in range(4)])
            dh = dz @ p["Wh"].reshape(4 * H, H)
            # 无遗忘门：∂c_t/∂c_{t-1} = 1，dc 原样传回上一步（常数误差传送带）
        return loss, g

    def acc(self, X, y):
        logit, _, _ = self.forward(X)
        return float(np.mean((logit.ravel() > 0) == (y > 0.5)))


def gradcheck(model, X, y, idx=0):
    """数值梯度核对，确认手工 BPTT 没写反"""
    loss, g = model.loss_grad(X, y)
    errs = []
    for name, arr in model.p.items():
        flat = arr.ravel()
        for j in range(min(3, flat.size)):
            old = flat[j]
            flat[j] = old + 1e-5
            lp, _ = model.loss_grad(X, y)
            flat[j] = old - 1e-5
            lm, _ = model.loss_grad(X, y)
            flat[j] = old
            num = (lp - lm) / 2e-5
            ana = g[name].ravel()[j]
            errs.append(abs(num - ana) / max(1e-6, abs(num) + abs(ana)))
    return max(errs)


def train(model, Xtr, ytr, Xte, yte, epochs=150, lr=2e-2, bs=64, seed=1):
    r = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in model.p.items()}
    v = {k: np.zeros_like(v) for k, v in model.p.items()}
    hist = {}
    for ep in range(1, epochs + 1):
        order = r.permutation(len(Xtr))
        for s in range(0, len(Xtr), bs):
            idx = order[s:s + bs]
            _, g = model.loss_grad(Xtr[idx], ytr[idx])
            for k in model.p:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                mh = m[k] / (1 - 0.9 ** ep)
                vh = v[k] / (1 - 0.999 ** ep)
                model.p[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
        if ep in (1, 10, 25, 50, 100, 150):
            hist[ep] = (model.loss_grad(Xtr, ytr)[0], model.acc(Xte, yte))
    return hist


if __name__ == "__main__":
    print("=== 实验 1：误差信号穿过 T 步后的雅可比范数 ‖∂h_T/∂h_0‖ ===")
    print(f"{'T':>4} {'rho=0.9':>12} {'rho=1.35':>12} {'常数误差传送带':>14}")
    for T in (1, 5, 10, 20, 50, 100):
        print(f"{T:>4} {rnn_jacobian_norm(T, 0.9):>12.3e} "
              f"{rnn_jacobian_norm(T, 1.35):>12.3e} {cec_norm(T):>14.1f}")

    print("\n=== 实验 2：延迟回忆（第 1 步的比特，到第 T 步才输出）===")
    T, H, N = 50, 24, 800
    Xtr, ytr = make_task(N, T, np.random.default_rng(7))
    Xte, yte = make_task(400, T, np.random.default_rng(8))
    print(f"序列长度 T={T}，隐藏单元 H={H}，训练样本 {N}，测试 400，随机猜的基线 0.500")

    for name, cls in (("vanilla RNN", VanillaRNN), ("LSTM(1997)", LSTM1997)):
        mdl = cls(H, seed=3)
        gc = gradcheck(mdl, Xtr[:32], ytr[:32])
        hist = train(mdl, Xtr, ytr, Xte, yte)
        print(f"\n[{name}] 梯度核对最大相对误差 {gc:.2e}")
        for ep, (l, a) in hist.items():
            print(f"  epoch {ep:>3}  train loss {l:.4f}  test acc {a:.3f}")
