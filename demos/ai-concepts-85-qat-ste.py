"""85 · 量化感知训练与直通估计器（STE）

CPU numpy 小实验，不是 GPU 实测。MLP 2-64-64-2 + ReLU + softmax，
合成「双月」3000 训练 / 3000 测试，Adam lr=0.01、1500 步、batch 64，3 个种子。

对照组：
  FP32  全精度训练 + 全精度推理
  PTQ   全精度训练，推理时把权重对称量化到 b bit（训练后量化，不重训）
  QAT   训练时就量化，反向用 STE：梯度直接穿过 round()

另测：潜权重相对量化值的膨胀、有效格距、STE 的「瞎更新」比例、round() 的真实导数。
"""
import numpy as np

# ---------------- 数据 ----------------
def make_moons(n, noise=0.10, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n // 2)
    x0 = np.stack([np.cos(t), np.sin(t)], 1)
    x1 = np.stack([1 - np.cos(t), 0.5 - np.sin(t)], 1)
    X = np.concatenate([x0, x1]) + rng.normal(0, noise, (n, 2))
    y = np.concatenate([np.zeros(n // 2, int), np.ones(n // 2, int)])
    p = rng.permutation(n)
    return X[p], y[p]


# ---------------- 量化器 ----------------
def qmax(bits):
    return 2 ** (bits - 1) - 1              # int2→1, int3→3, int4→7


def quantize(w, bits, per_channel=False):
    """对称均匀量化。per_channel=True 时每个输出神经元独立 scale。返回 (wq, step)。"""
    qm = qmax(bits)
    scale = (np.abs(w).max(axis=0, keepdims=True) if per_channel
             else np.full(1, np.abs(w).max())) / qm
    scale = np.maximum(scale, 1e-12)
    return np.clip(np.round(w / scale), -qm - 1, qm) * scale, scale


# ---------------- 模型 ----------------
class MLP:
    def __init__(self, sizes=(2, 64, 64, 2), seed=0):
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, np.sqrt(2.0 / a), (a, c))
                  for a, c in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(c) for c in sizes[1:]]

    def forward(self, X, bits=None, per_channel=True):
        h, self.cache = X, []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            Wuse = W if bits is None else quantize(W, bits, per_channel)[0]
            z = h @ Wuse + b
            self.cache.append((h, z, Wuse))
            h = np.maximum(z, 0) if i < len(self.W) - 1 else z
        return h

    def loss_grad(self, X, y, bits=None, per_channel=True):
        logits = self.forward(X, bits, per_channel)
        logits = logits - logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        n = len(y)
        loss = -np.log(p[np.arange(n), y] + 1e-12).mean()
        dz = p.copy(); dz[np.arange(n), y] -= 1; dz /= n
        gW = [None] * len(self.W); gb = [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            h, z, Wuse = self.cache[i]
            gW[i] = h.T @ dz                 # ← STE：d round(x)/dx 当成 1
            gb[i] = dz.sum(0)
            if i > 0:
                dz = (dz @ Wuse.T) * (self.cache[i - 1][1] > 0)
        return loss, gW, gb


def accuracy(model, X, y, bits=None, per_channel=True):
    return float((model.forward(X, bits, per_channel).argmax(1) == y).mean())


def train(model, X, y, bits=None, per_channel=True, steps=1500, lr=0.01, flip_log=None):
    mW = [np.zeros_like(w) for w in model.W]; vW = [np.zeros_like(w) for w in model.W]
    mb = [np.zeros_like(b) for b in model.b]; vb = [np.zeros_like(b) for b in model.b]
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(7); n = len(y)
    prev = [quantize(w, bits, per_channel)[0].copy() for w in model.W] if bits else None
    for t in range(1, steps + 1):
        idx = rng.integers(0, n, 64)
        _, gW, gb = model.loss_grad(X[idx], y[idx], bits, per_channel)
        for i in range(len(model.W)):
            for g, mm, vv, tgt in ((gW[i], mW, vW, model.W), (gb[i], mb, vb, model.b)):
                mm[i] = b1 * mm[i] + (1 - b1) * g
                vv[i] = b2 * vv[i] + (1 - b2) * g * g
                tgt[i] -= lr * (mm[i] / (1 - b1 ** t)) / (np.sqrt(vv[i] / (1 - b2 ** t)) + eps)
        if flip_log is not None and t % 50 == 0:
            ch = tot = 0
            for i, W in enumerate(model.W):
                q = quantize(W, bits, per_channel)[0]
                ch += int((np.abs(q - prev[i]) > 1e-12).sum()); tot += W.size
                prev[i] = q
            flip_log.append((t, ch / tot))
    return model


def finite_diff(bits=2, eps=1e-6, n=12):
    """round() 的「真实」导数：把权重挪 1e-6，看量化后的损失动多少。"""
    X, y = make_moons(600, seed=5)
    m = train(MLP(seed=5), X, y, steps=300)
    rng = np.random.default_rng(3); out = []
    base = m.loss_grad(X, y, bits)[0]
    for _ in range(n):
        i = rng.integers(0, len(m.W))
        idx = tuple(rng.integers(0, s) for s in m.W[i].shape)
        m.W[i][idx] += eps
        out.append(abs(m.loss_grad(X, y, bits)[0] - base) / eps)
        m.W[i][idx] -= eps
    return out


# ---------------- 主实验 ----------------
if __name__ == "__main__":
    rows, flips, diag = [], [], {}
    for seed in (1, 2, 3):
        Xtr, ytr = make_moons(3000, seed=seed)
        Xte, yte = make_moons(3000, seed=seed + 1000)
        fp = train(MLP(seed=seed), Xtr, ytr)
        r = dict(seed=seed, fp=accuracy(fp, Xte, yte),
                 ptq4=accuracy(fp, Xte, yte, 4, False), ptq3=accuracy(fp, Xte, yte, 3, False),
                 ptq2pc=accuracy(fp, Xte, yte, 2, True), ptq2=accuracy(fp, Xte, yte, 2, False))
        fl = []
        q2 = train(MLP(seed=seed), Xtr, ytr, bits=2, per_channel=False, flip_log=fl)
        r["qat2"] = accuracy(q2, Xte, yte, 2, False)
        r["qat2pc"] = accuracy(train(MLP(seed=seed), Xtr, ytr, bits=2, per_channel=True), Xte, yte, 2, True)
        r["qat3"] = accuracy(train(MLP(seed=seed), Xtr, ytr, bits=3, per_channel=False), Xte, yte, 3, False)
        rows.append(r)
        if seed == 1:
            flips = fl
            # 有效格距：量化器的 step 大小（越小越精细）
            for tag, model, bits, pc in (("fp@2bit", fp, 2, False), ("qat@2bit", q2, 2, False)):
                step = np.mean([quantize(W, bits, pc)[1].mean() for W in model.W])
                lat = np.mean([np.abs(W).mean() for W in model.W])
                qua = np.mean([np.abs(quantize(W, bits, pc)[0]).mean() for W in model.W])
                diag[tag] = (step, lat, qua)

    print("双月测试集 3000 条，3 个种子（mean / std）：")
    for k in ("fp", "ptq4", "ptq3", "ptq2pc", "ptq2", "qat3", "qat2pc", "qat2"):
        v = [r[k] for r in rows]
        print(f"  {k:6s} {np.mean(v):.4f} / {np.std(v):.4f}   {[round(x, 4) for x in v]}")

    s_fp, lat_fp, q_fp = diag["fp@2bit"]
    s_q, lat_q, q_q = diag["qat@2bit"]
    print("\nint2 量化器（per-tensor）:")
    print(f"  有效格距 step: 全精度权重 {s_fp:.4f}   QAT 潜权重 {s_q:.4f}   粗 {s_q/s_fp:.2f} 倍")
    print(f"  潜权重 |w| / 量化后 |w|: 全精度 {lat_fp/q_fp:.3f}   QAT {lat_q/q_q:.3f}")

    print("\nSTE 盲区：每 50 步里量化权重真正翻格的比例")
    for t, f in flips[::6]:
        print(f"  step {t:5d}  {f:.4f}")
    print(f"  全程平均 {np.mean([f for _, f in flips]):.4f}")

    fd = finite_diff()
    print(f"\nround() 的真实导数（|ΔL|/eps，12 次随机扰动）：中位数 {np.median(fd):.2e}，"
          f"非零 {sum(1 for d in fd if d > 1e-12)}/12")
