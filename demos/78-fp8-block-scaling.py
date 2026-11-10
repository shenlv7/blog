"""78 | FP8 的 3 位尾数 vs INT8 的 7 位：为什么梯度张量必须给浮点

纯 numpy + CPU，无 GPU、无真实 LLM。
三件事：
 1) 实现 e4m3（4 位指数 / 3 位尾数，max 448，最小正规数 2^-6，次正规步长 2^-9）
 2) 在一个跨 5 个数量级的重尾张量上比 INT8 per-tensor / FP8 per-tensor / FP8 128 分块
 3) 用三种精度真训一个两层 tanh 网络 600 步，看 loss 差多少
"""
import numpy as np

rng = np.random.default_rng(0)


# ---------- 量化器 ----------
def fp8_e4m3(x):
    """对称 e4m3 的 round-to-nearest。"""
    a = np.abs(x)
    e = np.floor(np.log2(np.maximum(a, 2.0 ** -6)))
    step = 2.0 ** (e - 3)                       # 3 位尾数 → 相对步长 2^-3
    q = np.round(a / step) * step
    q = np.where(a < 2.0 ** -6, np.round(a / 2.0 ** -9) * 2.0 ** -9, q)
    return np.sign(x) * np.clip(q, 0.0, 448.0)


def int8_sym(x):
    s = np.abs(x).max() / 127.0
    return np.clip(np.round(x / s), -127, 127) * s


def q_fp8_block(x, block=128, axis=1):
    """沿 axis 每 block 个元素一个 scale（等价 1x128 tile）。"""
    if axis == 1:
        n = x.shape[1]
        pad = (-n) % block
        xp = np.pad(x, ((0, 0), (0, pad)))
        b = xp.reshape(x.shape[0], -1, block)
        s = np.abs(b).max(axis=2, keepdims=True)
        s = np.where(s == 0, 1.0, s) / 448.0
        out = (fp8_e4m3(b / s) * s).reshape(x.shape[0], -1)[:, :n]
        return out
    return q_fp8_block(x.T, block, 1).T


def sqnr(x, q):
    return 10 * np.log10(np.sum(x ** 2) / max(np.sum((x - q) ** 2), 1e-300))


# ---------- 1. 重尾张量上的三种量化 ----------
# 模拟梯度/激活：lognormal 幅度，跨 5 个数量级
t = rng.lognormal(mean=-1.0, sigma=1.6, size=(512, 512))
t[rng.random(t.shape) < 0.01] *= 60          # 1% 的离群元素
print(f"张量：max={t.max():.3g}  min={t.min():.3g}  max/min={t.max()/t.min():.3g}")
print(f"10 分位 / 90 分位 = {np.percentile(t,10):.3g} / {np.percentile(t,90):.3g}")

for name, q in [("INT8 per-tensor", int8_sym(t)),
                ("FP8  per-tensor", fp8_e4m3(t * (448.0 / t.max())) * (t.max() / 448.0)),
                ("FP8  128-block", q_fp8_block(t))]:
    dead = float(np.mean(np.abs(q) < 1e-12))
    clip = float(np.mean(np.abs(q) >= 447.0))
    print(f"{name:16} SQNR {sqnr(t, q):6.2f} dB   归零 {dead*100:5.2f}%   顶格 {clip*100:5.2f}%")

# 另测一个"范围友好"的高斯张量（权重典型形态）
g = rng.normal(0, 0.02, (512, 512))
print("\n对照：高斯权重张量 (σ=0.02)")
for name, q in [("INT8 per-tensor", int8_sym(g)),
                ("FP8  per-tensor", fp8_e4m3(g * (448.0 / g.max())) * (g.max() / 448.0)),
                ("FP8  128-block", q_fp8_block(g))]:
    print(f"{name:16} SQNR {sqnr(g, q):6.2f} dB")

# ---------- 2. 量化训练 ----------
D, W, N = 16, 64, 4096
Xall = rng.normal(size=(N, D))
Yall = (np.tanh(Xall @ rng.normal(0, 0.7, (D, W))) @ rng.normal(0, 1 / np.sqrt(W), (W, 1)))
Yall /= Yall.std()


def init(seed=1):
    r = np.random.default_rng(seed)
    return [r.normal(0, 1 / np.sqrt(D), (D, W)), np.zeros(W),
            r.normal(0, 1 / np.sqrt(W), (W, 1)), np.zeros(1)]


def quant_mm(A, B, mode):
    if mode == "fp32":
        return A @ B
    if mode == "int8":
        return int8_sym(A) @ int8_sym(B)
    if mode == "fp8_tensor":
        sa, sb = 448.0 / np.abs(A).max(), 448.0 / np.abs(B).max()
        return (fp8_e4m3(A * sa) / sa) @ (fp8_e4m3(B * sb) / sb)
    if mode == "fp8_block":
        return q_fp8_block(A, axis=1) @ q_fp8_block(B, axis=0)


def train(mode, lr=3e-3, steps=600, bs=128):
    p = init()
    m = [np.zeros_like(x) for x in p]
    v = [np.zeros_like(x) for x in p]
    r = np.random.default_rng(42)
    hist = []
    for step in range(1, steps + 1):
        idx = r.integers(0, N, bs)
        x, y = Xall[idx], Yall[idx]
        h = np.tanh(quant_mm(x, p[0], mode) + p[1])
        out = quant_mm(h, p[2], mode) + p[3]
        g = 2 * (out - y) / bs
        gh = quant_mm(g, p[2].T, mode) if mode != "int8" else int8_sym(g) @ int8_sym(p[2].T)
        dz = gh * (1 - h ** 2)
        grads = [quant_mm(x.T, dz, mode) if mode != "int8" else int8_sym(x.T) @ int8_sym(dz),
                 dz.sum(0),
                 quant_mm(h.T, g, mode) if mode != "int8" else int8_sym(h.T) @ int8_sym(g),
                 g.sum(0)]
        for i in range(4):
            m[i] = 0.9 * m[i] + 0.1 * grads[i]
            v[i] = 0.999 * v[i] + 0.001 * grads[i] ** 2
            p[i] -= lr * (m[i] / (1 - 0.9 ** step)) / (np.sqrt(v[i] / (1 - 0.999 ** step)) + 1e-8)
        if step % 150 == 0:
            hh = np.tanh(Xall @ p[0] + p[1])
            hist.append(float(((hh @ p[2] + p[3] - Yall) ** 2).mean()))
    return hist, grads, p


print("\n量化训练（600 步，Adam 3e-3，fp32 梯度更新，只有 matmul 走低精度）")
ref = None
for mode in ("fp32", "int8", "fp8_tensor", "fp8_block"):
    h, gr, p = train(mode)
    if ref is None:
        ref = p
    dyn = max(np.abs(g).max() / max(np.abs(g).min(), 1e-30) for g in gr)
    drift = max(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))
                for a, b in ((p[0], ref[0]), (p[2], ref[2])))
    print(f"{mode:11} loss@150/300/450/600 = " + " / ".join(f"{l:.4f}" for l in h)
          + f"   参数偏离 fp32 {drift*100:5.2f}%   梯度 max/min = {dyn:.2e}")
