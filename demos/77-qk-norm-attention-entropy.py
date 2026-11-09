#!/usr/bin/env python3
"""77 · 注意力熵塌缩与 QK-norm —— 纯 numpy 模拟（非 GPU 实测）

模拟的只有一件事：logits = q·k/√d 的尺度如何决定 softmax 的熵、梯度，
以及 bf16 在该尺度下的舍入颗粒。q、k 的范数由外部参数 r 给出，
对应真实模型里 q = h W_q 时 ‖W_q‖ 无约束的漂移。

跑法：python3 77-qk-norm-attention-entropy.py
"""
import numpy as np

RNG = np.random.default_rng(7)
D_HEAD, N_KEYS, N_QUERIES = 64, 128, 256
SCALES = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0, 16.0]


def unit_rows(n, d, rng):
    """n 个 d 维向量，方向随机、每行 RMS = 1（模拟已被 LayerNorm 归一过的隐状态）。"""
    X = rng.normal(size=(n, d))
    return X / np.sqrt((X ** 2).mean(axis=1, keepdims=True))


def qk_logits(r, qk_norm=False, seed=0):
    """r = q、k 每行 RMS。logit std ≈ r²（推导见正文）。"""
    rng = np.random.default_rng(seed)
    q = unit_rows(N_QUERIES, D_HEAD, rng)
    k = unit_rows(N_KEYS, D_HEAD, rng)
    if qk_norm:                      # q、k 各自 RMSNorm：r 被钉回 1
        q, k = q / np.sqrt((q ** 2).mean(1, keepdims=True)) * 1.0, \
               k / np.sqrt((k ** 2).mean(1, keepdims=True)) * 1.0
        r = 1.0
    return (q * r) @ (k * r).T / np.sqrt(D_HEAD)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def to_bf16(x):
    """按 IEEE bfloat16 舍入（8 位显式尾数），回到 float64 参与后续计算。"""
    u = np.asarray(x, dtype=np.float32).view(np.uint32).copy()
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u & 0xFFFF0000).view(np.float32).astype(np.float64)


def stats(L, qk_norm=False, seed=0):
    p = softmax(L)
    ent = -np.sum(p * np.log2(np.maximum(p, 1e-30)), axis=1)
    y = np.zeros_like(p)
    y[np.arange(len(p)), L.argmax(1)] = 1.0
    grad = np.linalg.norm(p - y, axis=1)
    return dict(std=L.std(), mx=np.abs(L).max(), ent=ent.mean(),
                eff=(2.0 ** ent).mean(), grad=grad.mean(),
                peak=p.max(1).mean())


def bf16_weight_gap(L):
    """同一组 logits 用 bf16 存一遍，softmax 权重的最大相对偏差。"""
    p32, pbf = softmax(L), softmax(to_bf16(L))
    return float(np.abs(p32 - pbf).max() / max(p32.max(), 1e-12))


hdr = f"{'r':>5} {'logit std':>10} {'max|logit|':>11} {'熵 bit':>8} " \
      f"{'有效键':>7} {'峰值权重':>9} {'梯度范数':>9} {'bf16 权重偏差':>11}"
line = "-" * len(hdr)

print(f"设定：d_head={D_HEAD}，键数={N_KEYS}（熵上限 {np.log2(N_KEYS):.2f} bit），"
      f"查询数={N_QUERIES}\n")
print("一、不给 q/k 加约束（真实模型里 ‖W_q‖、‖W_k‖ 谁也没管）")
print(hdr)
print(line)
for r in SCALES:
    L = qk_logits(r)
    s = stats(L)
    print(f"{r:5.1f} {s['std']:10.1f} {s['mx']:11.1f} {s['ent']:8.3f} "
          f"{s['eff']:7.1f} {s['peak']:9.3f} {s['grad']:9.4f} {bf16_weight_gap(L):11.1%}")

print("\n二、QK-norm：q、k 出投影后各做一次 RMSNorm，尺度只由一个增益 γ 控制")
print(hdr.replace("    r", "    γ"))
print(line)
for g in [0.5, 0.8, 1.0, 1.4, 2.0, 2.8, 4.0]:
    rng = np.random.default_rng(0)
    q = unit_rows(N_QUERIES, D_HEAD, rng) * g      # RMSNorm 的逐维增益
    k = unit_rows(N_KEYS, D_HEAD, rng) * g
    L = (q @ k.T) / np.sqrt(D_HEAD)
    s = stats(L)
    print(f"{g:5.1f} {s['std']:10.1f} {s['mx']:11.1f} {s['ent']:8.3f} "
          f"{s['eff']:7.1f} {s['peak']:9.3f} {s['grad']:9.4f} {bf16_weight_gap(L):11.1%}")

print("\n三、熵塌到 1 bit 以内的临界尺度（二分法，numpy 版）")
lo, hi = 1.0, 8.0
for _ in range(40):
    mid = (lo + hi) / 2
    if stats(qk_logits(mid))['ent'] > 1.0:
        lo = mid
    else:
        hi = mid
Lc = qk_logits(lo)
print(f"r ≈ {lo:.3f}（logit std ≈ {Lc.std():.2f}）时平均熵跌破 1 bit；"
      f"此时有效键 {stats(Lc)['eff']:.1f}/{N_KEYS}，梯度范数 {stats(Lc)['grad']:.4f}")

print("\n四、bf16 的颗粒：logit 量级 → 权重级别的相对误差")
for r in [1.0, 4.0, 8.0, 16.0]:
    L = qk_logits(r)
    ulp = np.abs(L).max() / 256.0
    p32, pbf = softmax(L), softmax(to_bf16(L))
    rel = np.abs(p32 - pbf).max() / max(p32.max(), 1e-12)
    print(f"r={r:4.1f}  max|logit|={np.abs(L).max():10.1f}  bf16 ulp≈{ulp:8.2f}  "
          f"权重最大相对偏差 {rel:.1%}")
