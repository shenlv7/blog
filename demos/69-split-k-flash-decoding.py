#!/usr/bin/env python3
"""
69 · split-K / flash-decoding：把长上下文的 decode 注意力按 KV 切块并行
本机 numpy 实跑（非 GPU 实测）：数值部分是真算的，延迟部分是成本模型。
"""
import numpy as np

N, H, D = 32768, 8, 64
SCALE = 1.0 / np.sqrt(D)
rng = np.random.default_rng(20261101)

K = rng.standard_normal((H, N, D), dtype=np.float32)
V = rng.standard_normal((H, N, D), dtype=np.float32)
q = rng.standard_normal((H, D), dtype=np.float32)

# 针尖场景：在第 40% 位置放一个和 q 强对齐的 key，softmax 质量几乎全压在那里
NEEDLE = int(N * 0.4)
K2, V2 = K.copy(), V.copy()
K2[:, NEEDLE, :] = q * 12.0


def logits_of(Kx, qx):
    return np.einsum("hnd,hd->hn", Kx, qx, optimize=True) * SCALE


def ref_attention(z, Vx):
    """两遍流式精确参考：先全局 max，再累加。"""
    m = z.max(axis=1, keepdims=True)
    p = np.exp(z - m)
    l = p.sum(axis=1)
    acc = np.einsum("hn,hnd->hd", p, Vx, optimize=True)
    return acc / l[:, None], m.ravel(), l


def split_k(z, Vx, splits, merge="lse"):
    """把 KV 按 token 切成 splits 段，每段独立 softmax，再合并。"""
    Hh, Nn = z.shape
    edges = np.linspace(0, Nn, splits + 1).astype(int)
    ms, ls, accs = [], [], []
    for s in range(splits):
        a, b = edges[s], edges[s + 1]
        zb = z[:, a:b]
        m_i = zb.max(axis=1)
        p = np.exp(zb - m_i[:, None])
        l_i = p.sum(axis=1)
        acc_i = np.einsum("hn,hnd->hd", p, Vx[:, a:b, :], optimize=True)
        ms.append(m_i); ls.append(l_i); accs.append(acc_i / l_i[:, None])
    ms = np.stack(ms, 1); ls = np.stack(ls, 1); accs = np.stack(accs, 1)  # (H,S),(H,S),(H,S,D)

    if merge == "lse":            # 正确做法：按局部 max 重新加权
        m = ms.max(axis=1, keepdims=True)
        w = np.exp(ms - m)
        l = (ls * w).sum(axis=1)
        acc = (accs * (ls * w)[:, :, None]).sum(axis=1)
        return acc / l[:, None]
    if merge == "avg":            # 错法一：把各段的归一化结果直接平均
        return accs.mean(axis=1)
    if merge == "weighted":       # 错法二：按段长度/局部分母加权，忽略 max 漂移
        return (accs * ls[:, :, None]).sum(axis=1) / ls.sum(axis=1)[:, None]
    raise ValueError(merge)


def rel_err(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


print("=" * 78)
print(f"场景: N={N} tokens, {H} 个 KV 头, d={D}, float32; 单次 decode 一个 token")
print("=" * 78)

for name, Kx, Vx in [("随机键（softmax 平坦）", K, V), ("针尖键（softmax 近 one-hot）", K2, V2)]:
    z = logits_of(Kx, q)
    ref, m, l = ref_attention(z, Vx)
    print(f"\n--- {name} ---")
    print(f"全局 max 与分母: max logit={m.max():.3f}, 有效参与 token 数 l={l.mean():.1f}"
          f" / {N}（{l.mean() / N * 100:.2f}%）")
    print(f"{'splits':>7} {'合并方式':<10} {'最大相对误差':>14} {'是否可用':>8}")
    for S in (1, 2, 4, 8, 16, 32, 64):
        out = split_k(z, Vx, S, "lse")
        print(f"{S:>7} {'lse':<10} {rel_err(out, ref):>14.3e} {'ok' if rel_err(out, ref) < 1e-5 else 'BAD':>8}")
        if S == 32:
            for mk in ("avg", "weighted"):
                e = rel_err(split_k(z, Vx, S, mk), ref)
                print(f"{S:>7} {mk:<10} {e:>14.3e} {'ok' if e < 1e-5 else 'BAD':>8}")

# ── 精度：合并本身 vs 存储 dtype ───────────────────────────────────────────
def cos(a, b):
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b)))


print("\n--- 精度：合并本身 vs 存储 dtype（随机键，输出幅度小，看绝对误差与余弦）---")
z32 = logits_of(K, q)
ref32, _, l32 = ref_attention(z32, V)
ref64, _, _ = ref_attention(logits_of(K.astype(np.float64), q.astype(np.float64)),
                            V.astype(np.float64))
out64 = split_k(logits_of(K.astype(np.float64), q.astype(np.float64)),
                V.astype(np.float64), 32, "lse")
print(f"float64 参考 vs 32 路合并        : max|Δ|={np.abs(out64 - ref64).max():.3e}  cos={cos(out64.ravel(), ref64.ravel()):.12f}")

K16 = K.astype(np.float16).astype(np.float32)   # 模拟 KV cache 存 fp16
V16 = V.astype(np.float16).astype(np.float32)
ref16, _, _ = ref_attention(logits_of(K16, q), V16)
print(f"KV 落成 fp16 本身 vs float32 参考: max|Δ|={np.abs(ref16 - ref32).max():.3e}  cos={cos(ref16.ravel(), ref32.ravel()):.10f}")
out16 = split_k(logits_of(K16, q), V16, 32, "lse")
print(f"fp16 存储 + 32 路合并 vs 其参考  : max|Δ|={np.abs(out16 - ref16).max():.3e}")
print(f"（参考输出自身量级 |ref|max={np.abs(ref32).max():.3e}，l={l32.mean():.0f} 个 token 参与，"
      f"平均后互相抵消，所以相对误差会被放大）")

# ── 延迟成本模型（假设值，非实测）────────────────────────────────────────
SMS, BW_TOTAL, LAUNCH_US = 132, 3.35e12, 4.0
BW_SM = BW_TOTAL / SMS


def cost(N_ctx, h_kv, d, splits, bytes_per_val=2):
    kv_bytes = N_ctx * h_kv * 2 * d * bytes_per_val       # K 和 V
    blocks = h_kv * splits
    bw = min(blocks, SMS) * BW_SM
    t_body = kv_bytes / bw
    t_merge = LAUNCH_US * 1e-6 if splits > 1 else 0.0
    return kv_bytes, blocks, t_body + t_merge, t_body


print("\n--- 延迟成本模型（H100 假设：132 SM / 3.35 TB/s / 每 SM 25.4 GB/s / kernel 固定 4 µs）---")
kv, _, _, _ = cost(131072, 8, 128, 1)
print(f"128K 上下文、8 KV 头、d=128、fp16：单层 KV 读取量 = {kv / 2**20:.0f} MiB/步")
print(f"{'splits':>7} {'块数':>6} {'有效SM':>7} {'聚合带宽':>10} {'单层延迟':>10} {'相对 1 路':>9}")
base = None
for S in (1, 2, 4, 8, 16, 32, 64):
    kv, blocks, t, tb = cost(131072, 8, 128, S)
    if base is None:
        base = t
    print(f"{S:>7} {blocks:>6} {min(blocks, SMS):>7} "
          f"{min(blocks, SMS) * BW_SM / 1e9:>8.0f}GB/s {t * 1e6:>9.0f}µs {base / t:>8.2f}x")

print("\n--- 什么时候 split-K 反而更慢（同一成本模型）---")
print(f"{'上下文':>8} {'1 路':>10} {'16 路':>10} {'谁快':>8}")
for n in (1024, 2048, 4096, 8192, 32768, 131072):
    _, _, t1, _ = cost(n, 8, 128, 1)
    _, _, t16, _ = cost(n, 8, 128, 16)
    print(f"{n:>8} {t1 * 1e6:>9.1f}µs {t16 * 1e6:>9.1f}µs {'16 路' if t16 < t1 else '1 路':>8}")

# 交叉点：kv/BW8 - kv/BW132 = 4µs
kv_per_tok = 8 * 2 * 128 * 2
n_star = LAUNCH_US * 1e-6 / (1 / (8 * BW_SM) - 1 / (SMS * BW_SM)) / kv_per_tok
print(f"\n单层交叉点 ≈ {n_star:.0f} tokens（低于它，合并 kernel 的固定开销吃掉并行收益）")
print(f"聚合带宽饱和点 = 132 SM / 8 KV 头 ≈ {np.ceil(SMS / 8):.0f} 路，再多只是空转")

# ── 什么时候不需要 split-K：batch 已经填满 SM ────────────────────────────
print("\n--- batch 大小的作用（8 KV 头，看块数够不够填满 132 个 SM）---")
print(f"{'batch':>6} {'1 路块数':>9} {'1 路有效SM':>11} {'最短够用路数':>13}")
for b in (1, 2, 4, 8, 16, 32, 64):
    blocks1 = b * 8
    need = int(np.ceil(SMS / blocks1))
    print(f"{b:>6} {blocks1:>9} {min(blocks1, SMS):>11} {need:>13}")

# 局部结果占用的 scratch：H * splits * d 个 float（acc）+ H * splits 个 float（m、l）
for S in (16, 32, 64):
    acc_b = 8 * S * 128 * 4
    lse_b = 8 * S * 4 * 2
    print(f"splits={S:>2}: 局部结果 {acc_b / 1024:.0f} KiB + lse {lse_b / 1024:.1f} KiB")

# ── kernel 层加速 ≠ 端到端加速：同一层还要读权重 ─────────────────────────
print("\n--- 端到端：一层 decode 的读取量拆解（70B 级、单卡、80 层、权重 fp16）---")
w_bytes = (70e9 / 80) * 2
t_w = w_bytes / BW_TOTAL
print(f"权重读取 {w_bytes / 2**20:.0f} MiB/层 → {t_w * 1e6:.0f} µs（跑满带宽）")
print(f"{'上下文':>8} {'KV MiB':>8} {'1 路/层':>10} {'16 路/层':>10} {'kernel 加速':>11} {'端到端加速':>10}")
for n in (4096, 32768, 131072):
    kvb, _, t1, _ = cost(n, 8, 128, 1)
    _, _, t16, _ = cost(n, 8, 128, 16)
    print(f"{n:>8} {kvb / 2**20:>8.0f} {t1 * 1e6:>9.0f}µs {t16 * 1e6:>9.0f}µs "
          f"{t1 / t16:>10.2f}x {(t1 + t_w) / (t16 + t_w):>9.2f}x")


