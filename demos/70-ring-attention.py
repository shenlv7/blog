"""70 · Ring Attention：把序列切成 P 段，每段各算局部 softmax，靠环形传递 KV 合并成全局结果。
本脚本做三件事：
  1) 用 float64 验证分块 + 环形传递的注意力与朴素全注意力数值一致（前缀和式在线 softmax）
  2) 用 float32 报一个真实可复现的误差量级
  3) 按 H100 / NVLink 参数算通信-计算比、单卡显存、以及因果掩码下的负载不均衡
所有延迟数字都是成本模型推算，不是 GPU 实测。
"""
import numpy as np

rng = np.random.default_rng(0)
D_HEAD = 64
TOK = 512          # 验证用序列长度
RANKS = 4          # 环形设备数（每台设备负责一个 query 块）


def full_attention(Q, K, V):
    z = Q @ K.T / np.sqrt(Q.shape[1])
    S = Q.shape[0]
    mask = np.tril(np.ones((S, S), dtype=bool))
    z = np.where(mask, z, -np.inf)
    z = z - z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    return p @ V


def ring_attention(Q, K, V, ranks, causal=True):
    """每台设备持有自己的 Q 块与 K/V 块；KV 每步沿环传递一格。
    返回 (输出, 每台设备实际执行的步数) —— 步数用来量化因果掩码下的负载不均。"""
    S, d = Q.shape
    B = S // ranks
    q_blocks = [Q[i * B:(i + 1) * B] for i in range(ranks)]
    k_blocks = [K[i * B:(i + 1) * B] for i in range(ranks)]
    v_blocks = [V[i * B:(i + 1) * B] for i in range(ranks)]

    out = np.zeros_like(Q)
    steps_done = []
    for r in range(ranks):                      # 模拟 rank r 上的完整循环
        q = q_blocks[r]
        m = np.full((B, 1), -np.inf)            # 运行最大值
        l = np.zeros((B, 1))                    # 运行分母
        acc = np.zeros((B, d))                  # 运行分子
        steps = 0
        for t in range(ranks):                  # 本地已持有 rank r 的 KV，第 t 步收到 (r-t)%ranks
            src = (r - t) % ranks
            # 因果剪枝：整块都在本 query 块之后就直接跳过，KV 继续往下一格传
            if causal and src > r:
                continue
            k, v = k_blocks[src], v_blocks[src]
            z = q @ k.T / np.sqrt(d)
            if causal and src == r:
                local = np.tril(np.ones((B, B), dtype=bool))
                z = np.where(local, z, -np.inf)
            m_new = np.maximum(m, z.max(axis=-1, keepdims=True))
            alpha = np.exp(m - m_new)           # 旧累加量的重标系数
            p = np.exp(z - m_new)
            l = l * alpha + p.sum(axis=-1, keepdims=True)
            acc = acc * alpha + p @ v
            m = m_new
            steps += 1
        out[r * B:(r + 1) * B] = acc / l
        steps_done.append(steps)
    return out, steps_done


for dtype in (np.float64, np.float32):
    Q = rng.standard_normal((TOK, D_HEAD)).astype(dtype)
    K = rng.standard_normal((TOK, D_HEAD)).astype(dtype)
    V = rng.standard_normal((TOK, D_HEAD)).astype(dtype)
    ref = full_attention(Q, K, V)
    got, steps = ring_attention(Q, K, V, RANKS)
    err = np.abs(got - ref).max()
    scale = np.abs(ref).max()
    print(f"{np.dtype(dtype).name}: 最大绝对误差 {err:.3e}，相对误差 {err/scale:.3e}")
    print(f"  因果剪枝后每台设备步数 {steps}（P={RANKS}，满额 {RANKS} 步）")

# ---- 成本模型：128K 上下文，d=128，bf16，8 卡 ----
S, d, P, BITS = 131072, 128, 8, 2
B = S // P
flops_peak, bw_link = 989e12, 900e9      # H100 fp16 张量核 / NVLink 单向有效
kv_full = S * d * BITS * 2               # K+V 全长
per_rank = (B * d * BITS) * 3            # Q、K、V 各一块
comm_step = 2 * B * d * BITS             # 一步传一个 K 块 + 一个 V 块
comm_total = comm_step * P               # 环形一圈被每台设备收到的总量
compute_step = 4 * B * B * d             # q·k 与 p·v 各 2B²d
t_comm = comm_step / bw_link
t_comp = compute_step / flops_peak
print(f"\nS={S} d={d} P={P} bf16: 全量 KV {kv_full/2**20:.0f} MiB → 单卡常驻 {per_rank/2**20:.0f} MiB")
print(f"每步通信 {comm_step/2**20:.1f} MiB ({t_comm*1e6:.1f} µs) / 每步计算 {compute_step/1e9:.1f} GFLOP ({t_comp*1e6:.0f} µs) → 通信占 {t_comm/t_comp:.1%}")
print(f"每台设备绕环一圈收到 {comm_total/2**20:.0f} MiB，等于全长 K+V 的 {comm_total/kv_full:.2f}x")
print(f"通信=计算的临界块长 B* = {BITS*flops_peak/(2*bw_link):.0f} token")
for p in (4, 8, 16, 32):
    b = S // p
    r_ = (2 * b * d * BITS / bw_link) / (4 * b * b * d / flops_peak)
    print(f"  P={p:2d} 块长 {b:6d} → 通信/计算 = {r_:.2f}")

# 因果剪枝下，块 i 只需 0..i 共 i+1 步 → 连续分片时各设备步数 1..P，利用率 (P+1)/(2P)
for p in (4, 8, 16):
    steps = np.arange(1, p + 1)
    print(f"P={p:2d} 连续分片步数 {steps.tolist()}，利用率 {steps.mean()/steps.max():.0%}")

# 8 个块分给 2 台主机：连续切 vs 首尾配对（zig-zag）
def cost(blocks):
    return sum(b + 1 for b in blocks)
print(f"8 块 / 2 主机 连续切 {cost([0,1,2,3])} vs {cost([4,5,6,7])}"
      f"；首尾配对 {cost([0,7])} vs {cost([3,4])}，另两对 {cost([1,6])} / {cost([2,5])}")
