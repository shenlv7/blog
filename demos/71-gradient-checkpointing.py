"""激活重算（gradient checkpointing）最小实验：numpy 手写反向，CPU 实跑。

模型：L 个残差 MLP 块，x -> x + relu(x W1) W2，损失 = 0.5*sum(x^2)
比较三种策略：
  A 全存（标准反向）
  B 每层检查点
  C 每 sqrt(L) 层检查点（子线性显存）
指标：需要保留的激活字节数、额外前向 FLOPs、真实墙钟时间、梯度最大误差。
"""
import numpy as np
import time
import math

rng = np.random.default_rng(0)
B, T, D, EXP, L = 4, 128, 128, 4, 32
H = EXP * D
N = B * T * D            # 每个 (B,T,D) 张量的元素数
FLOAT = 4                # float32 字节
W1 = [(rng.standard_normal((D, H)) * math.sqrt(2.0 / D)).astype(np.float32) for _ in range(L)]
W2 = [(rng.standard_normal((H, D)) * (0.05 / math.sqrt(H))).astype(np.float32) for _ in range(L)]
x0 = rng.standard_normal((B, T, D)).astype(np.float32)


def fwd_block(x, i):
    h = x @ W1[i]
    a = np.maximum(h, 0.0)
    y = a @ W2[i]
    return x + y, (x, h, a)


def bwd_block(dy, cache, i):
    x, h, a = cache
    dW2 = a.reshape(-1, H).T @ dy.reshape(-1, D)
    da = dy @ W2[i].T
    dh = da * (h > 0)
    dW1 = x.reshape(-1, D).T @ dh.reshape(-1, H)
    dx = dy + dh @ W1[i].T
    return dx, dW1, dW2


def loss_grad(x):
    return 0.5 * float((x * x).sum()), x


# 每个块缓存 3 个 (B,T,D) 大小的张量：x、h、a（h 只用来取 relu mask，按同尺寸计）
BYTES_PER_CACHE = 3 * N * FLOAT
BYTES_PER_BOUNDARY = N * FLOAT
FLOPS_FWD_BLOCK = 2 * 2 * B * T * D * H      # 两次 matmul，2 FLOP/MAC
FLOPS_BWD_BLOCK = 2 * FLOPS_FWD_BLOCK        # 反向约为前向 2 倍


def run_full():
    x = x0
    caches = []
    for i in range(L):
        x, c = fwd_block(x, i)
        caches.append(c)
    loss, dx = loss_grad(x)
    for i in reversed(range(L)):
        dx, _, _ = bwd_block(dx, caches[i], i)
    return loss, dx, L * BYTES_PER_CACHE, L * FLOPS_FWD_BLOCK * 3


def run_layer_ckpt():
    xs = [x0]
    x = x0
    for i in range(L):
        x, _ = fwd_block(x, i)
        xs.append(x)                      # 只留块边界
    loss, dx = loss_grad(x)
    peak = len(xs) * BYTES_PER_BOUNDARY
    for i in reversed(range(L)):
        # 从边界重算这一块的中间量，算完立刻丢弃
        _, c = fwd_block(xs[i], i)
        peak = max(peak, len(xs) * BYTES_PER_BOUNDARY + BYTES_PER_CACHE)
        dx, _, _ = bwd_block(dx, c, i)
    return loss, dx, peak, (L * FLOPS_FWD_BLOCK * 3) + L * FLOPS_FWD_BLOCK


def run_sqrt_ckpt():
    k = math.ceil(math.sqrt(L))
    seg = math.ceil(L / k)
    bounds = list(range(0, L, seg))
    xs = {0: x0}
    x = x0
    for i in range(L):
        x, _ = fwd_block(x, i)
        if (i + 1) in bounds:
            xs[i + 1] = x                 # 每 seg 层留一个边界
    loss, dx = loss_grad(x)
    peak = len(xs) * BYTES_PER_BOUNDARY
    extra = 0
    for b in reversed(bounds):
        end = min(b + seg, L)
        y = xs[b]
        caches = []
        for i in range(b, end):           # 重算整段并缓存该段中间量
            y, c = fwd_block(y, i)
            caches.append(c)
        extra += (end - b) * FLOPS_FWD_BLOCK
        peak = max(peak, len(xs) * BYTES_PER_BOUNDARY + len(caches) * BYTES_PER_CACHE)
        for i in reversed(range(b, end)):
            dx, _, _ = bwd_block(dx, caches[i - b], i)
    total = 3 * L * FLOPS_FWD_BLOCK + extra
    return loss, dx, peak, total


def timed(fn, n=3):
    best = 1e9
    for _ in range(n):
        t = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t)
    return out, best


def report(name, fn):
    (loss, dx, peak, flops), t = timed(fn)
    print(f"{name:14s} peak={peak/2**20:7.2f} MiB  TFLOP={flops/1e9:7.2f} "
          f"time={t*1e3:7.1f} ms  loss={loss:.6f}  dx[0,0,:3]={dx[0,0,:3]}")
    return dx


print(f"B={B} T={T} D={D} L={L} H={H}  每块缓存={BYTES_PER_CACHE/2**20:.2f} MiB")
d_full = report("A 全存", run_full)
d_layer = report("B 每层检查点", run_layer_ckpt)
d_sqrt = report("C sqrt 检查点", run_sqrt_ckpt)
print("梯度最大绝对误差  B vs A:", float(np.abs(d_layer - d_full).max()))
print("梯度最大绝对误差  C vs A:", float(np.abs(d_sqrt - d_full).max()))

# 成本模型：换算到 8B 级模型一层（B*T=8192, D=4096, H=4D）
for bt, d in [(8192, 4096)]:
    blk = 3 * bt * d * 4
    print(f"\n[成本模型] B*T={bt} D={d}: 单块缓存 {blk/2**20:.0f} MiB, "
          f"80 层全存 {80*blk/2**30:.1f} GiB, 每层检查点 {81*bt*d*4/2**30:.2f} GiB, "
          f"sqrt 检查点 10 边界+9 段缓存 {(10+9)*bt*d*4/2**20:.0f} MiB")

# ---- 附：随机性必须复现。块内加 dropout，比较「重算时重新抽 mask」与「恢复 RNG 状态」----
def blk_fwd_drop(x, i, r, rate=0.1):
    h = x @ W1[i]
    a = np.maximum(h, 0.0)
    keep = ((r.random(a.shape) >= rate).astype(np.float32) / (1.0 - rate))
    return x + (a * keep) @ W2[i], (x, h, a, keep)


def blk_bwd_drop(dy, c, i):
    x, h, a, keep = c
    dW2 = (a * keep).reshape(-1, H).T @ dy.reshape(-1, D)
    dh = (dy @ W2[i].T) * keep * (h > 0)
    dW1 = x.reshape(-1, D).T @ dh.reshape(-1, H)
    return dy + dh @ W1[i].T, dW1, dW2


def drop_run(Ls=8, restore_rng=True):
    r = np.random.default_rng(7)
    states, xs, x = [], [x0], x0
    for i in range(Ls):
        states.append(r.bit_generator.state)   # 块的 RNG 状态（重算前必须复位到这里）
        x, _ = blk_fwd_drop(x, i, r)
        xs.append(x)
    loss, dx = loss_grad(x)
    for i in reversed(range(Ls)):
        if restore_rng:
            r.bit_generator.state = states[i]
        _, c = blk_fwd_drop(xs[i], i, r)       # 重算
        dx, _, _ = blk_bwd_drop(dx, c, i)
    return dx


def drop_ref(Ls=8):
    r = np.random.default_rng(7)
    x, cs = x0, []
    for i in range(Ls):
        x, c = blk_fwd_drop(x, i, r)
        cs.append(c)
    loss, dx = loss_grad(x)
    for i in reversed(range(Ls)):
        dx, _, _ = blk_bwd_drop(dx, cs[i], i)
    return dx


dref = drop_ref()
for tag, flag in [("重抽 mask（错）", False), ("恢复 RNG 状态", True)]:
    d = drop_run(restore_rng=flag)
    rel = float(np.abs(d - dref).max() / max(np.abs(dref).max(), 1e-12))
    print(f"dropout 重算 · {tag:14s} 相对误差 {rel:.3e}")
