"""ai-concepts-91: goodput —— SLO 之下的吞吐才算吞吐
离散事件模拟（不是 GPU 实测）：连续批处理 + 分块 prefill。
两个实验：
  A. 负载固定 5 req/s，扫「显存允许的最大批大小」→ 看扩容的收益在哪停
  B. 批大小固定 128（KV cache 上限），扫到达率 → 看吞吐与 goodput 何时分岔
纯标准库，总运行约 3 秒。
"""
import random

SEED = 91
PRE_CAP = 1.5       # TTFT SLO：从到达到首 token ≤ 1500 ms
TPOT_CAP = 0.060    # TPOT SLO：相邻 token 间隔 ≤ 60 ms（≈17 tok/s）

# 成本模型（自上而下近似，不是实测值）
PREFILL_RATE = 11000.0     # prefill 聚合吞吐 tok/s（算力受限）
PREFILL_BUDGET = 2048      # 每迭代最多推进的 prefill token（分块 prefill）
DEC_BASE = 0.010           # 每迭代固定开销 s
DEC_PER_SEQ = 0.00028      # 每迭代每序列附加耗时 s（KV 读写 ≈ 线性）
KW_PER_TOKEN = 128 * 1024  # 每 token KV：32 层 × 8 KV 头 × 128 维 × fp16
N_REQ = 1500


def sample_prompt(rng):
    return int(min(4000, max(100, rng.lognormvariate(6.55, 0.55))))   # 中位数 ~700


def sample_output(rng):
    return rng.randint(128, 320)                                       # 均值 224


def percentile(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q / 100.0
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def simulate(max_batch, arrival_rate):
    rng = random.Random(SEED)
    queue, decoding, prefilling, done = [], [], [], []
    t, next_arrival, n_arrived = 0.0, rng.expovariate(arrival_rate), 0
    t_prev, area_insys = 0.0, 0.0      # 时间加权：系统内请求数的平均值

    while len(done) < N_REQ:
        while n_arrived < N_REQ and next_arrival <= t:
            queue.append({'arrive': next_arrival, 'P': sample_prompt(rng),
                          'O': sample_output(rng)})
            n_arrived += 1
            next_arrival += rng.expovariate(arrival_rate)

        free = max_batch - len(decoding) - len(prefilling)   # KV cache 槽位
        while queue and free > 0:
            r = queue.pop(0)
            r.update(prompt_left=r['P'], o_left=r['O'], ttft=None)
            prefilling.append(r)
            free -= 1

        if not decoding and not prefilling:
            t = max(t, next_arrival)
            continue

        # 一个迭代：分块 prefill + 所有就绪序列各生成 1 个 token
        budget, used, ready = PREFILL_BUDGET, 0, []
        for r in prefilling:
            take = min(r['prompt_left'], budget)
            r['prompt_left'] -= take
            budget -= take
            used += take
            if r['prompt_left'] == 0:
                ready.append(r)
            if budget <= 0:
                break

        t_end = t + used / PREFILL_RATE + DEC_BASE + DEC_PER_SEQ * (len(decoding) + len(ready))
        area_insys += (len(queue) + len(decoding) + len(prefilling)) * (t_end - t_prev)
        t_prev = t_end

        for r in ready:
            prefilling.remove(r)
            r['ttft'] = t_end
            decoding.append(r)

        alive = []
        for r in decoding:
            r['o_left'] -= 1
            if r['o_left'] == 0:
                r['finish'] = t_end
                done.append(r)
            else:
                alive.append(r)
        decoding = alive
        t = t_end

    ttfts = [r['ttft'] - r['arrive'] for r in done]
    tpots = [(r['finish'] - r['ttft']) / (r['O'] - 1) for r in done]
    met = [1 for p, q in zip(ttfts, tpots) if p <= PRE_CAP and q <= TPOT_CAP]
    sojourn = [r['finish'] - r['arrive'] for r in done]
    return {
        'batch': max_batch, 'lam': arrival_rate, 'elapsed': t,
        'rps': len(done) / t, 'tokps': sum(r['O'] for r in done) / t,
        'p50_ttft': percentile(ttfts, 50), 'p99_ttft': percentile(ttfts, 99),
        'p50_tpot': percentile(tpots, 50),
        'goodput': len(met) / t, 'met': len(met) / len(done),
        'insys_avg': area_insys / t,
        'littles_law': (len(done) / t) * (sum(sojourn) / len(sojourn)),
    }


def kv_capacity(gpu_gb=80, weights_gb=16, ctx=4096):
    return int((gpu_gb - weights_gb) * 1024 ** 3 / (KW_PER_TOKEN * ctx))


HDR = (f"{'最大批':>5} {'λ':>4} {'耗时s':>6} {'吞吐req/s':>9} {'tok/s':>6} "
       f"{'P50TTFT':>8} {'P99TTFT':>8} {'P50TPOT':>8} {'goodput':>8} {'达标率':>6}")
def row(r):
    return (f"{r['batch']:>5} {r['lam']:>4.1f} {r['elapsed']:>6.0f} {r['rps']:>9.2f} "
            f"{r['tokps']:>6.0f} {r['p50_ttft']*1000:>7.0f}ms {r['p99_ttft']*1000:>7.0f}ms "
            f"{r['p50_tpot']*1000:>7.0f}ms {r['goodput']:>8.2f} {r['met']:>5.0%}")


if __name__ == '__main__':
    print(f'{N_REQ} 个请求；SLO: TTFT≤{PRE_CAP*1000:.0f}ms、TPOT≤{TPOT_CAP*1000:.0f}ms')
    print(f'80G 卡 / 16G 权重 / 4k 上下文 → KV cache 上限 {kv_capacity()} 条\n')
    print('实验 A：λ=5 req/s，扫最大批大小')
    print(HDR); print('-' * 88)
    for B in (8, 16, 32, 48, 64, 96, 128):
        print(row(simulate(B, 5.0)))
    print('\n实验 B：最大批=128（显存上限），扫到达率')
    print(HDR); print('-' * 88)
    for lam in (4.0, 5.0, 6.0, 7.0, 8.0, 10.0):
        print(row(simulate(128, lam)))
    r = simulate(128, 10.0)
    print(f"\nLittle 定律自检（λ=10）：队列时间加权均值 {r['insys_avg']:.0f} 条，"
          f"λ×平均停留 = {r['littles_law']:.0f} 条")
