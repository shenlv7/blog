#!/usr/bin/env python3
"""KV cache 驱逐：H2O 与 随机/滑窗 的对照（纯 numpy 离散模拟，不是 GPU 实测）。

模拟的注意力形状取自已知经验事实：质量集中在 attention sink + 少量 heavy hitter
（对多数 query 稳定），另有局部性偏好与逐 query 噪声。不模拟真实权重，
所以结论只在"注意力分布形状"这一层成立。

度量（都以 full cache 为基准）：
  mass   : 保留集合上的注意力质量占 full 的比例（1.0 = 没丢东西）
  needle : 该 query 的注意力峰值 token 是否还在缓存里
  p_peak : 驱逐后峰值 token 拿到的新概率（full 里的值作为参照）
实验 2 把 token 1000..1007 设成"前 200 个 query 不相关、之后才相关"，
用来看基于历史注意力的驱逐在时间上的盲区。
"""
import numpy as np

T, DV, NQ, BUD = 2048, 32, 256, 410


def build_logits(rng, nq=NQ, t=T, hh=24):
    hh_idx = rng.choice(np.arange(1, t), size=hh, replace=False)
    base = 0.9 * rng.normal(size=(nq, t))
    base[:, 0] += 4.0                     # attention sink
    base[:, hh_idx] += 4.5                # heavy hitters
    base[:, -64:] += 1.2                  # 局部性
    return base, hh_idx


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def run(nq=NQ, budget=BUD, seed=0, delayed=False):
    rng = np.random.default_rng(seed)
    L, hh = build_logits(rng, nq=nq)
    recent, n_keep = 128, budget - 128 - 1
    h2o = set(range(T))               # 先用满缓存观察到第 8 个 query，再开始淘汰
    rnd = set([0]) | set(range(T - recent, T)) | set(
        rng.choice(np.arange(1, T - recent), size=n_keep, replace=False).tolist())
    win = set(range(T - budget, T))
    pol = {'h2o': h2o, 'random': rnd, 'window': win}
    acc = {k: dict(mass=[], needle=[], ppeak=[]) for k in ('full', *pol)}
    got, want = {}, {}
    scores = np.zeros(T)
    late = np.arange(1000, 1008)
    for q in range(nq):
        lg = L[q].copy()
        if delayed:
            lg[late] += -8.0 if q < 200 else 6.0   # 早期完全不相关，第 200 个 query 突然相关
        a_full = softmax(lg)
        peak_full = int(a_full.argmax())
        p_full = a_full[peak_full]
        acc['full']['mass'].append(1.0); acc['full']['needle'].append(1.0)
        acc['full']['ppeak'].append(1.0)
        if delayed and q >= 200:
            m = float(a_full[late].sum())
            want['full'] = want.get('full', 0.0) + m
            got['full'] = got.get('full', 0.0) + m
        for name, s in pol.items():
            keep = np.zeros(T, bool); keep[list(s)] = True
            a = softmax(np.where(keep, lg, -np.inf))
            acc[name]['mass'].append(float(a_full[keep].sum()))
            acc[name]['needle'].append(float(keep[peak_full]))
            acc[name]['ppeak'].append(float(a[peak_full] if keep[peak_full] else 0.0) / p_full)
            if name == 'h2o':
                a_h2o = a
            if delayed and q >= 200:
                got[name] = got.get(name, 0.0) + float(a[late].sum())
                want[name] = want.get(name, 0.0) + float(a_full[late].sum())
        scores += a_h2o          # 只累计"真的能看见的"注意力：被淘汰的 token 永远拿不到新分
        if q >= 8:                        # 先积累注意力再开始淘汰
            h2o = set([0]) | set(range(T - recent, T))
            cand = np.argsort(-scores[1:T - recent])[:n_keep] + 1
            h2o |= set(cand.tolist())
            pol['h2o'] = h2o
    res = {k: {m: float(np.mean(v)) for m, v in d.items()} for k, d in acc.items()}
    res['hh'] = len(hh)
    res['late'] = {k: got[k] / max(want[k], 1e-9) for k in got}
    return res


if __name__ == '__main__':
    print(f'模拟配置: T={T} 预算={BUD}({BUD/T:.0%}) 最近窗口=128 sink=1 nq={NQ}\n')
    for seed in (0, 1, 2):
        r = run(seed=seed)
        print(f'seed={seed}')
        for name in ('full', 'h2o', 'random', 'window'):
            d = r[name]
            print(f"  {name:6s} 质量保留={d['mass']:.3f}  峰值留存={d['needle']:.3f}  峰值概率占比={d['ppeak']:.3f}")
    print('\n== 延迟相关：token 1000..1007 直到第 200 个 query 才相关 ==')
    for seed in (0, 1, 2):
        r = run(seed=seed, delayed=True)
        print(f"seed={seed}: " + '  '.join(f"{n} 该段质量回收={r['late'][n]:.3f}" for n in ('full', 'h2o', 'random', 'window')))
