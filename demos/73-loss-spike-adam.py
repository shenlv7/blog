#!/usr/bin/env python3
"""73 · 梯度尖峰与 Adam 二阶矩污染（CPU numpy，非 GPU 实测）

Part 1：标量精确解 —— 尖峰之后有效步长怎么变、多久恢复
Part 2：小型 MLP + 一批脏标签，五种处置方式的对照（5 个种子）
"""
import math
import numpy as np

B1, B2 = 0.9, 0.999


# ---------------- Part 1 ----------------
def scalar_trace(spike, warm=2000, horizon=12000, b1=B1, b2=B2):
    m = v = 0.0
    for t in range(1, warm + 1):
        m = b1 * m + (1 - b1) * 1.0
        v = b2 * v + (1 - b2) * 1.0
    base = (m / (1 - b1 ** warm)) / math.sqrt(v / (1 - b2 ** warm))
    out = []
    for k in range(horizon):
        g = spike if k == 0 else 1.0
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        t = warm + k + 1
        out.append((m / (1 - b1 ** t)) / math.sqrt(v / (1 - b2 ** t)) / base)
    return out


def scalar_report():
    print('== Part 1: 标量 Adam，warmup 2000 步后单步尖峰 s 倍，之后梯度回到 1 ==')
    print(f'{"s":>6} {"k=0":>7} {"k=1":>7} {"k=10":>7} {"k=100":>7} '
          f'{"k=1000":>8} {"k=5000":>8} {"恢复步数":>9}')
    for s in (10, 40, 100, 1000):
        tr = scalar_trace(s)
        rec = next((k for k in range(20, len(tr)) if tr[k] >= 0.9), -1)
        print(f'{s:>6} {tr[0]:>7.2f} {tr[1]:>7.2f} {tr[10]:>7.3f} {tr[100]:>7.3f} '
              f'{tr[1000]:>8.3f} {tr[5000]:>8.3f} {rec:>9}')
    print('理论上界：首步 = (1-b1)/sqrt(1-b2) = %.3f；'
          '尖峰后第 k 步的抑制 ≈ 1/sqrt(1 + s^2(1-b2)b2^k)'
          % ((1 - B1) / math.sqrt(1 - B2)))


# ---------------- Part 2 ----------------
def make_data(n=4096, d=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, d))
    y = (np.sin(3 * X[:, 0]) + 0.6 * X[:, 1] ** 2 + 0.3 * X[:, 2] * X[:, 3])[:, None]
    return X, y


class MLP:
    def __init__(self, d, h=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, math.sqrt(2 / d), (d, h))
        self.b1 = np.zeros(h)
        self.W2 = rng.normal(0, math.sqrt(2 / h), (h, 1))
        self.b2 = np.zeros(1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def loss_grads(self, X, y):
        z = X @ self.W1 + self.b1
        a = np.tanh(z)
        p = a @ self.W2 + self.b2
        loss = float(((p - y) ** 2).mean())
        g = 2 * (p - y) / len(X)
        gz = (g @ self.W2.T) * (1 - a ** 2)
        return loss, [X.T @ gz, gz.sum(0), a.T @ g, g.sum(0)]


def run(variant, clip_norm, seed, spike_scale=200.0, steps=1500, spike_at=400, lr=3e-3):
    X, y = make_data(seed=0)
    Xv, yv = make_data(n=1024, seed=999)
    rng = np.random.default_rng(100 + seed)
    m = MLP(8, seed=seed)
    ms = [np.zeros_like(p) for p in m.params()]
    vs = [np.zeros_like(p) for p in m.params()]
    val, stepn = {0: m.loss_grads(Xv, yv)[0]}, {}
    spike_gn = None
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(X), 64)
        xb, yb = X[idx], y[idx]
        bad = t == spike_at and variant != 'ctrl'
        if bad:
            yb = yb * spike_scale                 # 一批脏标签：数值被放大 spike_scale 倍
        _, gs = m.loss_grads(xb, yb)
        gnorm = math.sqrt(sum(float(np.sum(g ** 2)) for g in gs))
        if t == spike_at:
            spike_gn = gnorm
        if bad and variant == 'skip':
            continue                              # 整批丢弃，优化器状态不动
        if bad and variant in ('clip', 'reset'):
            s = min(1.0, clip_norm / gnorm)
            gs = [g * s for g in gs]
        tot = 0.0
        for i, (p, g) in enumerate(zip(m.params(), gs)):
            ms[i] = B1 * ms[i] + (1 - B1) * g
            vs[i] = B2 * vs[i] + (1 - B2) * g * g
            d = (ms[i] / (1 - B1 ** t)) / (np.sqrt(vs[i] / (1 - B2 ** t)) + 1e-8)
            p -= lr * d
            tot += float(np.sum(d ** 2))
        stepn[t] = math.sqrt(tot)                 # 单位学习率的参数位移
        if bad and variant == 'reset':
            ms = [np.zeros_like(p) for p in m.params()]
            vs = [np.zeros_like(p) for p in m.params()]
        if t % 25 == 0 or t == steps:
            val[t] = m.loss_grads(Xv, yv)[0]
    return dict(val=val, step=stepn, spike_gn=spike_gn)


def part2():
    print('\n== Part 2: MLP 8-64-1, Adam lr=3e-3, 1500 步, 第 400 步灌一批脏标签 ==')
    seeds = (0, 1, 2, 3, 4)
    ctrls, g400 = {}, []
    for sd in seeds:
        ctrls[sd] = run('ctrl', 1.0, sd)
        g400.append(ctrls[sd]['spike_gn'])
    clip_norm = 2.0 * float(np.median(g400))
    print(f'第 400 步正常梯度范数中位数 = {np.median(g400):.3f}，裁剪阈值取 2 倍 = {clip_norm:.3f}\n')

    def report(name, variant, scale=200.0):
        rs, rel = [], {400: [], 420: [], 450: [], 600: []}
        gn = []
        for sd in seeds:
            h = ctrls[sd] if variant == 'ctrl' else run(variant, clip_norm, sd, scale)
            c = ctrls[sd]
            if h['spike_gn']:
                gn.append(h['spike_gn'] / c['spike_gn'])
            for k in rel:
                if k in h['step']:
                    rel[k].append(h['step'][k] / c['step'][k])
            peak = max(v for t, v in h['val'].items() if t >= 400)
            rs.append((peak, h['val'][1500], h['val'][1500] / c['val'][1500]))
        a = np.array(rs)
        r = {k: (np.mean(v) if v else float('nan')) for k, v in rel.items()}
        print('%-10s 梯度×%5.0f | 步长比 400:%5.2f 420:%5.2f 450:%5.2f 600:%5.2f | '
              '峰值val %7.3f 末val %.4f 末/对照 %.2f±%.2f'
              % (name, np.mean(gn) if gn else 1.0, r[400], r[420], r[450], r[600],
                 a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 2].std()))

    report('ctrl', 'ctrl')
    report('noclip×200', 'noclip', 200.0)
    report('noclip×2000', 'noclip', 2000.0)
    report('clip@2×', 'clip')
    report('skip', 'skip')
    report('clip+reset', 'reset')


if __name__ == '__main__':
    scalar_report()
    part2()
