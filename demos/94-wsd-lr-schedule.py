#!/usr/bin/env python3
"""第 94 篇 demo：余弦退火 vs WSD（warmup-stable-decay）学习率调度。

全部在本机 CPU 上用 numpy 跑（单隐层 tanh MLP + Adam，4096 样本的回归任务），
不是 GPU 实测、也不是 LLM；LLM 上的数字来自论文，正文分开标注。

E1 同预算：cosine(3000) vs WSD(3000)
E2 checkpoint 复用：WSD 中间 checkpoint + 10% cooldown，对比从头跑同样步数的 cosine
E3 续训：多给 1050 步，cosine 终点要 rewarm，WSD 稳定段 checkpoint 直接续
"""
import json
import math
import os

import numpy as np

SEED = int(os.environ.get('SEED', '20261126'))
RNG = np.random.default_rng(SEED)
D_IN, H, N_TRAIN, N_VAL = 8, 32, 4096, 4096
LR_MAX, B1, B2, EPS = 3e-3, 0.9, 0.999, 1e-8
T_TOTAL = 3000


def make_data(n, rng):
    X = rng.normal(size=(n, D_IN))
    a = np.array([1.0, 0.6, -0.4, 0.9, 0.3, -0.7, 0.5, 0.2])
    b = np.array([-0.3, 0.8, 0.7, -0.5, 0.9, 0.4, -0.6, 0.1])
    y = (np.sin(X @ a) + 0.5 * np.cos(2.0 * (X @ b))
         + 0.3 * np.sin(X[:, 0] * X[:, 1]) + 0.2 * X[:, 2] * X[:, 3])
    return X, y + rng.normal(scale=0.05, size=n)


XTR, YTR = make_data(N_TRAIN, RNG)
XVA, YVA = make_data(N_VAL, RNG)


def schedule(name, total, warmup=100, decay_frac=0.2):
    stable_end = total - int(total * decay_frac)

    def lr(t):
        if t < warmup:
            return LR_MAX * t / warmup
        if name == 'cosine':
            return LR_MAX * 0.5 * (1.0 + math.cos(math.pi * (t - warmup) / (total - warmup)))
        if name == 'wsd':
            if t < stable_end:
                return LR_MAX
            return LR_MAX * max(0.0, 1.0 - (t - stable_end) / (total - stable_end))
        raise ValueError(name)
    return lr


def init_params(rng):
    return {'W1': rng.normal(scale=1.0 / math.sqrt(D_IN), size=(D_IN, H)),
            'b1': np.zeros(H),
            'W2': rng.normal(scale=1.0 / math.sqrt(H), size=(H, 1)),
            'b2': np.zeros(1)}


def loss_and_grad(p, X, y):
    h = np.tanh(X @ p['W1'] + p['b1'])
    yhat = (h @ p['W2'] + p['b2']).ravel()
    err = yhat - y
    n = X.shape[0]
    g_y = (2.0 / n) * err[:, None]
    g = {'W2': h.T @ g_y, 'b2': g_y.sum(0)}
    g_z = (g_y @ p['W2'].T) * (1.0 - h ** 2)
    g['W1'] = X.T @ g_z
    g['b1'] = g_z.sum(0)
    return float(np.mean(err ** 2)), g


def evaluate(p):
    h = np.tanh(XVA @ p['W1'] + p['b1'])
    return float(np.mean(((h @ p['W2'] + p['b2']).ravel() - YVA) ** 2))


def adam_step(p, g, m, v, lr, step):
    for k in p:
        m[k] = B1 * m[k] + (1 - B1) * g[k]
        v[k] = B2 * v[k] + (1 - B2) * g[k] ** 2
        p[k] = p[k] - lr * (m[k] / (1 - B1 ** step)) / (np.sqrt(v[k] / (1 - B2 ** step)) + EPS)


def zeros_like(p):
    return {k: np.zeros_like(a) for k, a in p.items()}


def train(sched, steps, p=None, m=None, v=None, checkpoints=(), bc0=0):
    if p is None:
        p, m, v = init_params(RNG), None, None
        m, v = zeros_like(p), zeros_like(p)
    hist, ck = [], {}
    for i in range(steps):
        loss, g = loss_and_grad(p, XTR, YTR)
        adam_step(p, g, m, v, sched(bc0 + i), i + 1)
        hist.append(loss)
        if (i + 1) in checkpoints:
            ck[i + 1] = (dict(p), dict(m), dict(v), evaluate(p))
    return dict(p), dict(m), dict(v), hist, ck


def tail(xs, k=50):
    return float(np.mean(xs[-k:]))


def main():
    out = {}
    cos_sched, wsd_sched = schedule('cosine', T_TOTAL), schedule('wsd', T_TOTAL)

    # ---------- E1 ----------
    p_cos, m_cos, v_cos, h_cos, _ = train(cos_sched, T_TOTAL)
    p_wsd, m_wsd, v_wsd, h_wsd, ck = train(wsd_sched, T_TOTAL, checkpoints=(1000, 2000))
    out['E1'] = {
        'cosine_val': round(evaluate(p_cos), 4), 'wsd_val': round(evaluate(p_wsd), 4),
        'cosine_train_tail': round(tail(h_cos), 5), 'wsd_train_tail': round(tail(h_wsd), 5),
        'train_loss_at_1000_2000_3000': {
            'cosine': [round(float(np.mean(h_cos[i - 50:i])), 5) for i in (1000, 2000, 3000)],
            'wsd': [round(float(np.mean(h_wsd[i - 50:i])), 5) for i in (1000, 2000, 3000)]},
        'lr_at_1000_2000': {'cosine': [round(cos_sched(i), 6) for i in (1000, 2000)],
                            'wsd': [round(wsd_sched(i), 6) for i in (1000, 2000)]},
    }

    # ---------- E2 ----------
    br = {}
    for ckpt, (pc, mc, vc, val_before) in ck.items():
        dec = ckpt // 10
        bsched = schedule('wsd', ckpt + dec, warmup=1, decay_frac=dec / (ckpt + dec))
        pb, _, _, hb, _ = train(bsched, dec, p=pc, m=mc, v=vc, bc0=ckpt)
        br[str(ckpt)] = {'cooldown_steps': dec, 'val_before': round(val_before, 4),
                         'val_after': round(evaluate(pb), 4),
                         'train_tail_after': round(tail(hb), 5),
                         'extra_compute_frac_of_run': round(dec / T_TOTAL, 3)}

    cos_from_scratch = {}
    for ckpt in ck:
        tot = ckpt + ckpt // 10
        pc2, _, _, hc, _ = train(schedule('cosine', tot, warmup=min(100, tot // 10)), tot)
        cos_from_scratch[str(ckpt)] = {'total_steps': tot, 'val': round(evaluate(pc2), 4),
                                       'train_tail': round(tail(hc), 5)}

    out['E2'] = {'branches': br, 'cosine_from_scratch': cos_from_scratch,
                 'compute_wsd_one_run_plus_branches': T_TOTAL + sum(v['cooldown_steps'] for v in br.values()),
                 'compute_cosine_separate_runs': T_TOTAL + sum(v['total_steps'] for v in cos_from_scratch.values())}

    # ---------- E3 ----------
    # cosine 终点 LR=0，继续训练必须重新拉高 LR：warmup 100 + cosine 1050
    val_cos_before = evaluate(p_cos)
    p_rw, _, _, h_rw, _ = train(schedule('cosine', 1050, warmup=100), 1050, p=p_cos, m=m_cos, v=v_cos)
    # WSD 从 2000 步的稳定段 checkpoint 起，再跑 750 稳定 + 300 线性衰减
    pc, mc, vc, _ = ck[2000]
    val_wsd_before = evaluate(pc)
    p_ext, m_ext, v_ext, h_s, _ = train(lambda t: LR_MAX, 750, p=dict(pc), m=dict(mc), v=dict(vc), bc0=2000)
    p_ext, m_ext, v_ext, h_d, _ = train(schedule('wsd', 300, warmup=1, decay_frac=1.0), 300,
                                       p=p_ext, m=m_ext, v=v_ext)

    out['E3'] = {
        'budget_steps': 1050,
        'cosine_rewarm': {'val_before': round(val_cos_before, 4), 'val_after': round(evaluate(p_rw), 4),
                          'train_tail_before': round(tail(h_cos), 5), 'train_tail_after': round(tail(h_rw), 5),
                          'max_loss_during_rewarm': round(max(h_rw), 5)},
        'wsd_extend': {'val_before': round(val_wsd_before, 4), 'val_after': round(evaluate(p_ext), 4),
                       'train_tail_before': round(float(np.mean(h_wsd[1950:2000])), 5),
                       'train_tail_after': round(tail(h_d), 5),
                       'max_loss_during_extend': round(max(h_s), 5)},
    }
    out['seed'] = SEED
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
