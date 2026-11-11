#!/usr/bin/env python3
"""79 · 递归训练自己的输出：尾部与词汇表是怎么消失的

本地离散模拟，不是 GPU / 真实 LLM 实测。
模型族 = 无上下文的一元语言模型（unigram），真实分布 = Zipf(s=1.1, V=2000)。
每一代：从上一代模型采样 n=2000 个 token → 做 MLE 重新数频次 → 得到下一代模型。
对照组：每代额外掺入 2000 个真实样本（累积式）；以及加性平滑 α>0。
"""
import numpy as np

V = 2000                      # 词表大小
S = 1.1                       # Zipf 指数（合成数据，不是语料拟合值）
N = 2000                      # 每代采样量
GENS = 12
SEEDS = 5
TAIL_FROM = 1000              # 排名 1001..2000 视为「尾部」


def zipf_true(v=V, s=S):
    p = 1.0 / np.arange(1, v + 1) ** s
    return p / p.sum()


TRUE = zipf_true()
TAIL_MASS_TRUE = TRUE[TAIL_FROM:].sum()
TV = 0.5  # TV 距离里的常数，下面用两倍


def sample(p, n, rng):
    return rng.choice(V, size=n, p=p)


def mle(counts, alpha=0.0):
    """一元模型：加性平滑后的 MLE。alpha=0 就是纯数频次（没见过的词概率为 0）。"""
    c = counts + alpha
    return c / c.sum()


def tv(p, q):
    return 0.5 * np.abs(p - q).sum()


def run(alpha=0.0, accumulate_real=False, seed=0):
    rng = np.random.default_rng(seed)
    p = mle(np.bincount(sample(TRUE, N, rng), minlength=V), alpha)  # 第 1 代：真实数据训练
    rows = []
    for g in range(1, GENS + 1):
        if g > 1:
            x = sample(p, N, rng)
            if accumulate_real:                      # 累积真实数据：不清空旧数据
                x = np.concatenate([x, sample(TRUE, N, rng)])
            p = mle(np.bincount(x, minlength=V), alpha)
        alive = int((p > 0).sum())
        tail = p[TAIL_FROM:].sum()
        dead_mass = TRUE[p == 0].sum()               # 真实分布里落在「被判死」类型上的质量
        head10 = np.sort(p)[-10:].sum()
        ent = -(p[p > 0] * np.log2(p[p > 0])).sum()
        rows.append((g, alive, tail, tv(p, TRUE), dead_mass, head10, ent))
    return rows


def show(name, rows_all):
    print(f'\n{name}')
    print(f'{"代":>3} {"存活词数":>8} {"尾部质量":>8} {"TV距离":>7} {"真值落在死词上":>12} {"top10占比":>9} {"熵(bit)":>8}')
    for i in range(GENS):
        alive = np.array([r[i][1] for r in rows_all])
        tail = np.array([r[i][2] for r in rows_all])
        d = np.array([r[i][3] for r in rows_all])
        dm = np.array([r[i][4] for r in rows_all])
        h10 = np.array([r[i][5] for r in rows_all])
        e = np.array([r[i][6] for r in rows_all])
        print(f'{i+1:>3} {alive.mean():>8.0f} {tail.mean():>7.2%} {d.mean():>7.4f} '
              f'{dm.mean():>12.2%} {h10.mean():>8.2%} {e.mean():>8.2f}')
    return rows_all


print(f'真实分布：Zipf(s={S}, V={V})，尾部（排名 {TAIL_FROM+1}-{V}）真实质量 = '
      f'{TAIL_MASS_TRUE:.2%}，每代采样 n={N}，{GENS} 代，{SEEDS} 个随机种子')

base = show('A. 纯递归（每代只用上一代的样本，无平滑）',
            [run(0.0, False, s) for s in range(SEEDS)])
acc = show('B. 累积真实数据（每代掺入 n 个真实样本）',
           [run(0.0, True, s) for s in range(SEEDS)])
sm = show('C. 只加平滑 α=0.5（不掺真实数据）',
          [run(0.5, False, s) for s in range(SEEDS)])

print('\n关键差值（第 1 代 → 第 12 代）：')
for name, rows in (('A 纯递归', base), ('B 累积真实数据', acc), ('C 平滑 α=0.5', sm)):
    alive1 = np.mean([r[0][1] for r in rows]); alive12 = np.mean([r[-1][1] for r in rows])
    tail1 = np.mean([r[0][2] for r in rows]); tail12 = np.mean([r[-1][2] for r in rows])
    d1 = np.mean([r[0][3] for r in rows]); d12 = np.mean([r[-1][3] for r in rows])
    print(f'{name}: 词表 {alive1:.0f}→{alive12:.0f}，'
          f'尾部 {tail1:.3%}→{tail12:.3%}，TV {d1:.4f}→{d12:.4f}')

# 解析侧：α 平滑每代把「偏离均匀」按 n/(n+αV) 收缩；灭绝阈 p*≈1/n
factor = N / (N + 0.5 * V)
uni = np.full(V, 1.0 / V)
print(f'\n解析校验：α=0.5 的收缩因子 n/(n+αV) = {factor:.4f}，11 代后残差 {factor ** 11:.4%}')
print(f'与均匀分布的 TV（理论极限）= {tv(uni, TRUE):.4f}，实测第 12 代 = '
      f'{sm[0][-1][3]:.4f}（{sm[0][-1][3] / tv(uni, TRUE):.0%}）')
print(f'真实分布里 p < 1/n = 1/{N} 的类型 {int((TRUE < 1 / N).sum())} 个，'
      f'合计质量 {TRUE[TRUE < 1 / N].sum():.2%} —— 这就是采样噪声的灭绝线')

# 长跑：纯递归跑 40 代，看塌缩有没有自停止
rng = np.random.default_rng(0)
p = mle(np.bincount(sample(TRUE, N, rng), minlength=V))
print('\n纯递归长跑（seed=0）：')
for g in range(1, 41):
    p = mle(np.bincount(sample(p, N, rng), minlength=V))
    if g in (1, 6, 12, 20, 30, 40):
        print(f'  第 {g:2d} 代：存活 {int((p > 0).sum()):3d} 个，TV {tv(p, TRUE):.4f}，'
              f'top10 {np.sort(p)[-10:].sum():.2%}，熵 {-(p[p>0]*np.log2(p[p>0])).sum():.2f} bit')
