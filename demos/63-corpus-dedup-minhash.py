#!/usr/bin/env python3
"""63 · 语料去重：MinHash + LSH banding 的真实候选量、召回与代价。

构造受控近重复语料（250 个源文档 × 8 个编辑强度，其中 200 个源带共享模板页脚），
用 MinHash 签名 + 分带 LSH 找候选对，和 oracle（同源）对比。
输出：不同 (b, r) 下的候选对数、召回、精确率；shingle 长度的影响；模板页脚造成的假阳性；
      按文档频率砍掉高频 shingle 之后的效果。
"""
import itertools
import random
import time
import zlib
from collections import defaultdict

import numpy as np

M31 = 2147483647  # 梅森素数，a*hv 不会溢出 int64
SEED = 20261026
rng = random.Random(SEED)
nprng = np.random.default_rng(SEED)

# ---------- 语料构造 ----------
SYL = ['ka', 'to', 'ri', 'men', 'sol', 'van', 'dex', 'qua', 'lun', 'pex',
       'tor', 'mis', 'gal', 'zen', 'fyr', 'nub', 'cor', 'hast', 'vel', 'wyn']
VOCAB = [a + b for a in SYL for b in SYL[:10]]  # 200 个伪词
DOC_TOKENS = 140
N_SRC = 250
N_BOILER = 200
EDIT_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60]  # 7 个变体
# 24 个词的共享页脚：版权声明 + 免责模板，几乎所有真实站点都有
BOILER = ('all rights reserved this document is provided as is without warranty '
          'of any kind either express or implied see the license file for the '
          'full terms and conditions').split()


def make_source():
    return [rng.choice(VOCAB) for _ in range(DOC_TOKENS)]


def mutate(doc, frac):
    """按比例做 替换/删除/插入，得到一篇编辑过的变体。"""
    if frac == 0.0:
        return list(doc)
    n = len(doc)
    k = max(1, int(round(n * frac)))
    mode_r, mode_d = 0.5, 0.3
    out = list(doc)
    ops = []
    for _ in range(k):
        x = rng.random()
        ops.append('r' if x < mode_r else ('d' if x < mode_r + mode_d else 'i'))
    for op in ops:
        if not out:
            break
        pos = rng.randrange(len(out))
        if op == 'r':
            out[pos] = rng.choice(VOCAB)
        elif op == 'd':
            out.pop(pos)
        else:
            out.insert(pos, rng.choice(VOCAB))
    return out


def shingles(tokens, k):
    return {' '.join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)}


def hash_shingles(sh):
    return np.unique(np.fromiter((zlib.crc32(s.encode()) % M31 for s in sh),
                                 dtype=np.int64))


# ---------- MinHash ----------
def minhash_signatures(hashes_list, num_perm, a, b):
    sig = np.empty((len(hashes_list), num_perm), dtype=np.int64)
    for i, H in enumerate(hashes_list):
        if H.size == 0:
            sig[i] = M31 - 1
            continue
        # (num_perm, m)：每行一个置换，取每行的最小值
        sig[i] = ((a[:, None] * H[None, :] + b[:, None]) % M31).min(axis=1)
    return sig


# ---------- LSH banding ----------
def lsh_pairs(sig, bands, rows):
    buckets = defaultdict(list)
    for j in range(bands):
        seg = sig[:, j * rows:(j + 1) * rows]
        for idx in range(sig.shape[0]):
            buckets[seg[idx].tobytes()].append(idx)
    pairs = set()
    for bucket in buckets.values():
        if len(bucket) > 1:
            pairs.update(itertools.combinations(bucket, 2))
    return pairs


def jaccard(a, b):
    if not a and not b:
        return 1.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def build_corpus():
    docs, truth, tag = [], [], []
    for sid in range(N_SRC):
        src = make_source()
        if sid < N_BOILER:
            src = src + BOILER          # 一半以上的源文档共享同一段页脚
        docs.append(src)
        truth.append(sid)
        tag.append('copy')
        for f in EDIT_LEVELS:
            docs.append(mutate(src, f))
            truth.append(sid)
            tag.append(f'{f:g}')
    return docs, np.array(truth), tag


def evaluate(sig, same_pairs, n, sep='[3]'):
    print(f'\n{sep} LSH banding（签名 {sig.shape[1]}）：候选量 vs 召回 vs 精确率')
    print(f'{"b×r":>7} | {"阈值":>6} | {"候选对":>8} | {"占全量":>7} | '
          f'{"召回":>6} | {"副本召回":>8} | {"精确率":>6}')
    for bands, rows in [(8, 16), (16, 8), (16, 4), (16, 2),
                        (32, 4), (32, 2), (48, 2), (64, 2)]:
        if bands * rows > sig.shape[1]:
            continue
        thr = (1.0 / bands) ** (1.0 / rows)
        cand = lsh_pairs(sig, bands, rows)
        hit = len(cand & same_pairs)
        rec = hit / len(same_pairs)
        rec_dup = len(cand & DUP_PAIRS) / len(DUP_PAIRS)
        prec = hit / len(cand) if cand else 0.0
        print(f'{bands:3d}×{rows:<3d} | {thr:6.3f} | {len(cand):8d} | '
              f'{len(cand)/(n*(n-1)/2)*100:6.2f}% | {rec:6.3f} | '
              f'{rec_dup:8.3f} | {prec:6.3f}')


DUP_PAIRS = set()


def main():
    global DUP_PAIRS
    t0 = time.time()
    docs, truth, tag = build_corpus()
    n = len(docs)
    print(f'语料：{n} 篇（{N_SRC} 源 × 8 篇），平均长度 '
          f'{np.mean([len(d) for d in docs]):.1f} 词；{N_BOILER} 个源带 24 词共享页脚')

    groups = defaultdict(list)
    for i, s in enumerate(truth):
        groups[s].append(i)
    same_pairs = set()
    for g in groups.values():
        same_pairs.update(itertools.combinations(g, 2))
    DUP_PAIRS = {(s * 8, s * 8 + 1) for s in range(N_SRC)}
    print(f'oracle 同源对 {len(same_pairs)} 对；全量 {n*(n-1)//2} 对；'
          f'1e6 篇时暴力是 5e11 次交集')

    # ---------- [1] shingle 长度 ----------
    print('\n[1] 同一批编辑强度，只看 shingle 长度怎么改分数')
    print(f'{"编辑比例":>8} | {"3-gram":>7} | {"5-gram":>7} | {"8-gram":>7}')
    sets = {k: [shingles(d, k) for d in docs] for k in (3, 5, 8)}
    probe = list(range(0, 100))
    for f_i, f in enumerate(EDIT_LEVELS):
        row = []
        for k in (3, 5, 8):
            vals = [jaccard(sets[k][s * 8], sets[k][s * 8 + 1 + f_i]) for s in probe]
            row.append(np.mean(vals))
        print(f'{f:8.2f} | {row[0]:7.3f} | {row[1]:7.3f} | {row[2]:7.3f}')

    # ---------- [2] MinHash ----------
    K, PERM = 5, 128
    hashes_list = [hash_shingles(sh) for sh in sets[K]]
    a = nprng.integers(1, M31, size=PERM, dtype=np.int64)
    b = nprng.integers(0, M31, size=PERM, dtype=np.int64)
    t1 = time.time()
    sig = minhash_signatures(hashes_list, PERM, a, b)
    dt = time.time() - t1
    print(f'\n[2] MinHash：{PERM} 置换 × {n} 篇 {dt:.2f}s'
          f'（{n/dt:.0f} 篇/秒，单核）；签名矩阵 {sig.nbytes/1e6:.2f} MB')
    errs = []
    check = [(s * 8, s * 8 + 1) for s in range(150)]
    check += [(rng.randrange(n), rng.randrange(n)) for _ in range(300)]
    for i, j in check:
        if i != j:
            errs.append(abs(np.mean(sig[i] == sig[j]) -
                            jaccard(sets[K][i], sets[K][j])))
    print(f'    签名估计的绝对误差：均值 {np.mean(errs):.4f}，'
          f'p95 {np.percentile(errs, 95):.4f}')

    # ---------- [3] 各配置 ----------
    evaluate(sig, same_pairs, n)

    # ---------- [4] 逐档召回 ----------
    cand = lsh_pairs(sig, 32, 4)
    print('\n[4] 固定 b=32,r=4（阈值≈0.50）：按编辑比例看召回')
    print(f'{"编辑比例":>8} | {"真 Jaccard":>10} | {"召回":>6} | {"签名估计":>8}')
    for f_i, f in enumerate(EDIT_LEVELS):
        tot = hit = 0
        est = []
        for s in range(N_SRC):
            i0, i1 = s * 8, s * 8 + 1 + f_i
            tot += 1
            hit += (i0, i1) in cand
            est.append(np.mean(sig[i0] == sig[i1]))
        j_true = np.mean([jaccard(sets[K][s * 8], sets[K][s * 8 + 1 + f_i])
                          for s in range(N_SRC)])
        print(f'{f:8.2f} | {j_true:10.3f} | {hit/tot:6.3f} | {np.mean(est):8.3f}')

    # ---------- [5] 假阳性来源 ----------
    print('\n[5] 假阳性从哪来')
    fp = cand - same_pairs
    n_boiler = len({i // 8 for i in range(n)} & set(range(N_BOILER)))
    print(f'    候选 {len(cand)} 对，跨源 {len(fp)} 对（'
          f'{len(fp)/max(1,len(cand))*100:.1f}%）；带页脚的源 {n_boiler} 个')
    if fp:
        with_b = sum(1 for i, j in fp if i // 8 < N_BOILER and j // 8 < N_BOILER)
        print(f'    其中两端都带页脚：{with_b} 对（{with_b/len(fp)*100:.1f}%）')
        sims = [jaccard(sets[K][i], sets[K][j]) for i, j in list(fp)[:400]]
        print(f'    跨源对真实 Jaccard：均值 {np.mean(sims):.3f}，'
              f'最大 {np.max(sims):.3f}（样本 {len(sims)}）')

    # ---------- [6] 砍高频 shingle ----------
    print('\n[6] 按文档频率砍掉「到处都是」的 shingle（df > 5%）之后再跑')
    df = defaultdict(int)
    for sh in sets[K]:
        for s in sh:
            df[s] += 1
    cut = {s for s, c in df.items() if c > 0.05 * n}
    print(f'    被砍掉的 shingle：{len(cut)} 个 / 共 {len(df)} 个'
          f'（占 {len(cut)/len(df)*100:.2f}%）')
    sets2 = [sh - cut for sh in sets[K]]
    before = np.mean([len(s) for s in sets[K]])
    after = np.mean([len(s) for s in sets2])
    print(f'    每篇 shingle 数 {before:.1f} → {after:.1f}'
          f'（砍掉 {before-after:.1f} 个，占 {(before-after)/before*100:.1f}%）')
    hashes2 = [hash_shingles(sh) for sh in sets2]
    sig2 = minhash_signatures(hashes2, PERM, a, b)
    evaluate(sig2, same_pairs, n, sep='[7]')

    print(f'\n总耗时 {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
