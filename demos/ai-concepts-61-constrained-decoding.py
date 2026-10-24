#!/usr/bin/env python3
"""第 61 篇 demo：语法约束解码（JSON logit masking），字符级 n-gram 模型 + JSON 下推自动机 + BPE 词表。

不是 GPU 实测、不是真实 LLM：被测模型是一个从合成 JSON 语料统计出来的字符级 3-gram，
唯一目的是让「合法 token 集合」这个问题可以被精确测量。

跑法：python3 61-constrained-decoding.py
"""
import json
import random
import re
from collections import defaultdict

import numpy as np

DIG = set('0123456789')
STATUS = ['ok', 'error', 'pending']
TAGS = ['nlp', 'cv', 'infra', 'rl', 'audio', 'graph', 'eval', 'sec']

# ---------------------------------------------------------------- 1. 语料
def make_corpus(n=4000, seed=7):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        rec = {
            'id': rng.randint(1, 9999),
            'status': rng.choice(STATUS),
            'score': round(rng.uniform(0, 1), 2),
            'tags': rng.sample(TAGS, rng.randint(1, 3)),
        }
        out.append(json.dumps(rec, separators=(',', ':'), sort_keys=False))
    return out


# ---------------------------------------------------------------- 2. 字符级 n-gram（带 backoff）
class CharNGram:
    def __init__(self, corpus, order=3, alpha=0.4):
        self.order, self.alpha = order, alpha
        self.counts = [defaultdict(lambda: defaultdict(float)) for _ in range(order + 1)]
        self.vocab = set()
        for line in corpus:
            s = '\x02' * order + line
            for i, c in enumerate(s):
                ctx = s[max(0, i - order):i]
                for k in range(order + 1):
                    self.counts[k][ctx[-k:] if k else ''][c] += 1
                self.vocab.add(c)
        self.vocab = sorted(self.vocab)
        self.chars = np.array(self.vocab)

    def probs(self, ctx):
        """返回 {char: 概率}，按最长可用上下文 backoff。"""
        for k in range(min(self.order, len(ctx)), -1, -1):
            d = self.counts[k].get(ctx[-k:] if k else '')
            if d and sum(d.values()) >= 8:
                tot = sum(d.values()) + self.alpha * len(self.vocab)
                return {c: (v + self.alpha) / tot for c, v in d.items()}, True
        n = len(self.vocab)
        return {c: 1.0 / n for c in self.vocab}, False

    def sample(self, ctx, rng, allowed=None, temp=0.75):
        p, _ = self.probs(ctx)
        if allowed is not None:
            p = {c: v for c, v in p.items() if c in allowed}
            if not p:                      # 掩码把概率全清零 = 死局，回退到均匀
                p = {c: 1.0 for c in allowed}
        ks = list(p)
        w = np.array([p[c] for c in ks], dtype=np.float64) ** (1.0 / temp)
        w = w / w.sum()
        return ks[min(int(np.searchsorted(np.cumsum(w), rng.random())), len(ks) - 1)]


# ---------------------------------------------------------------- 3. BPE 词表
def build_bpe(corpus, target=260):
    """极简 BPE：只做字符对合并，词表 = 基础字符 + 合并出的片段。"""
    from collections import Counter
    seqs = [list(s) for s in corpus]
    vocab = set(''.join(corpus))
    merges = []
    while len(vocab) < target:
        pairs = Counter()
        for s in seqs:
            for a, b in zip(s, s[1:]):
                pairs[(a, b)] += 1
        if not pairs:
            break
        (a, b), cnt = pairs.most_common(1)[0]
        if cnt < 20:
            break
        new = a + b
        merges.append((a, b))
        vocab.add(new)
        merged = []
        for s in seqs:
            i, cur = 0, []
            while i < len(s):
                if i + 1 < len(s) and s[i] == a and s[i + 1] == b:
                    cur.append(new); i += 2
                else:
                    cur.append(s[i]); i += 1
            merged.append(cur)
        seqs = merged
    return merges, sorted(vocab)


def tokenize(s, merges):
    toks = list(s)
    for a, b in merges:
        out, i = [], 0
        while i < len(toks):
            if i + 1 < len(toks) and toks[i] == a and toks[i + 1] == b:
                out.append(a + b); i += 2
            else:
                out.append(toks[i]); i += 1
        toks = out
    return toks


# ---------------------------------------------------------------- 4. JSON 下推自动机（增量、不可变状态）
# state = (mode, stack, extra)
#   mode: value | key | key2 | arrval | arrval2 | instr | colon | num | lit | after | done
#   stack: tuple of 'o'/'a'（对象/数组的括号栈）
def s_value():  return ('value', (), None)


_ALLOWED_CACHE = {}


def allowed(st):
    if st in _ALLOWED_CACHE:
        return _ALLOWED_CACHE[st]
    out = _allowed(st)
    _ALLOWED_CACHE[st] = out
    return out


def _allowed(st):
    mode, stack, extra = st
    if mode in ('value', 'arrval', 'arrval2'):
        a = {'{', '[', '"', '-'} | DIG | {'t', 'f', 'n'}
        if mode == 'arrval':  a |= {']'}
        return a
    if mode == 'key':   return {'"', '}'}
    if mode == 'key2':  return {'"'}
    if mode == 'colon': return {':'}
    if mode == 'instr':
        role, esc = extra
        if esc: return set('"\\/bfnrt')
        return set('\x22\\') | ESC_OK_SET
    if mode == 'num':
        num = extra
        a = set(DIG)
        if '.' not in num: a |= {'.'}
        if num not in ('-',) and not num.endswith('.'):
            a |= {',', '}', ']'}
        return a
    if mode == 'lit':   return {extra[0]}
    if mode == 'after':
        if not stack: return set()
        return {',', '}'} if stack[-1] == 'o' else {',', ']'}
    return set()


ESC_OK_SET = {chr(i) for i in range(32, 127)} | {'\t'}


_STEP_CACHE = {}


def step(st, c):
    k = (st, c)
    if k in _STEP_CACHE:
        return _STEP_CACHE[k]
    out = _step(st, c)
    _STEP_CACHE[k] = out
    return out


def _step(st, c):
    mode, stack, extra = st
    if mode in ('value', 'arrval', 'arrval2'):
        if c == '{':  return ('key', stack + ('o',), None)
        if c == '[':  return ('arrval', stack + ('a',), None)
        if c == '"':  return ('instr', stack, ('val', False))
        if c in DIG or c == '-': return ('num', stack, c)
        if c in 'tfn': return ('lit', stack, {'t': 'rue', 'f': 'alse', 'n': 'ull'}[c])
        return None
    if mode == 'key':
        if c == '"': return ('instr', stack, ('key', False))
        if c == '}' and stack and stack[-1] == 'o':
            ns = stack[:-1]
            return ('after', ns, None) if ns else ('done', ns, None)
        return None
    if mode == 'key2':
        return ('instr', stack, ('key', False)) if c == '"' else None
    if mode == 'colon':
        return ('value', stack, None) if c == ':' else None
    if mode == 'instr':
        role, esc = extra
        if esc:
            return ('instr', stack, (role, False)) if c in '"\\/bfnrt' else None
        if c == '\\': return ('instr', stack, (role, True))
        if c == '"':
            return ('after', stack, None) if role == 'val' else ('colon', stack, None)
        if 32 <= ord(c) < 127: return ('instr', stack, (role, False))
        return None
    if mode == 'num':
        if c in DIG: return ('num', stack, extra + c)
        if c == '.' and '.' not in extra: return ('num', stack, extra + '.')
        if c in ',}]' and extra not in ('-',) and not extra.endswith('.'):
            ns = stack
            return _step(('after', ns, None), c)    # 数字在这里收尾，分隔符交给 after
        return None
    if mode == 'lit':
        if c != extra[0]: return None
        rest = extra[1:]
        return ('after', stack, None) if not rest else ('lit', stack, rest)
    if mode == 'after':
        if not stack: return None
        if c == ',':
            return ('key2' if stack[-1] == 'o' else 'arrval2', stack, None)
        if c == '}' and stack[-1] == 'o':
            ns = stack[:-1]
            return ('after', ns, None) if ns else ('done', ns, None)
        if c == ']' and stack[-1] == 'a':
            ns = stack[:-1]
            return ('after', ns, None) if ns else ('done', ns, None)
        return None
    return None


# ---------------------------------------------------------------- 5. 词表前缀树 + 掩码枚举
def build_trie(vocab):
    root = {}
    for t in vocab:
        node = root
        for ch in t:
            node = node.setdefault(ch, {})
        node['#'] = t
    return root


def allowed_tokens(trie, st, stats=None):
    """用前缀树枚举当前状态下允许的 token，节点数 ∝ 合法前缀数，不是 |V|。"""
    out, nodes = [], 0

    def dfs(node, s):
        nonlocal nodes
        nodes += 1
        if '#' in node:
            out.append(node['#'])
        A = allowed(s)
        for ch, child in node.items():
            if ch == '#':
                continue
            if ch in A:
                ns = step(s, ch)
                if ns is not None:
                    dfs(child, ns)

    dfs(trie, st)
    if stats is not None:
        stats['nodes'] += nodes
    return out


# ---------------------------------------------------------------- 6. 生成
def gen_unconstrained(model, rng, max_len=200, temp=0.75):
    ctx, out = '\x02' * model.order, []
    for _ in range(max_len):
        c = model.sample(ctx, rng, temp=temp)
        out.append(c)
        ctx = (ctx + c)[-model.order:]
    return ''.join(out)


def gen_constrained(model, rng, trie, vocab, token_level=True, max_steps=400, temp=0.75):
    ctx, st, out, steps, frac = '\x02' * model.order, s_value(), [], 0, []
    while steps < max_steps and st[0] != 'done':
        if token_level:
            cand = allowed_tokens(trie, st)
        else:
            cand = list(allowed(st))
        if not cand:
            return ''.join(out), steps, frac, False
        frac.append(len(cand) / (len(vocab) if token_level else 95))
        if token_level:
            # 用首字符的概率给 token 打分（真实系统对每个 token 算完整 logprob）
            p, _ = model.probs(ctx)
            scored = {}
            for t in cand:
                q = 1.0
                for ch in t[:3]:
                    q *= p.get(ch, 1e-6)
                scored[t] = q
            ks = list(scored)
            w = np.array([scored[t] for t in ks]) ** (1.0 / temp)
            w = w / w.sum()
            tok = ks[min(int(np.searchsorted(np.cumsum(w), rng.random())), len(ks) - 1)]
        else:
            tok = model.sample(ctx, rng, allowed=set(cand), temp=temp)
        for ch in tok:
            ns = step(st, ch)
            if ns is None:
                return ''.join(out), steps, frac, False
            st = ns
        out.append(tok)
        ctx = (ctx + tok)[-model.order:]
        steps += 1
    return ''.join(out), steps, frac, st[0] == 'done'


# ---------------------------------------------------------------- 7. 评测
def schema_ok(s):
    """返回 (类别, 说明)。strict 才是真正想要的，语法合法 ≠ 合规。"""
    dup = []

    def hook(pairs):
        ks = [k for k, _ in pairs]
        dup.extend([k for k in ks if ks.count(k) > 1])
        return dict(pairs)

    try:
        r = json.loads(s, object_pairs_hook=hook)
    except Exception:
        return 'invalid', 'JSON 语法非法'
    if not isinstance(r, dict):
        return 'notobj', '顶层不是对象'
    if dup:
        return 'dupkey', f'重复键 {dup[0]}'
    if set(r) != {'id', 'status', 'score', 'tags'}:
        return 'keys', f'键集合不对 {sorted(r)[:4]}'
    if not isinstance(r['status'], str) or r['status'] not in STATUS:
        return 'value', f"status 越界 {str(r.get('status'))[:12]}"
    if not isinstance(r['score'], (int, float)) or not (0 <= r['score'] <= 1):
        return 'value', f"score 越界 {str(r.get('score'))[:12]}"
    if not isinstance(r['tags'], list) or not all(t in TAGS for t in r['tags']):
        return 'value', 'tags 含未知名'
    if not isinstance(r['id'], int):
        return 'value', 'id 非整数'
    return 'strict', ''


def measure_intervention(model, rng, N=200, temp=0.35):
    """掩码到底改写了多少比例的预测：每步看模型 top-1 是否本来合法。"""
    ill = tot = 0
    for _ in range(N):
        ctx, st = '\x02' * model.order, s_value()
        while st[0] != 'done' and tot < 40000:
            p, _ = model.probs(ctx)
            top = max(p, key=p.get)
            A = allowed(st)
            tot += 1
            if top not in A:
                ill += 1
            c = model.sample(ctx, rng, allowed=A, temp=temp)
            st = step(st, c)
            ctx = (ctx + c)[-model.order:]
    return ill, tot


def main():
    import time
    t_bpe = time.perf_counter()
    corpus = make_corpus()
    model = CharNGram(corpus)
    merges, vocab = build_bpe(corpus[:1500])
    trie = build_trie(vocab)
    print(f'语料 {len(corpus)} 条 | 字符表 {len(model.vocab)} | BPE 合并 {len(merges)} 次 | '
          f'词表 {len(vocab)} | 建词表 {time.perf_counter() - t_bpe:.1f}s')
    print(f'样例: {corpus[0]}')

    N = 250
    cats = ['invalid', 'notobj', 'dupkey', 'keys', 'value', 'strict']
    # ① 无约束
    rng = np.random.default_rng(2026)
    cnt = defaultdict(int)
    ex = []
    for _ in range(N):
        s = gen_unconstrained(model, rng)
        t, _w = schema_ok(s)
        cnt[t] += 1
        if len(ex) < 2 and t == 'invalid':
            ex.append(s[:66])
    print(f'\n① 无约束（字符级，200 字符截断）: ' + ' '.join(f'{c}={cnt[c]}' for c in cats) +
          f' | JSON 语法合法 {N - cnt["invalid"]}/{N}')
    # 公平性对照（零假设）：无约束 + 先知截断 —— 只在能解析的 '}' 处截，排除「不知道该停」
    rng = np.random.default_rng(2026)
    oracle = 0
    for _ in range(N):
        s = gen_unconstrained(model, rng)
        for j in range(1, len(s) + 1):
            if s[j - 1] == '}':
                try:
                    json.loads(s[:j]); oracle += 1; break
                except Exception:
                    pass
    print(f'   对照：无约束 + 先知截断（在每个右括号处试解析）: JSON 语法合法 {oracle}/{N} = {oracle / N:.1%}')
    for e in ex:
        print('   例:', e)

    # ② 三种温度 × 字符级/token 级掩码
    print('\n② 约束解码（400 条，采样温度扫描）')
    print('   | 温度 | 粒度 | 语法合法 | strict | ' + ' | '.join(cats[1:5]) + ' | 平均步数 | 平均合法比例 |')
    detail = {}
    for temp in (0.35, 0.5, 0.75):
        for label, tok_level in (('char', False), ('token', True)):
            rng = np.random.default_rng(2026)
            cnt = defaultdict(int)
            steps_all, frac_all, bad = [], [], []
            for _ in range(N):
                s, steps, frac, done = gen_constrained(model, rng, trie, vocab,
                                                       token_level=tok_level, temp=temp)
                t, why = schema_ok(s)
                cnt[t] += 1
                steps_all.append(steps)
                frac_all.extend(frac)
                if t != 'strict' and t != 'invalid' and len(bad) < 2:
                    bad.append((s[:74], why))
            syn = N - cnt['invalid']
            print(f'   | {temp:.2f} | {label:5s} | {syn:3d}/{N} | {cnt["strict"]:3d} '
                  f'({cnt["strict"] / N:.1%}) | ' +
                  ' | '.join(f'{cnt[c]:3d}' for c in cats[1:5]) +
                  f' | {np.mean(steps_all):5.1f} | {np.mean(frac_all):.1%} |')
            detail[(temp, label)] = bad
    for s, why in detail[(0.35, 'char')][:2]:
        print(f'   ✗ 0.35/char {why}: {s}')

    # ③ 剪枝强度
    print('\n③ 剪枝强度（token 级，典型一条生成的逐步统计）')
    rng2 = np.random.default_rng(5)
    s0, _, _, _ = gen_constrained(model, rng2, trie, vocab, temp=0.5)
    st = s_value()
    by_mode = defaultdict(list)
    for ch in s0:
        if st[0] == 'done':
            break
        by_mode[st[0]].append((len(allowed(st)), len(allowed_tokens(trie, st))))
        st = step(st, ch)
    print('   模式     步数 | 平均允许字符 | 平均允许 token（占词表）')
    for mode, rows in by_mode.items():
        print(f'   {mode:8s} {len(rows):3d} | {np.mean([r[0] for r in rows]):6.1f} | '
              f'{np.mean([r[1] for r in rows]):6.1f}/{len(vocab)} '
              f'({np.mean([r[1] for r in rows]) / len(vocab):5.1%})')

    # ③b 掩码改写比例
    ill, tot = measure_intervention(model, np.random.default_rng(9))
    print(f'\n③b 掩码改写：模型 top-1 每步本该输出非法字符的比例 {ill}/{tot} = {ill / tot:.1%}')

    # ④ 掩码成本：前缀树 DFS vs 朴素逐 token 模拟，两个词表规模
    print('\n④ 掩码成本')
    for V_target in (230, 20000):
        if V_target == 230:
            V, tr = vocab, trie
        else:
            r = np.random.default_rng(3)
            alphabet = list('abcdefghijklmnopqrstuvwxyz0123456789_{}[]":,.')

            def rnd_tok():
                L = int(r.integers(2, 9))
                return ''.join(alphabet[int(r.integers(0, len(alphabet)))] for _ in range(L))

            big = set(rnd_tok() for _ in range(V_target))
            big |= set(alphabet)
            V, tr = sorted(big), build_trie(sorted(big))
        st, nodes, hits = s_value(), 0, 0
        t0 = time.perf_counter()
        for ch in s0:
            if st[0] == 'done':
                break
            stats = {'nodes': 0}
            hits += len(allowed_tokens(tr, st, stats))
            nodes += stats['nodes']
            st = step(st, ch)
        t_trie = time.perf_counter() - t0
        st, ops = s_value(), 0
        t0 = time.perf_counter()
        for ch in s0:
            if st[0] == 'done':
                break
            A = allowed(st)
            for tk in V:
                cur, ok = st, True
                for c in tk:
                    if c not in A:
                        ok = False
                        break
                    cur = step(cur, c)
                    ops += 1
                    if cur is None:
                        ok = False
                        break
            st = step(st, ch)
        t_naive = time.perf_counter() - t0
        print(f'   |V|={len(V):6d} | 前缀树 {t_trie * 1000:8.2f} ms（访问节点 {nodes}） '
              f'| 朴素 {t_naive * 1000:8.2f} ms（字符转移 {ops}） | 朴素/前缀树 {t_naive / t_trie:5.1f}x')
    print('   注：|V|=20000 的是随机字符串词表，只为测规模趋势，不是真实 tokenizer')


if __name__ == '__main__':
    main()
