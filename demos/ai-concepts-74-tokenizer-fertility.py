"""74 篇 demo：同一段内容的中英两版，在 cl100k / o200k 两代词表下的 token 账。

只依赖 tiktoken（离线词表文件已缓存）。运行：uv run --no-project --with tiktoken python 74-tokenizer-fertility.py
"""
import tiktoken

SEGMENTS = [
    ("推理瓶颈",
     "注意力机制在推理阶段的主要瓶颈不是浮点运算，而是显存带宽：每生成一个 token，都要把整个 KV cache 读一遍。批处理能摊薄权重读取，却摊薄不了每个序列各自的缓存。",
     "At inference time the bottleneck of attention is not floating-point arithmetic but memory bandwidth: every generated token requires reading the entire KV cache. Batching amortizes weight reads, but it cannot amortize each sequence's own cache."),
    ("UTF-8 底账",
     "中文的每个汉字在 UTF-8 下占三个字节，而一个英文字母只占一个字节。分词器在字节层面做合并，所以同样一段意思，中文起始的符号序列长度就是英文的三倍左右。",
     "Each Chinese character occupies three bytes in UTF-8, while an English letter occupies only one. Tokenizers merge at the byte level, so for the same meaning the initial symbol sequence for Chinese is roughly three times longer than for English."),
    ("词表的取舍",
     "词表越大，能装下的整词越多，序列就越短；代价是嵌入矩阵和输出层的参数变多，长尾 token 的梯度也更稀疏。",
     "A larger vocabulary packs more whole words and therefore shorter sequences; the price is a bigger embedding matrix and output layer, plus sparser gradients for rare tokens."),
    ("口语版",
     "说白了，模型不认识汉字，它只认识编号。你给它的每个字，都要先被切成一串编号，切得碎，账单就厚。",
     "Put plainly, the model does not know Chinese characters, only ids. Every character you hand it is first sliced into a run of ids, and the finer the slices, the thicker the bill."),
]

ENCS = {name: tiktoken.get_encoding(name) for name in ("cl100k_base", "o200k_base")}


def cjk_len(s):
    return sum(1 for ch in s if "\u4e00" <= ch <= "\u9fff")


def word_len(s):
    return len(s.split())


print("=" * 78)
print("同一段内容的两种语言版本，token 数对比（内容平行，语义等价）")
print("=" * 78)
print(f"{'段落':<10}{'汉字':>5}{'英文词':>7}{'cl100k中':>9}{'cl100k英':>9}{'o200k中':>8}{'o200k英':>8}")
rows = []
for name, zh, en in SEGMENTS:
    t = {}
    for k, enc in ENCS.items():
        t[k] = (len(enc.encode(zh)), len(enc.encode(en)))
    rows.append((name, cjk_len(zh), word_len(en), t))
    print(f"{name:<10}{cjk_len(zh):>5}{word_len(en):>7}"
          f"{t['cl100k_base'][0]:>9}{t['cl100k_base'][1]:>9}"
          f"{t['o200k_base'][0]:>8}{t['o200k_base'][1]:>8}")

print()
print("=" * 78)
print("聚合：token/汉字、token/英文词，以及中文相对英文的膨胀倍数")
print("=" * 78)
for k in ENCS:
    zh_tok = sum(r[3][k][0] for r in rows)
    en_tok = sum(r[3][k][1] for r in rows)
    zh_ch = sum(r[1] for r in rows)
    en_w = sum(r[2] for r in rows)
    print(f"{k:<14} 中文 {zh_tok:>4} tok / {zh_ch} 字 = {zh_tok/zh_ch:.3f} tok/字 | "
          f"英文 {en_tok:>4} tok / {en_w} 词 = {en_tok/en_w:.3f} tok/词 | "
          f"同内容中/英 = {zh_tok/en_tok:.2f}x")

print()
print("=" * 78)
print("换成运营口径：一份 100 万 token 的预算，能装多少内容")
print("=" * 78)
for k in ENCS:
    zh_tok = sum(r[3][k][0] for r in rows)
    en_tok = sum(r[3][k][1] for r in rows)
    zh_ch = sum(r[1] for r in rows)
    en_w = sum(r[2] for r in rows)
    per_tok_zh = zh_ch / zh_tok
    per_tok_en = en_w / en_tok
    win = 128_000
    print(f"{k:<14} 100 万 token ≈ {1_000_000*per_tok_zh/10000:5.1f} 万汉字 / "
          f"{1_000_000*per_tok_en/10000:5.1f} 万英文词 | "
          f"128k 窗口 ≈ {win*per_tok_zh/10000:5.1f} 万汉字")

print()
print("=" * 78)
print("切碎在哪：同一句话的 token 边界（cl100k vs o200k）")
print("=" * 78)
probe = "分布式训练的梯度累积需要按 token 数加权，否则长序列会被系统性低估。"
for k, enc in ENCS.items():
    ids = enc.encode(probe)
    pieces = [enc.decode([i]) for i in ids]
    print(f"{k:<14} {len(ids)} tok: {' | '.join(pieces)}")

def token_bytes(enc, text):
    """返回每个 token 覆盖的原始字节数（不需 re-encode，用 decode 的长度不稳，故直接看 id 的字节长度）"""
    out = []
    for tid in enc.encode(text):
        b = enc.decode_single_token_bytes(tid)
        out.append(len(b))
    return out


print()
print("=" * 78)
print("每个字符要几个 token（含中文标点；token 跨字时两边各记一次）")
print("=" * 78)
for k, enc in ENCS.items():
    dist = {1: 0, 2: 0, 3: 0, "4+": 0}
    for _, zh, _ in SEGMENTS:
        tb = token_bytes(enc, zh)
        bounds, acc = [], 0
        for n in tb:
            bounds.append((acc, acc + n))
            acc += n
        pos = 0
        for ch in zh:
            n = len(ch.encode())
            span = (pos, pos + n)
            hit = sum(1 for (a, b) in bounds if a < span[1] and b > span[0])
            dist[hit if hit in dist else "4+"] += 1
            pos += n
    total = sum(dist.values())
    print(f"  {k:<14} " + " | ".join(f"{kk} tok: {vv/total:.0%}" for kk, vv in dist.items())
          + f"  (共 {total} 字符)")

