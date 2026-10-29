#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI 的进化之路 · 03 (1950)
(A) 图灵那句「70%」作为统计命题要多少样本才站得住；
(B) 一个按 ELIZA(1966) 结构写的最小对话机，看「不理解的回答」长什么样。
纯标准库，/usr/bin/python3 直接跑。
"""
from math import comb
import re

# ---------------------------------------------------------------- A
# 每个判官一轮判定是一个伯努利：判对了机器就暴露。
# H0: p = 0.5（机器与人在同一侧，判官只能猜）
# H1: p = 0.30（图灵 1950 年给的阈值：平均判官判对的概率不超过 30%）
# 判定规则：n 轮里判对 k 次，若 k <= c 就宣布「通过」，c 取使 alpha <= 0.05 的最大值。

def cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))

def rule(n, alpha=0.05):
    c = -1
    for k in range(n + 1):
        if cdf(k, n, 0.5) <= alpha:
            c = k
    if c < 0:
        return None
    return c, cdf(c, n, 0.5), cdf(c, n, 0.30)

print("=== A. 把「70%」当假设检验：需要多少轮 ===")
print("n = 轮数(每轮一个判官一次判定)")
print("alpha = 机器其实只在猜、却被判通过的概率    power = 机器真到 30% 时能被判出来的概率")
print()
print("  n | 允许判对的上限 c |  alpha  | power")
print("----+------------------+---------+-------")
for n in [1, 2, 4, 8, 12, 16, 20, 24, 30, 36, 40, 50, 64, 80]:
    r = rule(n)
    if r is None:
        print(f"{n:3d} |   无法设定(样本太小)   |    -    |   -")
    else:
        c, a, pw = r
        print(f"{n:3d} |        {c:3d}         | {a:.4f}  | {pw:.4f}")

need = next(n for n in range(1, 500) if rule(n) and rule(n)[2] >= 0.80)
c, a, pw = rule(need)
print(f"\n达到 alpha<=0.05 且 power>=0.80 的最小轮数：n = {need}")
print(f"  规则为「{need} 轮中判对不超过 {c} 次即宣布通过」"
      f"→ 误判率 {a:.4f}，检出率 {pw:.4f}")

print("\n单局（n=1，也就是图灵写的「五分钟一次问答」）：")
print(f"  机器只在猜时被宣布通过的概率 = {cdf(0, 1, 0.5):.2f}")
print(f"  机器真到 30% 时被判通过的概率 = {cdf(0, 1, 0.30):.2f}")

print("\n多数表决把结论推向哪边（单判官被蒙住的概率固定为 0.70）：")
for n in [1, 3, 5, 7, 9, 11]:
    m = n // 2 + 1
    p = sum(comb(n, i) * 0.7 ** i * 0.3 ** (n - i) for i in range(m, n + 1))
    print(f"  {n:2d} 个判官多数投票，宣布通过的概率 = {p:.4f}")

# ---------------------------------------------------------------- B
# 最小 ELIZA：关键词 + 秩 + 分解模式 + 重组模板，零语义。
REFLECT = {"i": "you", "me": "you", "my": "your", "am": "are",
           "you": "i", "your": "my", "are": "am", "myself": "yourself"}
NORM = {"i'm": "i am", "i've": "i have", "don't": "do not", "you're": "you are"}

# (关键词, 秩, 分解模式, 重组模板)
RULES = [
    ("alike",   10, "* alike",     ["in what way", "what resemblance do you see",
                                    "what does that similarity suggest to you"]),
    ("always",   5, "* always *",  ["can you think of a specific example", "when",
                                    "really, always"]),
    ("sorry",    4, "* sorry *",   ["please do not apologise", "apologies are not necessary",
                                    "what feelings do you have when you apologise"]),
    ("everyone", 2, "* everyone *",["surely not everyone", "who, for example",
                                    "who are you thinking of"]),
    ("nobody",   2, "* nobody *",  ["are you thinking of a very special person",
                                    "who, for example", "surely someone"]),
    ("i am",     1, "i am *",      ["how long have you been {0}",
                                    "do you believe it is normal to be {0}",
                                    "why do you say you are {0}"]),
    ("depressed",3, "* depressed *",["i am sorry to hear you are depressed",
                                    "do you think coming here will help you not to be depressed"]),
    ("need",     2, "* need *",    ["why do you need {1}", "would it really help you to get {1}" ]),
    ("i",        0, "i * you",     ["perhaps in your fantasy we {0} each other",
                                    "why do you {0} me",
                                    "you like to think that you {0} me"]),
    ("you",      0, "you are *",   ["what makes you think i am {0}",
                                    "does it please you to believe i am {0}",
                                    "perhaps you would like to be {0}"]),
    ("my",       0, "my *",        ["your {0}", "why do you say your {0}",
                                    "does your {0} matter to you"]),
]
DEFAULT = ["i see", "please go on", "hmm, tell me more", "does that suggest anything to you"]
TEMPLATES = sum(len(r[3]) for r in RULES) + len(DEFAULT)


def compile_pattern(pat):
    toks = pat.split()
    parts = []
    for i, tok in enumerate(toks):
        if tok == "*":
            parts.append("(.*)" if i == len(toks) - 1 else "(.+?)")
        else:
            parts.append(re.escape(tok))
    return re.compile(r"^" + r"\s+".join(parts) + r"$", re.I)


class Eliza:
    def __init__(self):
        self.turn = 0
        self.used = []

    def reply(self, text):
        # 与 1966 年的实现一致：先按标点切成单句，只变换其中一句
        clauses = [self._norm(c.strip()) for c in re.split(r"[.!?,;:]", text) if c.strip()]
        joined = " " + " ".join(clauses) + " "
        best = None
        for kw, rank, pat, outs in RULES:
            if f" {kw} " in joined and (best is None or rank > best[1]):
                best = (kw, rank, pat, outs)
        if best:
            rx = compile_pattern(best[2])
            for cl in clauses:
                m = rx.match(cl)
                if m:
                    slots = [self._reflect(g) for g in m.groups()]
                    tpl = best[3][self.turn % len(best[3])]
                    self.turn += 1
                    self.used.append(tpl)
                    return tpl.format(*slots)
        self.turn += 1
        tpl = DEFAULT[self.turn % len(DEFAULT)]
        self.used.append(tpl)
        return tpl

    @staticmethod
    def _norm(s):
        for k, v in NORM.items():
            s = s.replace(k, v)
        return s.lower()

    @staticmethod
    def _reflect(s):
        return " ".join(REFLECT.get(w.lower(), w) for w in s.split())


SCRIPT = [
    "Men are all alike.",
    "They're always bugging us about something or other.",
    "Well, my boyfriend made me come here.",
    "He says I'm depressed much of the time.",
    "It's true. I am unhappy.",
    "I need some help, that much seems certain.",
    "Perhaps I could learn to get along with my mother.",
    "You are like my father in some ways.",
    "You don't argue with me.",
]

print("\n=== B. 一个不会理解的对话机（ELIZA 结构，约 70 行） ===")
e = Eliza()
for line in SCRIPT:
    print(f"人 > {line}")
    print(f"机 > {e.reply(line)}")
    print()

print("--- 这一轮对话的账面 ---")
print(f"输入 {len(SCRIPT)} 句，产出 {len(SCRIPT)} 句")
print(f"用到的不同模板数：{len(set(e.used))}；整个模板池一共 {TEMPLATES} 条")
print(f"平均输入长度：{sum(len(s) for s in SCRIPT) / len(SCRIPT):.1f} 字符")
print("程序里没有词义表、没有世界知识、没有对话状态；每个回答只由三样东西决定：")
print("  ① 句子里出现了哪个关键词  ② 该关键词的秩  ③ 模式里被 * 捕获的那一段")
import collections
print("默认兜底句出现次数：", collections.Counter(x for x in e.used if x in DEFAULT))
