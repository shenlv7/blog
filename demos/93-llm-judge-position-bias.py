#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""93 · LLM-as-a-judge 的位置偏差从哪来：允许 tie 与强制二选一，同一批配对

真实 API 实测（deepseek-chat，temperature=0，2026-11-25 单次运行），三段：
  A. 判别力对照：正确 vs 错误（两个方向）
  B. 长度偏差：同一份正确内容的精简版 vs 注水版（两个方向）
  C. 零假设：两份逐字相同的答案——标注良好的裁判应 100% 判 tie
  D. 契约对照：把 C 和 B 换成"必须二选一"，看 A/B 位置上的胜率是否失衡
指标：胜率、严格翻转率（换顺序后选中的语义是否改变）、分数差随位置的变化、tie 率。
输出：93-result.json + stdout 明细。
"""
import json, os, sys, time, urllib.request
from math import sqrt

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"

Q = [
    ("一副标准 52 张扑克牌随机抽一张，是红桃的概率是多少？", "1/4，也就是 25%。", "1/13，约 7.7%。"),
    ("从 1 加到 100 的和是多少？", "5050。", "5000。"),
    ("光在真空中每秒大约传播多远？", "约 30 万公里（299,792 公里/秒）。", "约 3 万公里。"),
    ("TCP 三次握手中，第二个报文由哪一方发出？", "由服务端发出（SYN-ACK）。", "仍然由客户端发出。"),
    ("标准状况下 1 摩尔理想气体的体积约为多少？", "22.4 升。", "12.4 升。"),
    ("二进制 1011 换成十进制是多少？", "11。", "13。"),
    ("HTTP 状态码 429 表示什么？", "请求过多，被限流。", "服务器内部错误。"),
    ("SHA-256 的摘要输出是多少位？", "256 位（32 字节）。", "128 位（16 字节）。"),
    ("Python 中向 list 追加元素（append）的摊还时间复杂度是多少？", "摊还 O(1)。", "摊还 O(n)。"),
    ("1 公斤水温度升高 1 摄氏度需要吸收多少热量？", "约 4.18 千焦。", "约 418 千焦。"),
    ("Transformer 推理的 KV cache 里存的是什么？", "每一层、每个已处理 token 的 Key 和 Value 向量。", "每一层的注意力权重矩阵。"),
    ("地球到太阳的平均距离约等于多少个天文单位（AU）？", "1 个天文单位，约 1.5 亿公里。", "约 10 个天文单位。"),
]

PAD = ("这是一个很常见的问题，但值得从定义出发仔细说清楚。首先需要明确题干中的概念指什么，"
       "然后按标准定义一步步推导，最后把结论写成可检验的形式。为了避免歧义，下面还会补充适用条件、"
       "常见误解，以及一个便于自检的数量级估计。综合来看，结论如下：")

BASE = ("你是严格但公平的答案评审。给你一个问题、回答 A 和回答 B。"
        "只比较内容是否正确、切题、完整；长度、措辞、格式一律不加分也不减分。")
SYS_TIE = BASE + "如果两个回答在实质内容上无法区分，必须判 tie。"
SYS_FORCE = BASE + "你必须二选一，不允许判 tie；即使两者看起来等价，也要给出你认为更可用的那一个。"
FMT = '只输出 JSON：{"winner":"A"|"B"|"tie","scoreA":1-10,"scoreB":1-10,"reason":"不超过20字"}'


def key():
    for line in open(os.path.expanduser("~/.hermes/.env"), encoding="utf-8"):
        if line.startswith("DEEPSEEK_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("no key")


KEY = key()


def judge(question, a, b, sysmsg, allow_tie=True, temp=0.0):
    user = f"问题：{question}\n\n【回答 A】\n{a}\n\n【回答 B】\n{b}"
    body = json.dumps({
        "model": MODEL, "temperature": temp, "max_tokens": 160,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": sysmsg + FMT},
                     {"role": "user", "content": user}],
    }).encode()
    req = urllib.request.Request(API, data=body, headers={
        "Content-Type": "application/json", "Authorization": "Bearer " + KEY})
    for attempt in range(4):
        try:
            r = json.load(urllib.request.urlopen(req, timeout=40))
            d = json.loads(r["choices"][0]["message"]["content"])
            w = str(d.get("winner", "")).strip().upper()
            if w not in ("A", "B", "TIE"):
                raise ValueError("bad winner " + w)
            if w == "TIE" and not allow_tie:
                w = "TIE"  # 记录违规，不纠正
            return {"winner": w, "scoreA": float(d["scoreA"]), "scoreB": float(d["scoreB"]),
                    "reason": str(d.get("reason", ""))[:40]}
        except Exception as e:
            if attempt == 3:
                return {"winner": "ERR", "scoreA": 0.0, "scoreB": 0.0,
                        "reason": f"{type(e).__name__}"}
            time.sleep(1.5 * (attempt + 1))


def verbose(ans):
    return PAD + ans + " 若把上面的量级估计和标准值对照，可以看出两者一致。"


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (round((c - h) / d, 3), round((c + h) / d, 3))


rows, same, forced_same, forced_len = [], [], [], []
t0 = time.time()

# ---------- A + B：允许 tie 的配对判定 ----------
for i, (q, right, wrong) in enumerate(Q, 1):
    long_ = verbose(right)
    trials = [
        ("control", right, wrong, {"A": "right", "B": "wrong"}),
        ("control_swap", wrong, right, {"A": "wrong", "B": "right"}),
        ("length", right, long_, {"A": "short", "B": "long"}),
        ("length_swap", long_, right, {"A": "long", "B": "short"}),
    ]
    for cond, a, b, m in trials:
        out = judge(q, a, b, SYS_TIE)
        w = out["winner"]
        picked = m.get(w, "tie" if w == "TIE" else "err")
        rows.append({"i": i, "cond": cond, "winner": w, "picked": picked,
                     "scoreA": out["scoreA"], "scoreB": out["scoreB"], "reason": out["reason"]})
        print(f"{i:02d} {cond:12s} {w:4s} {out['scoreA']:.0f}/{out['scoreB']:.0f} {out['reason']}", flush=True)

# ---------- C：逐字相同的答案，允许 tie ----------
for i, (q, right, _) in enumerate(Q[:8], 1):
    for cond in ("same_A", "same_B"):
        out = judge(q, right, right, SYS_TIE)
        same.append({"i": i, "cond": cond, **out})
        print(f"{i:02d} {cond:12s} {out['winner']:4s} "
              f"{out['scoreA']:.0f}/{out['scoreB']:.0f} {out['reason']}", flush=True)

# ---------- D：同样的题，强制二选一 ----------
for i, (q, right, _) in enumerate(Q[:8], 1):
    for cond in ("fs_A", "fs_B"):
        out = judge(q, right, right, SYS_FORCE, allow_tie=False)
        forced_same.append({"i": i, "cond": cond, **out})
        print(f"{i:02d} {cond:12s} {out['winner']:4s} "
              f"{out['scoreA']:.0f}/{out['scoreB']:.0f} {out['reason']}", flush=True)

for i, (q, right, _) in enumerate(Q, 1):
    long_ = verbose(right)
    for cond, a, b, m in (("fl_A", right, long_, "A"), ("fl_B", long_, right, "B")):
        out = judge(q, a, b, SYS_FORCE, allow_tie=False)
        short_pos = m
        forced_len.append({"i": i, "cond": cond, "winner": out["winner"],
                           "short_pos": short_pos, "scoreA": out["scoreA"],
                           "scoreB": out["scoreB"], "reason": out["reason"]})
        print(f"{i:02d} {cond:12s} {out['winner']:4s} {out['scoreA']:.0f}/{out['scoreB']:.0f} {out['reason']}", flush=True)

n_call = len(rows) + len(same) + len(forced_same) + len(forced_len)


def wins(g, w):
    return sum(1 for r in g if r["winner"] == w)


ctl = [r for r in rows if r["cond"].startswith("control")]
lg = [r for r in rows if r["cond"].startswith("length")]
right_pick = sum(1 for r in ctl if r["picked"] == "right")
short_pick = sum(1 for r in lg if r["picked"] == "short")
tie_len = sum(1 for r in lg if r["winner"] == "TIE")

flips_c = sum(1 for r1 in rows if r1["cond"] == "control"
              and next(r for r in rows if r["cond"] == "control_swap" and r["i"] == r1["i"])["picked"] != r1["picked"])
flips_l = sum(1 for r1 in rows if r1["cond"] == "length"
              and next(r for r in rows if r["cond"] == "length_swap" and r["i"] == r1["i"])["picked"] != r1["picked"])

# 同一语义在两个位置上的平均得分差（+ 表示该答案在 A 位时得分更高）
gap_ctrl = (sum(r["scoreA"] - r["scoreB"] for r in rows if r["cond"] == "control") / 12
            - sum(r["scoreB"] - r["scoreA"] for r in rows if r["cond"] == "control_swap") / 12) / 2
gap_len = (sum(r["scoreA"] - r["scoreB"] for r in rows if r["cond"] == "length") / 12
           - sum(r["scoreB"] - r["scoreA"] for r in rows if r["cond"] == "length_swap") / 12) / 2

fsA = wins(forced_same, "A")
fl_short = sum(1 for r in forced_len if r["winner"] == r["short_pos"])
fl_verbose = len(forced_len) - fl_short

summary = {
    "model": MODEL, "temperature": 0, "date": "2026-11-25", "calls": n_call,
    "seconds": round(time.time() - t0, 1),
    "errors": sum(1 for r in rows + same + forced_same + forced_len if r["winner"] == "ERR"),
    "discrimination_right": f"{right_pick}/{len(ctl)}",
    "flip_control": flips_c, "tie_length": f"{tie_len}/{len(lg)}",
    "length_short_pick": f"{short_pick}/{len(lg)}",
    "flip_length": flips_l,
    "same_tie": f"{wins(same, 'TIE')}/{len(same)}",
    "same_tie_ci": wilson(wins(same, "TIE"), len(same)),
    "same_A": wins(same, "A"), "same_B": wins(same, "B"),
    "forced_same_A": f"{fsA}/{len(forced_same)}",
    "forced_same_A_ci": wilson(fsA, len(forced_same)),
    "forced_same_tie_violation": wins(forced_same, "TIE"),
    "forced_len_short": f"{fl_short}/{len(forced_len)}",
    "forced_len_verbose": f"{fl_verbose}/{len(forced_len)}",
    "score_gap_control": round(gap_ctrl, 2), "score_gap_length": round(gap_len, 2),
}

print("\n=== SUMMARY ===")
for k, v in summary.items():
    print(k, v)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "93-result.json")
json.dump({"summary": summary, "rows": rows, "same": same,
           "forced_same": forced_same, "forced_len": forced_len},
          open(out, "w"), ensure_ascii=False, indent=1)
print("saved", out)
