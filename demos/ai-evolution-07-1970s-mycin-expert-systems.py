#!/usr/bin/env python3
"""MYCIN 式反向链推理器的最小实现（纯标准库，约 180 行）。

结构对齐 1976 年 Shortliffe 的 MYCIN：
  - 知识 = 一批 IF-THEN 规则，规则带 certainty factor (CF)，取值 [-1, 1]
  - 推理 = 从目标反向链，只问推理链上需要的属性
  - 不确定 = 前提取最小值，结论按 CF 相乘，多条证据按 MYCIN 组合函数叠加
  - 结论可以是多值的（每种药各自一个 CF），不是单值分类

不是 MYCIN 的移植：规则集是简化的示意（原系统约 450 条，Lisp 写在 PDP-10 上，
代码未公开）。这里只复刻推理流程与 CF 算术。
"""
import random
from itertools import permutations

# ---------------------------------------------------------------- 规则库
# (规则名, [(属性, 取值), ...], (结论属性, 结论值), 规则强度, 出处)
RULES = [
    ("R001", [("gram", "neg"), ("morph", "rod"), ("air", "aerobic")], ("identity", "e.coli"), 0.6, "肠杆菌科"),
    ("R002", [("gram", "neg"), ("morph", "rod"), ("air", "anaerobic")], ("identity", "bacteroides"), 0.6, "厌氧菌"),
    ("R003", [("gram", "neg"), ("morph", "rod"), ("air", "aerobic"), ("site", "urine")], ("identity", "e.coli"), 0.5, "尿路感染"),
    ("R004", [("gram", "pos"), ("morph", "coccus"), ("arrange", "chains")], ("identity", "streptococcus"), 0.7, "链状排列"),
    ("R005", [("gram", "pos"), ("morph", "coccus"), ("arrange", "clusters")], ("identity", "staphylococcus"), 0.7, "葡萄状排列"),
    ("R006", [("gram", "pos"), ("morph", "rod"), ("air", "aerobic")], ("identity", "listeria"), 0.4, "李斯特菌"),
    ("R007", [("site", "csf"), ("patient", "child")], ("identity", "hemophilus"), 0.5, "儿童脑膜炎常见菌"),
    ("R010", [("identity", "e.coli")], ("drug:gentamicin", "yes"), 0.6, "氨基糖苷类覆盖"),
    ("R011", [("identity", "bacteroides")], ("drug:metronidazole", "yes"), 0.7, "厌氧菌首选"),
    ("R012", [("identity", "staphylococcus")], ("drug:nafcillin", "yes"), 0.7, "抗葡萄球菌青霉素"),
    ("R013", [("identity", "streptococcus"), ("site", "csf")], ("drug:penicillin-g", "yes"), 0.6, "肺炎链球菌脑膜炎"),
    ("R014", [("identity", "listeria")], ("drug:ampicillin", "yes"), 0.5, "李斯特菌首选"),
    ("R015", [("drug:gentamicin", "yes"), ("host", "compromised")], ("drug:ticarcillin", "yes"), 0.5, "粒细胞缺乏时加 β-内酰胺"),
    ("R016", [("identity", "e.coli"), ("site", "csf")], ("drug:cefotaxime", "yes"), 0.5, "三代头孢透过血脑屏障"),
]

DRUGS = ["gentamicin", "metronidazole", "nafcillin", "penicillin-g", "ampicillin", "ticarcillin", "cefotaxime"]


# ------------------------------------------------------- MYCIN 的 CF 运算
def combine(x, y):
    """MYCIN 的证据组合函数，返回值同样落在 [-1, 1]。"""
    if x >= 0 and y >= 0:
        return x + y * (1 - x)
    if x < 0 and y < 0:
        return x + y * (1 + x)
    return (x + y) / (1 - min(abs(x), abs(y)))


def combine_all(cfs):
    acc = 0.0
    for c in cfs:
        acc = combine(acc, c)
    return acc


# ------------------------------------------------------------------ 推理机
class Engine:
    def __init__(self, rules, chart):
        self.rules = rules
        self.chart = chart        # 病历里能查到的原始事实，相当于被问的医生
        self.memo = {}
        self.asked = []           # 按顺序记录问过的属性
        self.trace = []

    def ask(self, attr):
        if attr not in self.asked:
            self.asked.append(attr)
        return self.chart.get(attr, (None, 0.0))

    def cf_of(self, attr):
        """反向链核心：返回属性 attr 的最可能取值与 CF。"""
        if attr in self.memo:
            return self.memo[attr]

        if attr in self.chart:                     # 可观测属性直接问，不推理
            self.memo[attr] = self.ask(attr)
            return self.memo[attr]

        self.memo[attr] = (None, 0.0)              # 占位，防止规则互指成环
        support = {}                               # 候选取值 -> [各规则贡献的 cf]
        for name, premise, concl, rcf, src in self.rules:
            if concl[0] != attr:
                continue
            pcf, ok = 1.0, True
            for p_attr, p_val in premise:
                val, cf = self.cf_of(p_attr)
                if val != p_val:
                    ok = False
                    break
                pcf = min(pcf, cf)                 # 合取取最小值（MYCIN 的 AND）
            if not ok:
                continue
            c = pcf * rcf                          # 前提强度 × 规则强度
            if abs(c) < 1e-9:
                continue
            support.setdefault(concl[1], []).append((name, c, pcf))
            self.trace.append((name, concl[1], pcf, rcf, c))

        best_val, best_cf = None, 0.0
        for val, items in support.items():
            total = combine_all([c for _, c, _ in items])
            if abs(total) > abs(best_cf):
                best_val, best_cf = val, total
        self.memo[attr] = (best_val, best_cf)
        return self.memo[attr]


CASES = {
    "A · 血培养：革兰阴性杆菌，需氧": {
        "gram": ("neg", 1.0), "morph": ("rod", 1.0), "air": ("aerobic", 1.0),
        "site": ("blood", 1.0), "host": ("normal", 1.0), "patient": ("adult", 1.0),
    },
    "B · 儿童脑膜炎，脑脊液涂片": {
        "gram": ("neg", 1.0), "morph": ("rod", 1.0), "air": ("aerobic", 1.0),
        "site": ("csf", 1.0), "host": ("compromised", 1.0), "patient": ("child", 1.0),
    },
    "C · 尿培养：革兰阴性杆菌": {
        "gram": ("neg", 1.0), "morph": ("rod", 1.0), "air": ("aerobic", 1.0),
        "site": ("urine", 1.0), "host": ("normal", 1.0), "patient": ("adult", 1.0),
    },
    "D · 革兰阳性杆菌，需氧": {
        "gram": ("pos", 1.0), "morph": ("rod", 1.0), "air": ("aerobic", 1.0),
        "site": ("blood", 1.0), "host": ("normal", 1.0), "patient": ("adult", 1.0),
    },
}


def consult(chart):
    eng = Engine(RULES, chart)
    org, org_cf = eng.cf_of("identity")
    drugs = []
    for d in DRUGS:
        v, cf = eng.cf_of(f"drug:{d}")
        if v == "yes" and cf > 0:
            drugs.append((d, cf))
    drugs.sort(key=lambda t: -t[1])
    return org, org_cf, drugs, eng


# --------------------------------------------------------------- 实验一：四个病例
print("=" * 76)
print("实验一  反向链只问需要的属性；结论是多值的（每种药各自一个 CF）")
print("=" * 76)
for title, chart in CASES.items():
    org, org_cf, drugs, eng = consult(chart)
    print(f"\n{title}")
    print(f"  问了 {len(eng.asked)} 条：{', '.join(eng.asked)}")
    print(f"  菌种 {org}  CF={org_cf:+.3f}")
    for d, cf in drugs:
        print(f"  建议 {d:<12} CF={cf:+.3f}")
    for name, concl, pcf, rcf, c in eng.trace:
        if name in ("R001", "R003", "R007", "R010", "R015", "R016"):
            print(f"    {name} 前提 {pcf:.3f} × 规则 {rcf:.2f} = {c:+.3f} -> {concl}")

# --------------------------------------------- 实验二：三条组合规则的手算核对
print("\n" + "=" * 76)
print("实验二  CF 组合函数的三个分支")
print("=" * 76)
print(f"  同号为正  combine(0.6, 0.5) = 0.6 + 0.5×(1-0.6) = {combine(0.6, 0.5):+.4f}")
print(f"  异号      combine(0.6,-0.3) = (0.6-0.3)/(1-0.3)  = {combine(0.6, -0.3):+.4f}")
print(f"  同为负    combine(-0.6,-0.3)= -0.6-0.3×(1-0.6)   = {combine(-0.6, -0.3):+.4f}")
print(f"  极端       combine(0.9, 0.9) = {combine(0.9, 0.9):+.4f}（永远到不了 1.0）")

# ------------------------------------------- 实验三：弱证据堆叠会不会饱和
print("\n" + "=" * 76)
print("实验三  每条 CF=0.2 的弱规则，堆到第几条冲过 0.9")
print("=" * 76)
acc, first90 = 0.0, None
for i in range(1, 21):
    acc = combine(acc, 0.2)
    if acc > 0.9 and first90 is None:
        first90 = i
    if i <= 4 or i % 4 == 0:
        print(f"    n={i:2d}   CF={acc:.4f}")
print(f"  第 {first90} 条越过 0.90；n=20 时 {acc:.4f}")
print(f"  闭式解 1-0.8^n 与迭代值一致：n=11 -> {1 - 0.8 ** 11:.4f}")

# ------------------------------------------- 实验四：合并顺序敏感性
print("\n" + "=" * 76)
print("实验四  合并顺序：{0.6,-0.3,0.5} 全部六种排列")
print("=" * 76)
group = {}
for p in permutations([0.6, -0.3, 0.5]):
    group[p] = combine_all(p)
for p, v in group.items():
    print("   " + " -> ".join(f"{x:+.1f}" for x in p) + f"   CF={v:+.6f}")
print(f"  极差 {max(group.values()) - min(group.values()):.2e}")

random.seed(7)
spread, worst = 0.0, None
for _ in range(100000):
    xs = [random.uniform(-1, 1) for _ in range(3)]
    rs = [combine_all(p) for p in permutations(xs)]
    s = max(rs) - min(rs)
    if s > spread:
        spread, worst = s, xs
print(f"  10 万组随机三元组：最大极差 {spread:.3e}")
print(f"  最坏一组 {[round(x,3) for x in worst]}，结果区间宽度 {spread:.3e}")

# ------------------------------------------- 实验五：规则库无覆盖
print("\n" + "=" * 76)
print("实验五  库里没有的形态：真菌血症")
print("=" * 76)
org, org_cf, drugs, eng = consult({"gram": ("pos", 1.0), "morph": ("yeast", 1.0),
                                   "air": ("aerobic", 1.0), "site": ("blood", 1.0),
                                   "host": ("compromised", 1.0), "patient": ("adult", 1.0)})
print(f"  identity -> {org}, CF={org_cf:+.3f}；建议药物 {len(drugs)} 个；触发的规则 {len(eng.trace)} 条")
covered = sorted({p[1] for r in RULES for p in r[1] if p[0] == "morph"})
print(f"  规则库里 morph 只覆盖 {covered}，yeast 没有对应规则，系统只能返回空")
