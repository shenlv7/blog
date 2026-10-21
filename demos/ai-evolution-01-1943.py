#!/usr/bin/env python3
"""
1943 McCulloch–Pitts 神经元：最小复现 + 三个可证伪的结论。

论文定义（"A Logical Calculus of the Ideas Immanent in Nervous Activity",
Bull. Math. Biophys. 5(4):115–133, 1943）:
  - 输入、输出都是二值（0 / 1），全或无（all-or-none）
  - 权重只有两种：兴奋 +1、抑制 -1
  - 抑制是绝对的：只要有一个抑制输入为 1，无论兴奋输入多少，输出恒为 0
  - 阈值 θ 固定，由人设定；**没有学习规则**
"""

from itertools import product


def mcp(inputs, exc, inh, theta):
    """一个 McCulloch–Pitts 神经元。inputs 是 0/1 列表，exc/inh 是下标集合。"""
    if any(inputs[i] for i in inh):        # 抑制绝对优先
        return 0
    return 1 if sum(inputs[i] for i in exc) >= theta else 0


def truth_table(n, fn):
    rows = []
    for xs in product([0, 1], repeat=n):
        rows.append((xs, fn(xs)))
    return rows


print("=" * 62)
print("1. AND（θ=2，两个兴奋输入）")
for xs, y in truth_table(2, lambda x: mcp(x, {0, 1}, set(), 2)):
    print(f"   {xs[0]} AND {xs[1]} = {y}")

print("\n2. OR（θ=1，两个兴奋输入）")
for xs, y in truth_table(2, lambda x: mcp(x, {0, 1}, set(), 1)):
    print(f"   {xs[0]} OR  {xs[1]} = {y}")

print("\n3. A AND NOT B（A 兴奋，B 抑制）—— 抑制是绝对的")
for xs, y in truth_table(2, lambda x: mcp(x, {0}, {1}, 1)):
    print(f"   {xs[0]} AND NOT {xs[1]} = {y}")

print("\n4. XOR 能不能用单神经元实现？（穷举所有 θ）")
xor = lambda x: x[0] ^ x[1]
target = {xs: xor(xs) for xs in product([0, 1], repeat=2)}
print(f"   目标: {target}")

ok = []
for theta in range(0, 4):
    for exc in ({0, 1}, {0}, {1}, set()):
        for inh in ({0}, {1}, {0, 1}, set()):
            if exc & inh:
                continue
            got = {xs: mcp(xs, exc, inh, theta) for xs in product([0, 1], repeat=2)}
            if got == target:
                ok.append((theta, exc, inh))

print(f"   穷举 θ∈[0,3] × 兴奋集 × 抑制集，能实现 XOR 的组合数 = {len(ok)}")
print("   → 单个 MCP 神经元是线性可分器，XOR 线性不可分，结构上做不了。")
print("     （1969 年 Minsky & Papert 正是用这一点给感知机判了死缓）")

print("\n5. 它真的不会学习：权重全靠人给")
print("   mcp() 里没有任何一行会修改 exc / inh / theta —— 1943 年的模型没有更新规则。")
print("   第一次出现可学习规则要等到 1949（Hebb）和 1957（Rosenblatt 感知机）。")
