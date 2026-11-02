#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ai-evolution-04：启发式的唯一硬指标——省掉多少无用枚举
8 数码，同一局面、同一目标、都取最优解，只比「展开多少状态」：
  A. 广度优先          不看目标
  B. A*，h = 错位块数   弱启发式
  C. A*，h = 曼哈顿距离 更强的同一个启发式
本 demo 不移植 1956 年的程序（Logic Theorist 的搜索空间是公式不是拼图），
只把当年那条直觉——用评价函数把搜索引向目标——做成可精确计数的量。
用法: /usr/bin/python3 ai-evolution-04-1956-dartmouth.py
"""

import heapq
import time

GOAL = (1, 2, 3, 4, 5, 6, 7, 8, 0)          # 0 = 空格
MOVES = {'U': (-3, 0), 'D': (3, 2), 'L': (-1, 0), 'R': (1, 2)}


def neighbors(s):
    z = s.index(0)
    r, c = divmod(z, 3)
    out = []
    for name, (d, _) in MOVES.items():
        if name == 'U' and r == 0:
            continue
        if name == 'D' and r == 2:
            continue
        if name == 'L' and c == 0:
            continue
        if name == 'R' and c == 2:
            continue
        t = list(s)
        t[z], t[z + d] = t[z + d], t[z]
        out.append(tuple(t))
    return out


def misplaced(s):
    return sum(1 for i in range(9) if s[i] and s[i] != GOAL[i])


def manhattan(s):
    d = 0
    for i in range(9):
        v = s[i]
        if v:
            d += abs(i // 3 - (v - 1) // 3) + abs(i % 3 - (v - 1) % 3)
    return d


def bfs(start):
    from collections import deque
    q = deque([(start, 0)])
    seen = {start}
    expanded = 0
    while q:
        s, g = q.popleft()
        expanded += 1
        if s == GOAL:
            return g, expanded
        for t in neighbors(s):
            if t not in seen:
                seen.add(t)
                q.append((t, g + 1))
    return None, expanded


def astar(start, h):
    open_set = [(h(start), 0, start)]
    best = {start: 0}
    expanded = 0
    while open_set:
        _, g, s = heapq.heappop(open_set)
        expanded += 1
        if s == GOAL:
            return g, expanded
        for t in neighbors(s):
            ng = g + 1
            if ng < best.get(t, 10 ** 9):
                best[t] = ng
                heapq.heappush(open_set, (ng + h(t), ng, t))
    return None, expanded


def board(s):
    return ' / '.join(''.join('.' if v == 0 else str(v) for v in s[i:i + 3]) for i in range(0, 9, 3))


if __name__ == '__main__':
    START = (7, 2, 4, 5, 0, 6, 8, 3, 1)     # 最优解 20 步
    print('初始 =', board(START))
    print('目标 =', board(GOAL))
    print('h(错位块数) =', misplaced(START), '  h(曼哈顿) =', manhattan(START))
    print()
    rows = []
    for name, fn in (('A. 广度优先        ', bfs),
                     ('B. A*  h=错位块数  ', lambda s: astar(s, misplaced)),
                     ('C. A*  h=曼哈顿距离', lambda s: astar(s, manhattan))):
        t0 = time.time()
        steps, expanded = fn(START)
        rows.append((name, steps, expanded))
        print(f'{name}  最优解={steps} 步  展开状态={expanded:>6}  耗时={(time.time()-t0)*1000:.0f}ms')
    print()
    base = rows[0][2]
    for name, steps, expanded in rows[1:]:
        print(f'{name}  展开量 = 广度优先的 1/{base / expanded:.1f}')
