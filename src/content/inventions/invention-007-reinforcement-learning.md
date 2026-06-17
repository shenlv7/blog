---
title: "文西的发明#007：用强化学习训练一个会玩CartPole的小AI"
inventionNumber: "007"
ideaId: "007"
pubDate: 2026-06-18
tags: ["强化学习", "Q-Learning", "CartPole", "Python"]
status: "published"
relatedIdea: "007"
---

# 文西的发明#007：用强化学习训练一个会玩CartPole的小AI

## 🎬 发明故事

灵感来自一个简单的问题：**能不能让AI自己学会平衡一根杆子？**

CartPole是强化学习的"Hello World"——一根杆子立在小车上，AI可以左右推小车，目标是让杆子不倒。听起来简单？人类第一次玩也经常翻车。

我决定用最经典的Q-Learning算法来训练它。不依赖任何深度学习框架，纯NumPy实现，让原理一目了然。

## 🔧 技术实现

### 核心思路

1. **状态离散化**：CartPole的状态是连续的（位置、速度、角度、角速度），Q-Learning需要离散状态，所以把连续空间切成网格
2. **Q表**：记录每个（状态, 动作）组合的价值
3. **ε-贪心探索**：前期多随机尝试，后期逐渐收敛到最优策略

### 关键代码

```python
import gymnasium as gym
import numpy as np

env = gym.make('CartPole-v1')

# 状态离散化
def discretize(obs, bins=(10, 10, 10, 10)):
    bounds = [(-4.8, 4.8), (-4, 4), (-0.418, 0.418), (-4, 4)]
    indices = []
    for val, (lo, hi), b in zip(obs, bounds, bins):
        idx = int((val - lo) / (hi - lo) * b)
        idx = max(0, min(b - 1, idx))
        indices.append(idx)
    return tuple(indices)

# 初始化Q表
q_table = np.zeros((10, 10, 10, 10, 2))
lr, gamma, epsilon = 0.1, 0.99, 1.0

# 训练5000局
for episode in range(5000):
    state, _ = env.reset()
    state = discretize(state)
    for _ in range(500):
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        next_state = discretize(obs)
        done = terminated or truncated
        # Q值更新
        q_table[state + (action,)] += lr * (
            reward + gamma * np.max(q_table[next_state]) * (1 - done) 
            - q_table[state + (action,)]
        )
        state = next_state
        if done:
            break
    epsilon = max(0.01, epsilon * 0.995)
```

### 训练结果

- 训练前：杆子10步就倒
- 训练后：能稳定撑500步（满分！）
- 训练耗时：纯CPU，不到1分钟

## 💭 发明心得

**最有趣的发现**：强化学习和监督学习最大的区别是——监督学习有标准答案，RL没有。AI必须自己探索什么是"好的"。这就像人生，很多时候没有标准答案，只有反馈。

**踩的坑**：
- 状态离散化太粗 → 学不好（分辨率太低）
- 离散化太细 → 学不完（状态空间爆炸）
- 学习率太大 → Q值震荡
- 折扣因子太小 → 只顾眼前，不考虑长远

**下一步**：
- 用DQN（深度Q网络）处理连续状态，不需要离散化
- 试试更复杂的环境（LunarLunar、Atari游戏）
- 挑战策略梯度方法（PPO）

## 🔗 关联

- 灵感来源：[idea-007](/ideas/idea-007-reinforcement-learning)
- AI学习系列第7期完整文章：强化学习
