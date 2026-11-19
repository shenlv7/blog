#!/usr/bin/env python3
"""87 · 模型编辑：线性联想记忆上的 ROME / MEMIT 复现

模型：W = argmin ||W K - V||^2  （K: d x N 的键，V: d x N 的值）
记忆：ŷ = W k ，在 50 个对象向量里取最近邻（cos）
编辑：改掉某条事实的对象，比较四种写法
  naive 单条：Δ = r k^T /(k^T k)                          r = v_new − W k*
  MEMIT 单条：Δ = r k^T C^{-1} /(k^T C^{-1} k)            C = K K^T / N
  naive 串行：逐条套单条公式，每次用当前 W 重算 r
  MEMIT 联合：Δ = R (Ke^T C^{-1} Ke)^{-1} Ke^T C^{-1}      一次解全部

不是 LLM 实测：d=256 的线性联想记忆，"同主语键高度相关"用来模拟真实 transformer
里同一主语各条事实的键在编辑方向上的共享结构。
"""
import numpy as np

rng = np.random.default_rng(7)

D, N_SUB, PER_SUB = 256, 40, 8     # 320 条事实
N_OBJ, N_CLUST = 50, 10            # 对象词表按 10 个簇分布，制造近邻混淆
CORR = 0.35                        # 同主语键的噪声幅度


def unit(x):
    return x / np.linalg.norm(x, axis=0, keepdims=True)


# ---------- 合成事实库 ----------
anchor = unit(rng.normal(size=(D, N_SUB)))
keys = [unit(anchor[:, s] + CORR * rng.normal(size=D) / np.sqrt(D))
        for s in range(N_SUB) for _ in range(PER_SUB)]
K = np.stack(keys, axis=1)
owner = np.repeat(np.arange(N_SUB), PER_SUB)
rel = np.tile(np.arange(PER_SUB), N_SUB)

clust = unit(rng.normal(size=(D, N_CLUST)))
objs = unit(clust[:, rng.integers(0, N_CLUST, N_OBJ)]
            + 0.55 * rng.normal(size=(D, N_OBJ)) / np.sqrt(D))
obj_of = rng.integers(0, N_OBJ, size=K.shape[1])
V = objs[:, obj_of]
N = K.shape[1]
C = K @ K.T / N


def train(K, V, ridge=1e-6):
    return V @ K.T @ np.linalg.inv(K @ K.T / K.shape[1] + ridge * np.eye(K.shape[0]))


def recall(W, Q):
    p = W @ Q
    p /= np.linalg.norm(p, axis=0, keepdims=True)
    return np.argmax(objs.T @ p, axis=0), p


def cos_to_truth(W, idx):
    _, p = recall(W, K[:, idx])
    pn = p / np.linalg.norm(p, axis=0, keepdims=True)
    return (pn * V[:, idx]).sum(axis=0)


def edit_weight(C, K_e, R):
    """min ||Δ||_C  s.t. Δ K_e = R"""
    Ci = np.linalg.inv(C)
    return R @ np.linalg.inv(K_e.T @ Ci @ K_e) @ K_e.T @ Ci


W0 = train(K, V)
pred0, _ = recall(W0, K)
base_acc = (pred0 == obj_of).mean()
print(f'基线：320 条事实 top-1 命中 {base_acc*100:.1f}%，'
      f'同主语键 cos {np.mean([K[:,i]@K[:,j] for i in range(N) for j in range(i+1,N) if owner[i]==owner[j]]):.3f}')

# ---------- 单条编辑 ----------
t = 12
new_obj = (obj_of[t] + 7) % N_OBJ
r = (objs[:, new_obj] - W0 @ K[:, t])[:, None]
mask = np.ones(N, bool); mask[t] = False
same = mask & (owner == owner[t])


def report_single(W, name):
    pr, _ = recall(W, K)
    cs = cos_to_truth(W, np.where(mask)[0])
    c0 = cos_to_truth(W0, np.where(mask)[0])
    print(f'{name:<16} 目标命中 {int(pr[t]==new_obj)}  未被编辑事实 cos {cs.mean():.4f}'
          f'（{np.mean(cs-c0):+.4f}）  top1 保留 {(pr[mask]==obj_of[mask]).mean()*100:.1f}%'
          f'  同主语 cos {cos_to_truth(W, np.where(same)[0]).mean():.4f}  ||ΔW||={np.linalg.norm(W-W0):.1f}')
    return cs


print('\n--- 单条编辑 ---')
cs_n = report_single(W0 + (r @ K[:, [t]].T) / (K[:, t] @ K[:, t]), 'naive')
cs_m = report_single(W0 + edit_weight(C, K[:, [t]], r), 'cov-weighted')

# 损伤随键重叠度变化
print('\n--- 旁路损伤 × 键重叠度 ---')
ov = K.T @ K[:, t]
Wn = W0 + (r @ K[:, [t]].T) / (K[:, t] @ K[:, t])
Wm = W0 + edit_weight(C, K[:, [t]], r)
for lo, hi in [(-0.1, .3), (.3, .6), (.6, .8), (.8, 1.01)]:
    idx = np.where(mask & (ov >= lo) & (ov < hi))[0]
    if len(idx) == 0:
        continue
    print(f'  cos(k*,k)∈[{lo:.1f},{hi:.1f}) n={len(idx):3d}  '
          f'naive Δcos {(cos_to_truth(Wn, idx)-cos_to_truth(W0, idx)).mean():+.4f}  '
          f'MEMIT Δcos {(cos_to_truth(Wm, idx)-cos_to_truth(W0, idx)).mean():+.4f}')

# ---------- 批量编辑 ----------
print('\n--- 批量编辑 20 条 ---')
B = 20
targets = rng.choice(N, size=B, replace=False)
new_objs = (obj_of[targets] + rng.integers(1, N_OBJ, size=B)) % N_OBJ
Rb = objs[:, new_objs] - W0 @ K[:, targets]


def report_batch(W, name):
    pr, _ = recall(W, K)
    keep = np.ones(N, bool); keep[targets] = False
    hit = (pr[targets] == new_objs).mean()
    untouched = (pr[keep] == obj_of[keep]).mean()
    dc, dn = (cos_to_truth(W, np.where(keep)[0])-cos_to_truth(W0, np.where(keep)[0])).mean(), np.linalg.norm(W-W0)
    print(f'{name:<18} 编辑命中 {hit*100:5.1f}%  未涉及事实 top1 {untouched*100:5.1f}%'
          f'  Δcos {dc:+.4f}  ||ΔW||={dn:7.1f}  损伤/范数 {abs(dc)/dn*1e5:5.2f}e-5')


W_seq = W0.copy()
for j, ti in enumerate(targets):
    rr = (objs[:, new_objs[j]] - W_seq @ K[:, ti])[:, None]
    W_seq = W_seq + (rr @ K[:, [ti]].T) / (K[:, ti] @ K[:, ti])
report_batch(W_seq, 'naive 串行')

W_joint_n = W0 + Rb @ np.linalg.pinv(K[:, targets])
report_batch(W_joint_n, 'naive 联合(pinv)')

W_joint = W0 + edit_weight(C, K[:, targets], Rb)
report_batch(W_joint, 'MEMIT 联合')
