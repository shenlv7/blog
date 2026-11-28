#!/usr/bin/env python3
"""第 96 篇：MoE 的负载均衡是 batch 的函数，不是路由器的属性。
纯 numpy 模拟（不是 GPU 实测）：256 专家 / top-8，逐 batch size 统计活跃专家比例、
负载变异系数、最热专家占比。另外算一遍 EP=8 下每 token 的 all-to-all 载荷。
"""
import json, os
import numpy as np

E, K, TRIALS = 256, 8, 400
BS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def run(bias_sigma, seed=96):
    rng = np.random.default_rng(seed)
    bias = rng.normal(0, bias_sigma, E) if bias_sigma else np.zeros(E)
    rows = []
    for B in BS:
        act, cvs, mx, shares = [], [], [], []
        for _ in range(TRIALS):
            logits = bias[None, :] + rng.normal(0, 1, (B, E))
            idx = np.argpartition(-logits, K, axis=1)[:, :K]
            cnt = np.bincount(idx.ravel(), minlength=E)
            act.append((cnt > 0).mean())
            cvs.append(cnt.std() / (cnt.mean() + 1e-12))
            mx.append(cnt.max() / (cnt.mean() + 1e-12))
            top = np.sort(cnt)[::-1][: max(1, E // 10)].sum() / cnt.sum()
            shares.append(top)
        rows.append({"batch": B, "active": float(np.mean(act)), "cv": float(np.mean(cvs)),
                     "max_over_mean": float(np.mean(mx)), "top10_share": float(np.mean(shares))})
        r = rows[-1]
        print(f"  B={B:>3}  活跃专家 {r['active']:.3f} ({r['active']*E:.1f}/{E})  "
              f"CV {r['cv']:.3f}  峰值/均值 {r['max_over_mean']:.2f}  "
              f"最热 10% 吃 {r['top10_share']:.1%}")
    return rows

print("== 均匀路由器（无偏置）")
uni = run(0.0)
print("== 有偏置的路由器（bias~N(0,0.7)，模拟训练后常见的专家热冷）")
skew = run(0.7)
print("== 偏置更强（sigma=1.2）")
hot = run(1.2)

# 理论对照：均匀 top-k 的期望覆盖率
print("\n== 解析式对照 E_act = E*(1-(1-k/E)^B)")
for B in BS:
    print(f"  B={B:>3}  理论 {E*(1-(1-K/E)**B)/E:.3f}")

# all-to-all 载荷：d_model=7168, fp8=1B, top-8
d, k, ep = 7168, 8, 8
per_tok = k * d          # bytes/token 发出去
print(f"\n== EP={ep}: 每 token 发送 {per_tok/1024:.0f} KB, 每 GPU 分到 {per_tok/1024/ep:.1f} KB")
for B in [1, 8, 64, 256]:
    print(f"  B={B:>3}  单次 all-to-all {per_tok*B/1024/1024:.2f} MB "
          f"(每 GPU {per_tok*B/1024/1024/ep:.2f} MB)")
# 专家权重读取：3*d*hidden, fp8
h = 2048
exp_w = 3 * d * h / 1024 / 1024
print(f"\n   单专家 FFN 权重 ≈ {exp_w:.0f} MB (3*{d}*{h}, fp8)")
print(f"   B=1 时每 token 触发 {K} 个专家 → 读 {K*exp_w:.0f} MB 权重，"
      f"只做 {K*2*d*h/1e9:.2f} GFLOP")
inten = K*2*d*h / (K*exp_w*1024*1024)
print(f"   → 算术强度 ≈ {inten:.2f} FLOP/byte")

json.dump({"uniform": uni, "skew": skew, "hot": hot},
          open(os.path.join(os.path.dirname(__file__), "96-moe-batch.json"), "w"),
          ensure_ascii=False, indent=1)
