"""第64篇 demo：双下降（double descent）——随机特征回归上的真实数值实验。

模型：y = f(x) + 噪声，x ∈ R^20。用 k 个随机 ReLU 特征做线性回归：
    phi_j(x) = relu(w_j · x + b_j),  W,b 冻结不再训练
    w_head = argmin ||Phi w - y||_2  （欠定时取最小范数解）
n_train = 100 固定，k 从 2 扫到 600，观察 train/test MSE。
硬件：3 核 ARM，纯 numpy，无 GPU，全程 CPU 实跑。
"""
import numpy as np

D_IN = 20
N_TRAIN = 100
N_TEST = 5000
NOISE = 0.5
SEEDS = range(5)
K_GRID = [2, 5, 10, 20, 40, 60, 80, 90, 100, 110, 125, 150, 200, 300, 400, 600]


def true_f(x):
    # 线性项 + 二次项 + 一个三次交互项：真实函数本身有非线性结构
    return (0.8 * x[:, 0] - 0.6 * x[:, 1]
            + 0.9 * x[:, 2] * x[:, 3]
            + 0.5 * x[:, 4] ** 2
            + 0.7 * x[:, 0] * x[:, 5] ** 2)


def make_data(rng):
    Xtr = rng.normal(size=(N_TRAIN, D_IN))
    Xte = rng.normal(size=(N_TEST, D_IN))
    ytr = true_f(Xtr) + NOISE * rng.normal(size=N_TRAIN)
    yte = true_f(Xte)  # 测试集不含噪声：测的是对真实函数的逼近误差
    return Xtr, ytr, Xte, yte


def feats(X, W, b):
    return np.maximum(X @ W.T + b, 1e-3)


rows = []
for k in K_GRID:
    tr_list, te_list, nrm_list = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        Xtr, ytr, Xte, yte = make_data(rng)
        # 随机特征参数：尺度 1/sqrt(D_IN)，让每层激活量级 ≈ O(1)
        W = rng.normal(scale=1.0 / np.sqrt(D_IN), size=(k, D_IN))
        b = rng.uniform(0, 1, size=k)
        A = feats(Xtr, W, b)
        At = feats(Xte, W, b)
        # 最小范数最小二乘：k<=n 时 pinv 给出普通最小二乘，k>n 时给出 min-norm 插值解
        w = np.linalg.pinv(A) @ ytr
        tr = float(np.mean((A @ w - ytr) ** 2))
        te = float(np.mean((At @ w - yte) ** 2))
        tr_list.append(tr)
        te_list.append(te)
        nrm_list.append(float(np.linalg.norm(w)))
    rows.append((k, np.mean(tr_list), np.mean(te_list), np.std(te_list),
                 np.mean(nrm_list)))

print(f"n_train={N_TRAIN}  n_test={N_TEST}  d_in={D_IN}  label_noise_sigma={NOISE}")
print(f"测试集无噪声；test MSE = 对真实函数 f 的期望平方误差")
print(f"{'k':>5} {'train_mse':>11} {'test_mse':>11} {'test_sd':>9} {'|w|':>9}")
peak = None
for k, tr, te, sd, nw in rows:
    mark = ""
    if k == N_TRAIN:
        mark = "  <-- 插值门槛 k = n_train"
    print(f"{k:5d} {tr:11.4f} {te:11.4f} {sd:9.4f} {nw:9.3f}{mark}")

# 找实测峰值
best = max(rows, key=lambda r: r[2])
small = rows[0]
tail = rows[-1]
print(f"\n峰值：k={best[0]}  test_mse={best[2]:.4f}")
print(f"k={small[0]}   test_mse={small[2]:.4f}   （k={small[0]} 远小于 n_train，训练极度欠拟合）")
print(f"k={tail[0]}   test_mse={tail[2]:.4f}   （参数是样本的 {tail[0]/N_TRAIN:.0f} 倍）")
print(f"峰值 / 尾部 = {best[2]/tail[2]:.2f}x")

# 实验 2：峰在哪？换样本量，看峰跟着 n 走还是跟着别的走
print("\n实验 2：峰的位置是否跟着 n_train 移动（每格 8 seeds，k 扫 0.25n..3n）")
print(f"{'n_train':>8} {'峰值k':>7} {'峰值test均':>11} {'峰值test中位':>12} {'尾部k':>7} {'尾部test均':>10} {'尾部test中位':>12}")
for n in [40, 80, 160]:
    best_k, best_v, tail_v, tail_k, best_med, tail_med = None, -1, None, None, None, None
    for k in sorted(set([max(2, int(n * f)) for f in
                         (0.25, 0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0, 3.0)])):
        tes = []
        for seed in range(8):
            rng = np.random.default_rng(3000 + seed)
            Xtr = rng.normal(size=(n, D_IN))
            Xte = rng.normal(size=(2000, D_IN))
            ytr = true_f(Xtr) + NOISE * rng.normal(size=n)
            yte = true_f(Xte)
            W = rng.normal(scale=1.0 / np.sqrt(D_IN), size=(k, D_IN))
            b = rng.uniform(0, 1, size=k)
            w = np.linalg.pinv(feats(Xtr, W, b)) @ ytr
            tes.append(float(np.mean((feats(Xte, W, b) @ w - yte) ** 2)))
        v = float(np.mean(tes))
        if v > best_v:
            best_k, best_v, best_med = k, v, float(np.median(tes))
        if tail_k is None or k > tail_k:
            tail_k, tail_v, tail_med = k, v, float(np.median(tes))
    print(f"{n:8d} {best_k:7d} {best_v:11.1f} {best_med:12.2f} {tail_k:7d} {tail_v:10.4f} {tail_med:12.4f}")

# 实验 3：门槛处那个尖峰有多脆？加一点点 ridge 就塌
print("\n实验 3：k = n_train = 100，只改 ridge λ（10 seeds）")
print(f"{'lambda':>10} {'train_mse':>10} {'test_mse':>12}")
for lam in [0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]:
    trs, tes = [], []
    for seed in range(10):
        rng = np.random.default_rng(9000 + seed)
        Xtr, ytr, Xte, yte = make_data(rng)
        k = N_TRAIN
        W = rng.normal(scale=1.0 / np.sqrt(D_IN), size=(k, D_IN))
        b = rng.uniform(0, 1, size=k)
        A, At = feats(Xtr, W, b), feats(Xte, W, b)
        if lam == 0.0:
            w = np.linalg.pinv(A) @ ytr
        else:
            w = np.linalg.solve(A.T @ A + lam * np.eye(k), A.T @ ytr)
        trs.append(float(np.mean((A @ w - ytr) ** 2)))
        tes.append(float(np.mean((At @ w - yte) ** 2)))
    print(f"{lam:10.0e} {np.mean(trs):10.4f} {np.mean(tes):12.4f}")

# 门槛附近加密扫描：确认峰是否就落在插值点
fine = [80, 85, 90, 95, 98, 99, 100, 101, 102, 105, 110, 120]
print("\n门槛附近加密扫描（10 seeds）:")
print(f"{'k':>5} {'test_mse':>10}")
for k in fine:
    tes = []
    for seed in range(10):
        rng = np.random.default_rng(7000 + seed)
        Xtr, ytr, Xte, yte = make_data(rng)
        W = rng.normal(scale=1.0 / np.sqrt(D_IN), size=(k, D_IN))
        b = rng.uniform(0, 1, size=k)
        w = np.linalg.pinv(feats(Xtr, W, b)) @ ytr
        tes.append(float(np.mean((feats(Xte, W, b) @ w - yte) ** 2)))
    print(f"{k:5d} {np.mean(tes):10.4f}")
