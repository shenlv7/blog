"""第65篇 demo：晚期交互（late interaction / ColBERT 的 MaxSim）vs 单向量双塔。

装置（本机 CPU 实跑，numpy，无 GPU、无预训练权重）：
  - D=128 维合成嵌入空间，带各向异性：每个词向量 = 0.5*公共轴 + 0.87*自身方向，
    因此任意两词余弦约 0.25（贴近真实句向量空间"所有词都长得有点像"的现象）
  - 常见词表 300 个；专名 200 个（查询真正要找的东西）
  - 文档 32 个 token：29 个常见词 + 3 个干扰专名；相关文档里用 1 个查询专名替换 1 个干扰词
    → 目标信号只占文档 1/32 的权重，长度与词表分布全部对齐，只差这一个词
  - 查询 12 个 token：11 个常见词 + 1 个目标专名（真实提问里也只有一个词是关键）
四种打分：
  A 单向量·均值池化（双塔标准做法）  score = cos(q_mean, d_mean)
  B 单向量·逐维最大池化              d = elementwise max over tokens
  C MaxSim（晚期交互）                score = mean_t max_j q_t · d_j
  D 两段式：A 取 top-50 → C 重排
指标：R@1 / R@10 / MRR；成本：每篇存储字节、查询延迟
"""
import time

import numpy as np

D, N_WORD, N_NAME = 128, 3000, 400
N_DOC, TOK_DOC, N_QRY = 4000, 32, 150
ANISO, NOISE = 0.25, 0.02
rng = np.random.default_rng(65)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


axis = norm(rng.normal(size=D)).astype(np.float32)          # 各向异性公共轴
words = norm(ANISO * axis + (1 - ANISO ** 2) ** 0.5 * norm(rng.normal(size=(N_WORD, D))))
names = norm(ANISO * axis + (1 - ANISO ** 2) ** 0.5 * norm(rng.normal(size=(N_NAME, D))))
print(f"词-词平均余弦 = {np.mean(words @ words.T):.3f}（各向异性 {ANISO}），"
      f"词-专名平均余弦 = {np.mean(words @ names.T):.3f}")

# 所有文档共享同一组 29 个背景词（模拟同领域语料），文档之间唯一的差别是专名
bg_doc = rng.integers(0, N_WORD, 29)
bg_qry = bg_doc[:11]                                  # 查询复用其中 11 个背景词
docs = np.empty((N_DOC, TOK_DOC, D), dtype=np.float32)
doc_name = rng.integers(N_QRY, N_NAME, N_DOC)          # 每篇一个干扰专名
for qi in range(N_QRY):                                # 每条查询恰好 2 篇正例
    doc_name[rng.choice(N_DOC, 2, replace=False)] = qi
docs[:, :29] = words[bg_doc]
docs[:, 29:] = names[rng.integers(N_QRY, N_NAME, (N_DOC, 3))]
docs[:, 29] = names[doc_name]                          # 每篇都真的提一次自己的专名
docs += NOISE * rng.normal(size=docs.shape)            # 词向量抖动
docs = docs.astype(np.float32)

qrys, rel = [], []
for _ in range(N_QRY):
    target = len(qrys)
    q = np.empty((12, D), dtype=np.float32)
    q[:11] = words[bg_qry]                              # 11 个背景词
    q[11] = names[target]                               # 12 个 token 里只有 1 个是目标专名
    qrys.append((q + NOISE * rng.normal(size=q.shape)).astype(np.float32))
    rel.append(np.where(doc_name == target)[0].tolist())

doc_mean = norm(docs.mean(1))
doc_max = norm(docs.max(1))
q_mean = norm(np.stack([q.mean(0) for q in qrys]))
flat = docs.reshape(-1, D)
print(f"docs={N_DOC} tok/doc={TOK_DOC} queries={N_QRY}，"
      f"每条查询平均正例 {np.mean([len(r) for r in rel]):.2f} 篇\n")


def metrics(fn):
    t0 = time.perf_counter()
    r1 = r10 = mrr = 0.0
    for qi in range(N_QRY):
        order = np.argsort(-fn(qi))
        pos = np.empty(N_DOC, dtype=np.int32)
        pos[order] = np.arange(N_DOC)
        best = pos[rel[qi]].min()
        r1 += best == 0
        r10 += best < 10
        mrr += 1.0 / (best + 1)
    return r1 / N_QRY, r10 / N_QRY, mrr / N_QRY, (time.perf_counter() - t0) / N_QRY * 1000


def maxsim(qi):
    # 每个 query token 到全文 32 个 token 的最大相似度，再对 query token 取均值
    return (qrys[qi] @ flat.T).reshape(12, N_DOC, TOK_DOC).max(2).mean(0)


def two_stage(qi, k=50):
    cand = np.argsort(-(q_mean[qi] @ doc_mean.T))[:k]
    s = np.full(N_DOC, -1e9, dtype=np.float32)
    s[cand] = maxsim(qi)[cand]
    return s


print(f"{'方法':24s} {'R@1':>6s} {'R@10':>6s} {'MRR':>6s} {'ms/query':>9s}")
for name, fn in [
    ("A 单向量·均值池化", lambda qi: q_mean[qi] @ doc_mean.T),
    ("B 单向量·逐维max", lambda qi: q_mean[qi] @ doc_max.T),
    ("C MaxSim(晚期交互)", maxsim),
    ("D 两段式 top50→MaxSim", two_stage),
]:
    r1, r10, mrr, ms = metrics(fn)
    print(f"{name:24s} {r1:6.3f} {r10:6.3f} {mrr:6.3f} {ms:9.1f}")

cov = np.mean([len(set(np.argsort(-(q_mean[q] @ doc_mean.T))[:50]) & set(rel[q])) / len(rel[q])
               for q in range(N_QRY)])
print(f"\n两段式上限：单向量 top-50 里正例覆盖率 = {cov:.3f}"
      f"（池子里没有的文档，MaxSim 再准也捞不回来）")
print(f"存储：单向量 {D*2} B(fp16)/篇 vs 晚期交互 {TOK_DOC*D} B(int8)/篇 = {TOK_DOC} 倍；"
      f"全库 {N_DOC*D*2/1e6:.1f} MB vs {N_DOC*TOK_DOC*D/1e6:.0f} MB")

# 分数间距：相关文档的平均分 vs 每篇查询里"最高的非相关文档"平均分
for name, fn in [("A 均值池化", lambda qi: q_mean[qi] @ doc_mean.T), ("C MaxSim", maxsim)]:
    gap_hi, gap_lo = [], []
    for qi in range(40):
        s = fn(qi)
        gap_hi.append(s[rel[qi]].mean())
        mask = np.ones(N_DOC, bool)
        mask[rel[qi]] = False
        gap_lo.append(s[mask].max())
    print(f"{name}: 相关 {np.mean(gap_hi):.4f} / 最强非相关 {np.mean(gap_lo):.4f} "
          f"→ 间距 {np.mean(gap_hi)-np.mean(gap_lo):+.4f}")
