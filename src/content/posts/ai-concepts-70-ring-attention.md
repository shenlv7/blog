---
title: "把 128K 上下文切成 8 段：单卡 KV 从 40 GiB 掉到 5 GiB，输出误差 3.3e-16，环形通信只占计算的 6.7%"
description: "numpy 实跑（不是 GPU 实测）：4 段环形传递 + 在线 softmax 合并，float64 最大绝对误差 7.8e-16（相对 3.3e-16），float32 为 5.8e-8。因果剪枝后每台设备步数 1..P，P=8 时利用率 56%，首尾配对重排把 2 主机的 10/26 步拉到 9/9。成本模型（H100 989 TFLOPS、NVLink 900 GB/s、bf16、S=131072、d=128、P=8）：块长 16384 时每步传 8.0 MiB/9.3 µs、算 137 GFLOP/139 µs，通信占 6.7%；临界块长 1099 token，P=32 时升到 0.27，P=64 时 0.53；跨节点 50 GB/s 的 IB 上比值再乘 18。"
pubDate: 2026-11-02
tags: ["长上下文", "并行训练", "Attention", "RingAttention", "实测"]
difficulty: "advanced"
series: "ai-concepts"
seriesOrder: 70
slug: "ai-concepts-70-ring-attention"
---

> Llama-3 70B 在 128K 上下文下，光 K 和 V 就是 **40 GiB**，比一层权重之外的任何东西都占地方。80GB 的卡装得下权重就装不下它。
> 数据并行复制、张量并行切 hidden、流水并行切层——**没有一种并行维度在切"序列"**。想继续加长，只能把序列本身劈开。
> 劈开之后每个 query 还得看见全部 KV，于是每台设备都得把别人的 K/V 收一遍。环形传递让这件事的常驻显存保持在 O(1) 块，且传输能藏在计算底下。

**读这篇要会**：softmax 的定义与最大值减法的数值动机 / FlashAttention 的分块与在线归一化思想 / 并行训练里 DP、TP、PP 各切的是哪个张量维度 / 一点 roofline 直觉（算术强度 = FLOPs ÷ 字节）。

## 🎯 卡点：挡住长上下文的是 S²，不是参数量

把 128K 的序列喂进 Llama-3 70B，先算三笔显存。权重 bf16 是 140 GB；每层 64 个 query 头、128 维，若把注意力打分矩阵物化出来是 $64\times131072^2\times2$ 字节 ≈ **2 TiB/层**；即使按 FlashAttention 的做法永不落 HBM，每层仍要持有 $2\times8\times131072\times128\times2$ ≈ 512 MiB 的 K/V，80 层累计 **40 GiB**。前两项决定了这张卡无论怎么优化都装不下，第三项决定了 40 GiB 里没有多少能省。

现有的三种并行各管一维：数据并行复制整份权重、按 batch 切样本；张量并行把 `d_model` 和 FFN 的中间维切开，每层两次 all-reduce；流水并行按层切段，靠 micro-batch 填气泡。三者都不动序列长度。Megatron 那套常被叫作 sequence parallelism 的东西切的其实是 LayerNorm、dropout 这些非注意力激活，注意力本身仍然要看全长——名字容易误导。

注意力是唯一一个"输出第 $i$ 个 token 需要输入第 $j$ 个 token"的操作，且 $j$ 覆盖全部历史。所以序列并行必然带来一次全对全的 KV 交换。真正的问题不是"要不要通信"，而是**通信能不能被算力盖住**。

| 并行维度 | 切什么 | 通信模式 | 通信量随规模 |
|---|---|---|---|
| 数据并行 DP | batch | 梯度 all-reduce | O(参数量) |
| 张量并行 TP | hidden / FFN 中间维 | 每层 2 次 all-reduce | O(激活) |
| 流水并行 PP | 层 | 段间激活传递 | O(batch×hidden) |
| 序列并行（环形） | 序列长度 | 每步传一个 K/V 块 | O(块长×d)，与 P 无关的常驻 |

## 🔬 机制：先做块内 softmax，再沿环合并

单头注意力的定义没什么可改的：

$$o=\frac{\sum_i e^{z_i-m}v_i}{\sum_i e^{z_i-m}},\qquad z_i=\frac{q\cdot k_i}{\sqrt d}$$

关键是那个 $m$。它只影响数值稳定性，不影响数学结果——所以每台设备可以先用**自己手上那块** K 的最大值做基准，算出一组局部的（分子，分母），等下一块 K 到了再重标：

$$m'=\max(m,\ \max_j z_j),\quad \alpha=e^{m-m'},\quad l'=\alpha l+\textstyle\sum_j e^{z_j-m'},\quad a'=\alpha a+\textstyle\sum_j e^{z_j-m'}v_j$$

因为 $m'\ge m$，$\alpha\le 1$，重标只会把旧累加量按比例缩小，不会溢出。走完全部 P 步之后 $a/l$ 就是全局精确结果。第 69 篇 split-K flash decoding 用的是同一套 log-sum-exp 合并，区别在拆分发生在**单个 kernel 内**（为了占满 SM），这里发生在**设备之间**（为了装下序列），并且 KV 块是流动的而不是各设备本来就有的。

环形的具体调度：设备 $r$ 持有 $(Q_r,K_r,V_r)$，第 $t$ 步它收到的是 $(r-t)\bmod P$ 的 KV 块，算完把手上这块往下一格递。P 步之后每个 query 块都和全部 KV 块碰过一次面，每台设备的常驻显存只有自己那 1/P 块，通信缓冲只是一个块的双缓冲。

这样做有三个直接后果。第一，显存从 $O(S)$ 变成 $O(S/P)$，128K 切 8 段后单卡 K/V 从 40 GiB 降到 5 GiB。第二，**总网络流量是 $P\times$ 全长 KV**——每台设备都绕环收了一遍，等于把 K/V 全量广播了 P 次；省下来的是显存和"不需要先聚合再算"，不是带宽。第三，既然每步的传输和每步的注意力计算互不依赖，两者可以放进不同 stream 重叠，利用率取决于**单步通信时间 ÷ 单步计算时间**。

## 🏗️ 工程：因果剪枝把负载打歪，配对切回来

训练时的注意力是因果的，块 $i$ 只需要看 $0..i$ 这些 KV 块，后面的一律整块跳过——不像 FlashAttention 那样在块内做三角掩码，环形的做法是**干脆不传**。于是设备 $i$ 的步数是 $i+1$，各设备步数恰好是 $1,2,\dots,P$，平均利用率 $(P+1)/2P$：P=4 是 62%，P=8 是 56%，P=16 是 53%。队列里最慢的那台决定整体时间，所以 P 越大，闲置越多。

修法不是改算法而是改**块到设备的映射**。8 个块分给 2 台主机时，连续切是 10 步 vs 26 步；改成首尾配对 $(0,7),(3,4),(1,6),(2,5)$，四组各 9 步，barrier 前没有谁在等谁。文献里这套重排叫 striped attention（Brandon 等，2023），代价是位置编码和 mask 要跟着块的物理位置走，实现上比连续切麻烦。

通信侧的账要用块长来算，这是整篇文章唯一需要记住的公式。单步传一份 K 和一份 V，$4Bd$ 字节；单步算 $q\cdot k$ 和 $p\cdot v$，$4B^2d$ FLOPs。两个一除：

$$\frac{t_{\text{comm}}}{t_{\text{comp}}}=\frac{4Bd/\text{BW}}{4B^2d/\text{FLOPS}}=\frac{\text{FLOPS}}{B\cdot \text{BW}}$$

代入 H100 的 989 TFLOPS 与 NVLink 的 900 GB/s，临界块长 $B^*=\text{FLOPS}/\text{BW}\approx1099$ token（bf16 下正好乘以字节数 2 再除 2 抵消）。块长 16384 时比值 0.067，一条 128K 序列 8 卡切分后每步传 8.0 MiB、算 137 GFLOP，通信只占 6.7%，塞进计算底下毫无压力。切到 32 卡，块长掉到 4096，比值升到 0.27；64 卡时块长 2048，0.53——已经要用双缓冲和更激进的 overlap 才压得住。

还有两个折扣必须打。989 TFLOPS 是密集 fp16 张量核的峰值，因果注意力带 softmax、带重标、带 mask，实测效率常在 40% 上下，比值要乘 2.5。跨出单机后链路换成 400 Gb/s 的 IB，有效 50 GB/s，比值再乘 18——8 卡跨两台机器做环形，通信立刻变成瓶颈。生产的做法要么把环限制在 NVLink 域内、机间只做分块重传，要么干脆用分层结构：机内环形、机间按块流水。

## 💻 代码：4 段环跑出 3.3e-16

下面这段是完整实现的核心（本地 numpy，float64 跑 512 token / 4 段），`src=(r-t)%P` 就是环形调度，`alpha` 是重标系数：

```python
for r in range(ranks):                      # rank r 上的完整循环
    q = q_blocks[r]
    m, l, acc = np.full((B,1), -np.inf), np.zeros((B,1)), np.zeros((B,d))
    for t in range(ranks):                  # 第 t 步收到 (r-t)%ranks 的 KV 块
        src = (r - t) % ranks
        if causal and src > r:              # 整块在因果线之后：不传也不算
            continue
        k, v = k_blocks[src], v_blocks[src]
        z = q @ k.T / np.sqrt(d)
        if causal and src == r:
            z = np.where(np.tril(np.ones((B,B), dtype=bool)), z, -np.inf)
        m_new = np.maximum(m, z.max(axis=-1, keepdims=True))
        alpha = np.exp(m - m_new)           # 旧累加量的重标系数
        p = np.exp(z - m_new)
        l = l * alpha + p.sum(axis=-1, keepdims=True)
        acc = acc * alpha + p @ v
        m = m_new
out[r*B:(r+1)*B] = acc / l
```

跑出来的实际结果：

```
float64: 最大绝对误差 7.772e-16，相对误差 3.260e-16
float32: 最大绝对误差 5.817e-08，相对误差 2.337e-08
  因果剪枝后每台设备步数 [1, 2, 3, 4]（P=4，满额 4 步）
S=131072 d=128 P=8 bf16: 全量 KV 64 MiB → 单卡常驻 12 MiB
每步通信 8.0 MiB (9.3 µs) / 每步计算 137.4 GFLOP (139 µs) → 通信占 6.7%
每台设备绕环一圈收到 64 MiB，等于全长 K+V 的 1.00x
  P= 4 块长  32768 → 通信/计算 = 0.03
  P= 8 块长  16384 → 通信/计算 = 0.07
  P=16 块长   8192 → 通信/计算 = 0.13
  P=32 块长   4096 → 通信/计算 = 0.27
8 块 / 2 主机 连续切 10 vs 26；首尾配对 9 vs 9
```

float32 的 5.8e-8 不是实现错误，是 512 维点积加 5 次重标在单精度下的正常损失，和 FlashAttention 同量级。延迟那一栏是成本模型算的，不是 GPU 实测——本机没有卡，这里只验证数值等价和带宽-算力比。

完整脚本（含负载均衡算例）在仓库：[demos/70-ring-attention.py](https://github.com/shenlv7/blog/blob/main/demos/70-ring-attention.py)。

## 💡 判据：块长过千，环才划算

环形注意力的适用面比它的名气窄。它是**训练和 prefill** 的手段：那里存在一个 $S\times S$ 的打分矩阵，才有东西可切。decode 阶段每步只有一个 query token，序列维上没有并行度可拆，长上下文 decode 的瓶颈是 KV 读取带宽，那是 split-K（69 篇）和 KV 分片的战场，两者不要混用同一套论证。

落地时只需三步估算。第一，块长 $B=S/P$ 有没有过千；第二，链路是 NVLink（900 GB/s）还是 IB（50 GB/s），后者要把比值乘 18；第三，按 40% 的张量核效率把通信占比乘以 2.5，看还剩多少余量。三项都过关，环形的通信就是免费的，显存按 $1/P$ 线性下降；任何一项不过关，先把 $P$ 降下来换块长，比优化 kernel 有用得多。

回看这个系列的分工：FlashAttention（第 29 篇）解决**一个设备内**少读多少字节，split-K（第 69 篇）解决**一张卡内**多少个 SM 同时在读，环形解决**一台机器内**序列放不下——三层正交，生产里通常同时开着，每一层都在替下一层腾出空间。
