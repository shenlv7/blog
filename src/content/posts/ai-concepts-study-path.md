---
title: "📚 AI 核心概念 · 学习路径（01–56 分组索引）"
description: "56 篇概念卡片没有目录。按主题分 8 组，标出前置与难度，给三条读法：实用派 / 工程派 / 研究派。"
pubDate: 2026-10-19
tags: ["索引", "学习路径", "ai-concepts"]
difficulty: "beginner"
series: "ai-concepts"
slug: "ai-concepts-study-path"
---

# AI 核心概念 · 学习路径

## 为什么有这一页

56 篇卡片是按天写下来的，编号顺序 ≈ 写作顺序，**不是学习顺序**。从首页点进来，看不出从哪篇开始、哪些篇需要前置知识。

这一页做三件事：**按主题分组、标出前置、给三条读法**。

读者画像是**会写代码的技术读者**（会 Python，知道 attention 大概在干什么）。想从零学 AI，从 A 组前三篇开始；只想把 AI 用好，可以跳过 D、E 两组 —— 它们讲的是模型内部和基础设施。

## 三条推荐路线

| 路线 | 给谁 | 顺序 | 篇数 |
|---|---|---|---|
| 🛠 实用派 | 产品 / 运营 / 独立开发者，想把 AI 用好 | A → B → F → H | 24 篇 |
| ⚙️ 工程派 | 做推理部署、要压成本 | A → C → D → F → H | 32 篇 |
| 🔬 研究派 | 读论文、做机制实验 | A → C → D → G → E | 38 篇 |

三条路线共用 A 组。C 组只有 3 篇，却是全部后续内容的地基 —— **跳过它，后面的名词会一直悬着**。

## 分组索引

### A · 入门地基（6 篇）

先把 Prompt / Token / Embedding / 温度的直觉建立起来。这组都能在半小时内读完。

- **前置**：无（会 Python、用过 ChatGPT 即可）
- **读完能回答**：为什么同一个模型别人用得比你好；为什么「字数」和 token 不是一回事。

- [01 · Prompt：为什么同一模型，别人的效果是你的十倍？](/blog/posts/ai-concepts-01-prompt/) · *入门*
- [02 · Token：AI眼中的文字长什么样？](/blog/posts/ai-concepts-02-token/) · *入门*
- [03 · LLM：大语言模型到底在做什么？](/blog/posts/ai-concepts-03-llm/) · *入门*
- [04 · Embedding：如何把文字变成数字？](/blog/posts/ai-concepts-04-embedding/) · *入门*
- [05 · Temperature：AI的创意旋钮怎么调？](/blog/posts/ai-concepts-05-temperature/) · *入门*
- [16 · 流式输出 Streaming：AI为什么一个字一个字蹦出来？](/blog/posts/ai-concepts-16-streaming/) · *入门*

### B · 应用层：把模型变成产品（9 篇）

不碰模型内部，只谈怎么用它：检索、工具调用、结构化输出、上下文管理。日常工程收益最大的一组。

- **前置**：A 组（会写 prompt、调过 API）
- **读完能回答**：RAG 什么时候救不了幻觉；工具调用与 MCP 的真实边界在哪。

- [06 · RAG：让AI不再胡说八道的外挂知识库](/blog/posts/ai-concepts-06-rag/) · *进阶*
- [09 · Fine-tuning：花小钱办大事的模型定制](/blog/posts/ai-concepts-09-finetuning/) · *进阶*
- [10 · Agent：让AI自己动手干活](/blog/posts/ai-concepts-10-agent/) · *进阶*
- [11 · Chain of Thought 思维链——让AI学会「慢思考」](/blog/posts/ai-concepts-11-chain-of-thought/) · *进阶*
- [12 · Function Calling：让AI学会「打电话」调用外部工具](/blog/posts/ai-concepts-12-function-calling/) · *硬*
- [13 · MCP：给AI装上「万能插头」](/blog/posts/ai-concepts-13-mcp/) · *硬*
- [14 · Context Window：AI的「短期记忆」极限](/blog/posts/ai-concepts-14-context-window/) · *进阶*
- [15 · Tokenization：AI是怎么「读懂」人类文字的？](/blog/posts/ai-concepts-15-tokenization/) · *进阶*
- [17 · 结构化输出：怎么让AI的回答变成程序能读的数据？](/blog/posts/ai-concepts-17-structured-output/) · *入门*

### C · 架构原理：它为什么能工作（3 篇）

Attention / Transformer / 位置编码。读 D、E、G 之前必须先过这组，否则后面全是名词。

- **前置**：A 组 + 线性代数（向量、矩阵乘）
- **读完能回答**：注意力在算什么；为什么位置需要编码进向量。

- [07 · Attention：Transformer的灵魂机制](/blog/posts/ai-concepts-07-attention/) · *硬*
- [08 · Transformer：改变一切的架构](/blog/posts/ai-concepts-08-transformer/) · *硬*
- [26 · 旋转位置编码 RoPE：为什么转一下向量，比加一个位置更管用](/blog/posts/ai-concepts-26-rope/) · *进阶*

### D · 推理效率：省显存、省时间（14 篇）

全系列最硬的一组。KV cache 是贯穿线：量化、GQA、PagedAttention、prefix caching、P/D 分离、MLA、KV 量化都在围着它转。

- **前置**：C 组（必须）
- **读完能回答**：一个 token 的 KV 值多少显存；为什么 decode 是带宽瓶颈而不是算力瓶颈。

- [20 · 专家混合 MoE：671B 参数只算 37B——我把 4 个专家塞进 numpy，看它们怎么分工](/blog/posts/ai-concepts-20-moe/) · *进阶*
- [21 · 量化 Quantization：把 16 位压成 4 位，模型凭什么还能用](/blog/posts/ai-concepts-21-quantization/) · *进阶*
- [22 · KV 缓存 KV Cache：一个 token 值多少显存，GQA/MQA/MLA 到底在省什么](/blog/posts/ai-concepts-22-kv-cache/) · *进阶*
- [23 · 投机解码 Speculative Decoding：小模型抢答、大模型判卷，账到底怎么算](/blog/posts/ai-concepts-23-speculative-decoding/) · *进阶*
- [29 · FlashAttention 分块注意力：拆成块算逐元素误差仍是 2.6e-16，显存从 128MB 掉到 96KB；可少写一行 rescale 校正，输出直接错掉 95.6%](/blog/posts/ai-concepts-29-flash-attention/) · *进阶*
- [30 · GQA 分组查询注意力：KV cache 从 64 GiB 砍到 16 GiB，把 K/V 头复制给整组 q 头逐元素误差精确为 0；均值池化转换相对误差 94.1%](/blog/posts/ai-concepts-30-grouped-query-attention/) · *进阶*
- [31 · 连续批处理 + PagedAttention：128 条请求 makespan 从 6093 步降到 2272 步（2.68x），KV cache 内部碎片从 60.9% 压到 0.81%，并发数从 2 条提到 6 条](/blog/posts/ai-concepts-31-paged-attention-and-continuous-batching/) · *进阶*
- [32 · 前缀缓存：共享的 2048 token 系统提示词把 prefill 从 599,534 token 砍到 35,182（省 94.1%），但把时间戳写在提示词开头就归零](/blog/posts/ai-concepts-32-prefix-caching/) · *进阶*
- [34 · 混合精度：FP16 的动态范围只有 40 个 2 的幂，一个 8 层网络的梯度就能把它吃光；loss scale 的可用区间只剩 2^16..2^21 这六个比特宽](/blog/posts/ai-concepts-34-mixed-precision/) · *进阶*
- [40 · 分块预填充：最大 ITL 恰好等于「固定开销 + chunk」——尾延迟是一张可以按 chunk 定价的价目表](/blog/posts/ai-concepts-40-chunked-prefill/) · *进阶*
- [41 · P/D 分离：把 prefill 从解码机器上搬走，ITL 从 430 掉到 21——账单换成了两个数字](/blog/posts/ai-concepts-41-pd-disaggregation/) · *进阶*
- [42 · 解码策略：greedy / beam / top-k / top-p —— 同一批 logits，五种花法，账都在细节里](/blog/posts/ai-concepts-42-decoding-strategies/) · *进阶*
- [47 · Multi-head Latent Attention（MLA）：把 KV 压成一个潜向量，代价是你得先学会那个子空间](/blog/posts/ai-concepts-47-multi-head-latent-attention/) · *硬*
- [56 · KV Cache 量化：K 按通道、V 按 token，这个不对称值 3.7 倍](/blog/posts/ai-concepts-56-kv-cache-quantization/) · *硬*

### E · 训练与对齐：怎么改模型（8 篇）

微调、蒸馏、偏好对齐、分布式训练，以及「学了新的忘了旧的」。

- **前置**：C 组
- **读完能回答**：LoRA 省的是什么；DPO 为什么可能让正确答案变少。

- [24 · LoRA 低秩适配：冻结大模型，只训一个薄薄的增量](/blog/posts/ai-concepts-24-lora/) · *进阶*
- [25 · 知识蒸馏：小模型怎么学到老师没说出口的判断](/blog/posts/ai-concepts-25-knowledge-distillation/) · *进阶*
- [27 · 对齐 RLHF 与 DPO：偏好差越训越大，为什么模型的正确答案却掉到只剩 10%](/blog/posts/ai-concepts-27-rlhf-dpo/) · *进阶*
- [28 · GRPO 组相对策略优化：减均值把梯度方差砍到一半，可模型答对的那 27% 的题上优势恒等于 0](/blog/posts/ai-concepts-28-grpo/) · *进阶*
- [35 · 分布式训练：三种并行各切一刀，ZeRO 切的是第四个东西——每张卡上那份完整复制品](/blog/posts/ai-concepts-35-distributed-training/) · *进阶*
- [48 · Model Merging（模型合并）：把两个微调模型加起来，你拿不到两个专家](/blog/posts/ai-concepts-48-model-merging/) · *硬*
- [49 · LoRA 合并：低秩的 delta 没有让合并变划算，只让它变便宜](/blog/posts/ai-concepts-49-lora-merging/) · *硬*
- [53 · 灾难性遗忘：先学 A 再学 B，A 的损失从 0.0002 涨到 5.79](/blog/posts/ai-concepts-53-catastrophic-forgetting/) · *硬*

### F · 检索与调度：系统层（5 篇）

向量检索、重排、缓存、路由。B 组讲「怎么用」，这组讲「怎么撑住量」。

- **前置**：B 组
- **读完能回答**：ANN 的召回上限被什么写死；语义缓存的命中率换来了什么代价。

- [36 · 向量检索与 ANN：IVF 的召回上限被探针覆盖率写死，PQ 的 21 倍压缩得用一层重排赎回](/blog/posts/ai-concepts-36-vector-search-ann/) · *进阶*
- [37 · 重排序：重排后的 R@1 被池内命中率逐格锁死，而 LLM 当重排器把正确文档从第 1 位压到了第 5 位](/blog/posts/ai-concepts-37-reranking/) · *进阶*
- [38 · 语义缓存：命中率从 33% 提到 88%，代价是每 5 条命中 1 条错答——错答按 12 倍计价时，最优阈值停在哪](/blog/posts/ai-concepts-38-semantic-caching/) · *进阶*
- [39 · 模型路由：思考模式在易题上贵 8 倍、在难题上只贵 1.4 倍——而那 25% 的差价卡在分类器的漏判率上](/blog/posts/ai-concepts-39-model-routing/) · *进阶*
- [55 · 混合检索：把 BM25 和一个瞎猜的检索器融合，nDCG 从 0.72 掉到 0.20](/blog/posts/ai-concepts-55-hybrid-retrieval/) · *硬*

### G · 可解释与前沿机制（7 篇）

注意力汇、提示压缩、推理时计算、状态空间模型、长上下文外推、叠加态与 SAE、激活引导。偏研究，但每篇都有可复现的最小实验。

- **前置**：C + D 组
- **读完能回答**：模型内部到底有没有「可读的方向」；哪些前沿机制的代价被宣传掩盖了。

- [43 · Attention Sink（注意力汇）：那 4 个没人读的 token，凭什么拿走一半的注意力](/blog/posts/ai-concepts-43-attention-sink/) · *硬*
- [44 · Prompt Compression（提示压缩）：你以为在省钱，它却在 16 倍处开始编数字](/blog/posts/ai-concepts-44-prompt-compression/) · *硬*
- [45 · Test-Time Compute（推理时计算扩展）：7 倍算力买回 +0.10，而天花板被一次温度 0 的调用钉死](/blog/posts/ai-concepts-45-test-time-compute/) · *硬*
- [46 · State Space Models（状态空间模型）：推理状态是常数，代价是你的记忆有上限](/blog/posts/ai-concepts-46-state-space-models/) · *硬*
- [50 · 长上下文外推：PI 就是把位置轴除以 8，其余手艺都在修补这个除法的副作用](/blog/posts/ai-concepts-50-context-extension/) · *硬*
- [51 · 叠加态与稀疏自编码器：5 个特征塞进 2 维，整齐长成了正五边形](/blog/posts/ai-concepts-51-superposition-sae/) · *硬*
- [54 · 激活引导：同一个方向向量，引导强度一样，改判率 1.37% 对 13.40%](/blog/posts/ai-concepts-54-activation-steering/) · *硬*

### H · 评估与安全（4 篇）

幻觉、提示注入、评测方法、生成式水印。做产品上线前该读的一组。

- **前置**：B 组
- **读完能回答**：为什么「我测了 22 次没打进去」反而是危险信号；pass@k 的常见错法偏了多少。

- [18 · 幻觉：AI为什么会一本正经地编出一篇不存在的论文？](/blog/posts/ai-concepts-18-hallucination/) · *入门*
- [19 · 提示注入：我拿 22 次真实攻击打自己，一次都没打进去——这才是最危险的地方](/blog/posts/ai-concepts-19-prompt-injection/) · *入门*
- [33 · 模型评测：pass@k 的无偏公式 vs 一个能偏出 9 个百分点的常见错法；LLM 裁判始终排在中间的那份答案只漂 0.5 分，挪到第一位的半对答案涨了 1.38 分](/blog/posts/ai-concepts-33-llm-evaluation/) · *进阶*
- [52 · 生成式文本水印：把词表随机涂绿一半，z 分数就自己长出来了](/blog/posts/ai-concepts-52-text-watermark/) · *硬*

## 怎么追更

每天 09:00 一篇，编号连续（当前到 56，下一篇 57）。想顺藤摸瓜就读相邻编号 —— 同一天写的东西往往前后脚。

每篇的写法是一致的：**结论先写在标题里 → 一个可复现的最小实验 → 明确写出「这篇不能证明什么」**。所有实验都在本机跑，不用 GPU、不调云 API，你照着代码能复现出同一批数字。

> 如果你走到某一篇发现「看不懂了」，那不是你的问题 —— 多数情况是前置那篇没读。按上面的组往回退一两篇，通常就通了。
