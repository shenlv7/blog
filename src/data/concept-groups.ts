// 概念卡片的主题分组：与 /posts/ai-concepts-study-path/ 索引页保持同一套 A–H 分组。
// 新增卡片：把编号加进对应组；忘了加也不会从概念页消失 —— 会落到「新近发布 · 待归类」。
export interface ConceptGroup {
  key: string;
  name: string;
  desc: string;
  color: string;
  nums: number[];
}

export const CONCEPT_GROUPS: ConceptGroup[] = [
  {
    key: 'A',
    name: '入门地基',
    desc: 'Prompt / Token / Embedding / 温度的直觉，都能在半小时内读完',
    color: '#3b82f6',
    nums: [1, 2, 3, 4, 5, 16, 74],
  },
  {
    key: 'B',
    name: '应用层：把模型变成产品',
    desc: '检索、工具调用、结构化输出、上下文管理 —— 日常工程收益最大',
    color: '#06b6d4',
    nums: [6, 9, 10, 11, 12, 13, 14, 15, 17, 61, 62],
  },
  {
    key: 'C',
    name: '架构原理：它为什么能工作',
    desc: 'Attention / Transformer / 位置编码 / 归一化统计量取自哪里',
    color: '#8b5cf6',
    nums: [7, 8, 26, 77, 82, 89],
  },
  {
    key: 'D',
    name: '推理效率：省显存、省时间',
    desc: '量化、调度、剪枝、专家并行 —— 全系列最硬的一组',
    color: '#f59e0b',
    nums: [20, 21, 22, 23, 29, 30, 31, 32, 34, 40, 41, 42, 47, 56, 58, 59, 69, 72, 85, 91, 96],
  },
  {
    key: 'E',
    name: '训练与对齐：怎么改模型',
    desc: '数据去重、微调、蒸馏、偏好对齐、分布式、优化器与调度',
    color: '#10b981',
    nums: [24, 25, 27, 28, 35, 48, 49, 53, 63, 67, 70, 71, 73, 75, 76, 78, 79, 80, 81, 83, 87, 88, 94],
  },
  {
    key: 'F',
    name: '检索与调度：系统层',
    desc: '向量检索、重排、缓存、路由、分块边界',
    color: '#6366f1',
    nums: [36, 37, 38, 39, 55, 60, 65, 68, 86, 90, 92, 95],
  },
  {
    key: 'G',
    name: '可解释与前沿机制',
    desc: '注意力汇、提示压缩、推理时计算、SSM、长上下文外推、SAE',
    color: '#ec4899',
    nums: [43, 44, 45, 46, 50, 51, 54, 64],
  },
  {
    key: 'H',
    name: '评估与安全',
    desc: '幻觉、提示注入、评测方法、水印、置信度标定',
    color: '#ef4444',
    nums: [18, 19, 33, 52, 57, 66, 84, 93],
  },
];
