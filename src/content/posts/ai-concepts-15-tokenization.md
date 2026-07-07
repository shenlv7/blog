---
title: "15 | Tokenization：AI是怎么「读懂」人类文字的？"
description: "深度解析Tokenization技术，揭秘大模型如何将文字转化为数字，从「字符」到「向量」的神奇旅程"
pubDate: 2026-07-09
tags: ["AI核心概念", "Tokenization", "分词", "BPE", "WordPiece"]
difficulty: intermediate
series: "ai-concepts"
seriesOrder: 15
---

# AI核心概念(15)：Tokenization——AI是怎么「读懂」人类文字的？

> "AI不认识字，它只认识数字。Tokenization就是把文字翻译成数字的密码本。"

## 🎯 什么是Tokenization？

**Tokenization（分词/标记化）** 是将文本**分割成更小的单元（Token）**并转换为数字ID的过程。它是大语言模型处理文本的**第一步**，也是最基础的一步。

### 一句话理解

```text
原始文本: "我喜欢AI"
    ↓ Tokenization
Token序列: ["我", "喜欢", "AI"]
    ↓ 查表转换
ID序列: [2345, 6789, 1024]
    ↓ Embedding
向量序列: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
    ↓ 送入模型
LLM处理
```

**本质**：Tokenization = 文本 → 数字的桥梁

## 🔬 为什么需要Tokenization？

### 问题：AI不认识文字

```text
计算机只认识数字，不认识"你好"

解决方案1：字符级
"你" → 1, "好" → 2
问题：序列太长，语义太碎

解决方案2：词级
"你好" → 1
问题：词表太大，无法处理新词

解决方案3：子词级（Tokenization的方案）
"你好" → ["你", "好"] 或 ["你好"]
平衡了词表大小和语义粒度
```

### 三种粒度对比

| 粒度 | 示例 | 优点 | 缺点 |
|------|------|------|------|
| **字符级** | "AI" → ["A", "I"] | 词表小，无OOV | 序列长，语义弱 |
| **词级** | "人工智能" → ["人工智能"] | 语义完整 | 词表大，有OOV |
| **子词级** | "人工智能" → ["人工", "智能"] | 平衡 | 需要训练 |

**OOV（Out of Vocabulary）**：词表外的词，无法处理

## 🏗️ 主流算法

### 1. BPE（Byte Pair Encoding）

BPE是GPT系列使用的分词算法，核心思想是**迭代合并最常见的字符对**。

```python
class BPETokenizer:
    """简化版BPE分词器"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.merges = []  # 合并规则
        self.vocab = {}   # 词表
    
    def train(self, corpus: list[str]):
        """训练BPE分词器"""
        # 1. 初始化：将所有文本拆分为字符
        word_freqs = {}
        for text in corpus:
            words = text.split()
            for word in words:
                # 将词拆分为字符序列，末尾加</w>标记词结束
                chars = list(word) + ["</w>"]
                key = " ".join(chars)
                word_freqs[key] = word_freqs.get(key, 0) + 1
        
        # 2. 迭代合并
        for i in range(self.vocab_size):
            # 统计所有相邻字符对的频率
            pair_freqs = self._get_pair_frequencies(word_freqs)
            
            if not pair_freqs:
                break
            
            # 找到频率最高的字符对
            best_pair = max(pair_freqs, key=pair_freqs.get)
            
            # 合并这个字符对
            word_freqs = self._merge_pair(word_freqs, best_pair)
            
            # 记录合并规则
            self.merges.append(best_pair)
            print(f"合并 {i+1}: {best_pair} → {''.join(best_pair)}")
        
        # 3. 构建词表
        self._build_vocab(word_freqs)
        
        return self
    
    def _get_pair_frequencies(self, word_freqs: dict) -> dict:
        """统计相邻字符对的频率"""
        pair_freqs = {}
        
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        
        return pair_freqs
    
    def _merge_pair(self, word_freqs: dict, pair: tuple) -> dict:
        """合并指定的字符对"""
        new_word_freqs = {}
        bigram = " ".join(pair)
        replacement = "".join(pair)
        
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
        
        return new_word_freqs
    
    def _build_vocab(self, word_freqs: dict):
        """构建词表"""
        # 收集所有出现过的子词
        all_tokens = set()
        for word in word_freqs:
            all_tokens.update(word.split())
        
        # 按长度排序，长的优先
        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens, key=len, reverse=True))}
    
    def tokenize(self, text: str) -> list[str]:
        """对文本进行分词"""
        tokens = []
        words = text.split()
        
        for word in words:
            # 将词拆分为字符
            chars = list(word) + ["</w>"]
            
            # 应用合并规则
            for merge in self.merges:
                i = 0
                while i < len(chars) - 1:
                    if chars[i] == merge[0] and chars[i + 1] == merge[1]:
                        chars = chars[:i] + ["".join(merge)] + chars[i + 2:]
                    else:
                        i += 1
            
            tokens.extend(chars)
        
        return tokens

# 训练示例
corpus = [
    "low lower newest wide",
    "low low low low",
    "newest newest newest",
    "wider widest wide"
]

tokenizer = BPETokenizer(vocab_size=50)
tokenizer.train(corpus)

# 测试分词
text = "lower newest"
tokens = tokenizer.tokenize(text)
print(f"输入: {text}")
print(f"Token: {tokens}")
```

### 2. WordPiece

WordPiece是BERT使用的分词算法，与BPE类似但选择合并对的标准不同。

```python
class WordPieceTokenizer:
    """简化版WordPiece分词器"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    
    def train(self, corpus: list[str]):
        """训练WordPiece"""
        # 1. 统计词频
        word_freqs = {}
        for text in corpus:
            for word in text.split():
                word_freqs[word] = word_freqs.get(word, 0) + 1
        
        # 2. 初始化词表（所有字符）
        chars = set()
        for word in word_freqs:
            chars.update(word)
        
        # 添加特殊token和基础字符
        for token in self.special_tokens:
            self.vocab[token] = len(self.vocab)
        for char in sorted(chars):
            self.vocab[char] = len(self.vocab)
        
        # 3. 迭代合并（使用互信息作为标准）
        for i in range(self.vocab_size - len(self.vocab)):
            pair_scores = self._calculate_pair_scores(word_freqs)
            
            if not pair_scores:
                break
            
            best_pair = max(pair_scores, key=pair_scores.get)
            new_token = best_pair[0] + best_pair[1]
            
            self.vocab[new_token] = len(self.vocab)
            print(f"添加 {i+1}: {new_token} (score: {pair_scores[best_pair]:.4f})")
        
        return self
    
    def _calculate_pair_scores(self, word_freqs: dict) -> dict:
        """计算字符对的互信息分数"""
        # 统计单个token频率
        token_freqs = {}
        pair_freqs = {}
        
        for word, freq in word_freqs.items():
            # 将词拆分为当前词表能识别的最长子词
            tokens = self._tokenize_word(word)
            
            for token in tokens:
                token_freqs[token] = token_freqs.get(token, 0) + freq
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + freq
        
        # 计算互信息: P(xy) / (P(x) * P(y))
        total_pairs = sum(pair_freqs.values())
        total_tokens = sum(token_freqs.values())
        
        scores = {}
        for pair, pair_freq in pair_freqs.items():
            p_xy = pair_freq / total_pairs
            p_x = token_freqs[pair[0]] / total_tokens
            p_y = token_freqs[pair[1]] / total_tokens
            
            if p_x > 0 and p_y > 0:
                scores[pair] = p_xy / (p_x * p_y)
        
        return scores
    
    def _tokenize_word(self, word: str) -> list[str]:
        """将词拆分为词表中的token"""
        tokens = []
        start = 0
        
        while start < len(word):
            end = len(word)
            found = False
            
            while end > start:
                substr = word[start:end]
                if start > 0:
                    substr = "##" + substr  # 非首token加##前缀
                
                if substr in self.vocab:
                    tokens.append(substr)
                    start = end
                    found = True
                    break
                end -= 1
            
            if not found:
                tokens.append("[UNK]")
                start += 1
        
        return tokens
    
    def tokenize(self, text: str) -> list[str]:
        """对文本进行分词"""
        tokens = []
        for word in text.split():
            tokens.extend(self._tokenize_word(word))
        return tokens

# 训练示例
corpus = [
    "unbelievable low lower",
    "un believable new newest",
    "low low low"
]

tokenizer = WordPieceTokenizer(vocab_size=50)
tokenizer.train(corpus)

# 测试
text = "unbelievable lower"
tokens = tokenizer.tokenize(text)
print(f"输入: {text}")
print(f"Token: {tokens}")
```

### 3. SentencePiece

SentencePiece是Google开发的分词器，支持BPE和Unigram两种算法，可以直接处理原始文本（不需要预分词）。

```python
# 使用SentencePiece库
import sentencepiece as spm

# 训练
spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe',  # 或 'unigram'
    character_coverage=0.9995,
    num_threads=4
)

# 加载
sp = spm.SentencePieceProcessor()
sp.load('tokenizer.model')

# 分词
text = "我喜欢人工智能"
tokens = sp.encode(text, out_type=str)
ids = sp.encode(text, out_type=int)

print(f"文本: {text}")
print(f"Token: {tokens}")
print(f"ID: {ids}")

# 解码
decoded = sp.decode(ids)
print(f"解码: {decoded}")
```

## 💻 实际应用

### 1. OpenAI的Tokenizer

```python
import tiktoken

# GPT-4使用的cl100k_base编码器
enc = tiktoken.encoding_for_model("gpt-4")

# 分词
text = "Hello, world! 你好，世界！"
tokens = enc.encode(text)
token_strs = [enc.decode([t]) for t in tokens]

print(f"文本: {text}")
print(f"Token数量: {len(tokens)}")
print(f"Token: {token_strs}")
print(f"ID: {tokens}")

# 计算成本
# GPT-4 Turbo: $10/1M input tokens
cost_per_token = 10 / 1_000_000
print(f"成本: ${len(tokens) * cost_per_token:.6f}")
```

### 2. 中文分词的挑战

```python
# 中文分词的特殊性

# 英文：天然有空格分隔
"Hello world" → ["Hello", "world"]

# 中文：没有空格
"我喜欢人工智能" → ?

# 可能的分词方式：
# 1. 字级别：["我", "喜", "欢", "人", "工", "智", "能"]
# 2. 词级别：["我", "喜欢", "人工智能"]
# 3. 子词：["我", "喜欢", "人工", "智能"]

# 不同分词导致不同token数量
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4")

texts = [
    "我喜欢人工智能",           # 没有空格
    "我 喜欢 人工智能",         # 手动分词
    "I like artificial intelligence",  # 英文对照
]

for text in texts:
    tokens = enc.encode(text)
    print(f"{text}: {len(tokens)} tokens")
```

### 3. 多语言处理

```python
# 不同语言的token效率差异

import tiktoken

enc = tiktoken.encoding_for_model("gpt-4")

texts = {
    "英文": "Artificial intelligence is transforming the world",
    "中文": "人工智能正在改变世界",
    "日文": "人工知能は世界を変えている",
    "韩文": "인공지능이 세계를 변화시키고 있습니다",
    "阿拉伯文": "الذكاء الاصطنعي يغير العالم",
}

print("各语言Token效率对比:")
print("-" * 50)

for lang, text in texts.items():
    tokens = enc.encode(text)
    chars = len(text)
    tokens_per_char = len(tokens) / chars
    print(f"{lang}: {len(tokens)} tokens / {chars} 字符 = {tokens_per_char:.2f} tokens/字符")

# 输出示例：
# 英文: 9 tokens / 49 字符 = 0.18 tokens/字符
# 中文: 12 tokens / 10 字符 = 1.20 tokens/字符
# 
# 中文的token效率比英文低6-7倍！
# 这就是为什么中文用户消耗token更多
```

## ⚡ Tokenization的影响

### 1. 对成本的影响

```text
同样内容，不同语言的token成本：

英文: "AI is great" → 3 tokens
中文: "人工智能很棒" → 6 tokens
成本差: 2倍

实际影响：
- 中文用户API成本是英文用户的2-3倍
- 长文本处理成本更高
- 需要更精简的prompt
```

### 2. 对性能的影响

```text
Token数量影响：

1. 推理速度：token越多，推理越慢
2. 上下文窗口：同样窗口能装的中文内容更少
3. 生成质量：token边界可能切断词语语义

优化建议：
- 使用简洁的中文表达
- 避免无意义的重复
- 合理使用中英文混合
```

### 3. 对模型能力的影响

```text
Tokenization导致的问题：

1. 数学能力差
   "123 + 456 = ?" → ["123", " +", " 456", " ="]
   模型看到的是token，不是数字位

2. 拼写能力弱
   "strawberry有几个r?" → ["str", "aw", "be", "rry"]
   模型无法直接数字符

3. 代码理解受限
   变量名可能被拆分成无意义的子词
```

## 📊 各模型Tokenizer对比

| 模型 | 算法 | 词表大小 | 特点 |
|------|------|---------|------|
| GPT-2/3 | BPE | 50,257 | 字节级BPE |
| GPT-3.5/4 | tiktoken(cl100k) | 100,256 | 更大词表，更高效 |
| BERT | WordPiece | 30,522 | 中文支持好 |
| T5 | SentencePiece | 32,000 | 多语言支持 |
| LLaMA | SentencePiece | 32,000 | 开源标准 |
| ChatGLM | SentencePiece | 64,794 | 中文优化 |
| Qwen | tiktoken | 151,643 | 中英文平衡 |

## 💡 核心要点总结

**Tokenization的本质**：
- ❌ 不是简单的分词
- ✅ 是文本到数字的编码系统

**三大主流算法**：
1. **BPE**：迭代合并最常见的字符对（GPT系列）
2. **WordPiece**：基于互信息的合并（BERT）
3. **SentencePiece**：直接处理原始文本（多语言）

**对使用者的影响**：
1. **成本**：token越多，API费用越高
2. **速度**：token越多，推理越慢
3. **质量**：token边界可能影响语义理解

**最佳实践**：
- 了解你的模型用什么tokenizer
- 中文表达尽量精简
- 重要信息放在token边界
- 使用专门工具计算token数

---

> **下期预告**：Embedding技术详解——如何让AI理解"国王-男人+女人=女王"？

---

*本文为AI核心概念系列第15期。*
*作者：赛博阿漆 | 发布日期：2026-07-09*
