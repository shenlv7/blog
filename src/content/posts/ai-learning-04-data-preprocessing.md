---
title: "AI学习之路(第4期)：数据预处理与特征工程"
description: "模型再牛，数据不行也白搭。从清洗、转换到特征构建，掌握让模型'吃好饭'的核心技能"
pubDate: 2026-06-03
tags: ["数据预处理", "特征工程", "Pandas", "Scikit-learn", "数据清洗", "AI基础"]
---

![数据预处理与特征工程](https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=800&h=400&fit=crop)

## 引言

前三期我们聊了模型、聊了网络、聊了架构，但有一个残酷的事实一直没说破：**在真实世界里，80%的时间花在数据上，而不是模型上。**

Kaggle上有个经典名言："Garbage in, garbage out"——垃圾进，垃圾出。再精妙的模型，喂它一堆乱七八糟的数据，出来的东西也只能是垃圾。

这一期，我们回到最基础也最重要的环节：怎么让数据变得"能吃"，怎么从中榨取出最有价值的信息。

## 一、数据清洗：给数据"洗个澡"

现实世界的数据，就像刚从菜市场买回来的菜——有泥巴、有烂叶子、还有虫子。第一步，得洗干净。

### 1.1 缺失值处理

数据集中最常见的问题就是缺失值。处理方式取决于缺失的比例和含义：

```python
import pandas as pd
import numpy as np

# 模拟一个有缺失值的数据集
df = pd.DataFrame({
    'age': [25, np.nan, 30, 45, np.nan, 28],
    'salary': [5000, 8000, np.nan, 12000, 7500, np.nan],
    'city': ['北京', '上海', None, '深圳', '北京', '上海']
})

# 1. 直接删除（缺失比例很小时）
df_drop = df.dropna()

# 2. 均值/中位数/众数填充
df['age'].fillna(df['age'].median(), inplace=True)  # 数值用中位数更稳健
df['city'].fillna(df['city'].mode()[0], inplace=True)  # 分类用众数

# 3. 前向/后向填充（时间序列常用）
df_ffill = df.fillna(method='ffill')

# 4. 标记缺失（让模型自己学）
df['age_missing'] = df['age'].isnull().astype(int)
```

**经验法则**：缺失超过70%的列，考虑直接丢弃；20%-70%之间，创建"是否缺失"的标记特征；20%以下，填充即可。

### 1.2 异常值检测

异常值就像人群中的姚明——不是数据错了，就是极端情况。

```python
# 方法1：IQR（四分位距）法
Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['salary'] < lower) | (df['salary'] > upper)]
print(f"发现 {len(outliers)} 个异常值")

# 方法2：Z-Score法（假设正态分布）
from scipy import stats
z_scores = np.abs(stats.zscore(df['salary'].dropna()))
outliers_z = df[z_scores > 3]  # 超过3个标准差

# 处理：截断（Winsorization）
df['salary_clipped'] = df['salary'].clip(lower, upper)
```

### 1.3 数据类型与格式统一

```python
# 日期解析
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# 类别统一（去空格、统一大小写）
df['city'] = df['city'].str.strip().str.lower()

# 去重
df.drop_duplicates(subset=['user_id'], keep='last', inplace=True)
```

![数据清洗流程](https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3?w=800&h=400&fit=crop)

## 二、特征转换：让数据"说人话"

清洗完的数据，还需要转换成模型能理解的形式。

### 2.1 数值缩放

不同特征的量纲差异巨大（年龄：0-100，收入：0-1000000），这会严重影响基于距离的模型（KNN、SVM）和梯度下降的收敛速度。

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# 标准化（Z-Score）：均值0，方差1 —— 最常用
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 归一化（Min-Max）：缩放到[0,1]
minmax = MinMaxScaler()
X_norm = minmax.fit_transform(X)

# 稳健缩放：对异常值不敏感
robust = RobustScaler()
X_robust = robust.fit_transform(X)
```

**选择指南**：
- 数据近似正态分布 → StandardScaler
- 需要固定范围（如图像像素） → MinMaxScaler
- 有明显异常值 → RobustScaler
- 树模型（随机森林、XGBoost） → 可以不缩放

### 2.2 类别编码

模型只认数字，不认文字。类别特征需要编码。

```python
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

# 标签编码（有序类别）
le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])
# 小学→0, 初中→1, 高中→2, 本科→3 ...

# 独热编码（无序类别）
ohe = OneHotEncoder(sparse_output=False)
city_encoded = ohe.fit_transform(df[['city']])
# 北京 → [1,0,0], 上海 → [0,1,0], 深圳 → [0,0,1]

# Pandas快捷方式
df = pd.get_dummies(df, columns=['city'], drop_first=True)  # drop_first避免多重共线性
```

### 2.3 数值变换

有些特征的分布偏态严重，取个对数能让它更"正常"。

```python
# 对数变换（处理右偏分布，如收入、房价）
df['log_salary'] = np.log1p(df['salary'])  # log1p避免log(0)

# Box-Cox变换（自动找最佳变换）
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='box-cox')  # 要求数据>0
X_transformed = pt.fit_transform(X_positive)

# 分箱（连续变量离散化）
df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 100],
                        labels=['少年', '青年', '中年', '老年'])
```

## 三、特征工程：从"有数据"到"有特征"

这是最考验功力的环节。好的特征能让简单模型吊打复杂模型。

### 3.1 数值特征构造

```python
# 交互特征
df['income_per_year'] = df['salary'] / (df['work_years'] + 1)
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

# 多项式特征
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=True)
X_poly = poly.fit_transform(X)

# 聚合特征（分组统计）
city_stats = df.groupby('city')['salary'].agg(['mean', 'std', 'max'])
city_stats.columns = ['city_salary_mean', 'city_salary_std', 'city_salary_max']
df = df.merge(city_stats, on='city', how='left')
```

### 3.2 时间特征

时间是被严重低估的特征金矿。

```python
# 基础分解
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek  # 0=周一
df['hour'] = df['date'].dt.hour
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# 周期性编码（让模型理解"12月和1月很近"）
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# 滚动窗口特征
df['salary_rolling_3m'] = df.groupby('user_id')['salary'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
```

### 3.3 文本特征

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF
tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
text_features = tfidf.fit_transform(df['text_column'])

# 简单文本统计
df['text_len'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['has_question'] = df['text'].str.contains(r'\?').astype(int)
```

![特征工程思维](https://images.unsplash.com/photo-1509228468518-180dd4864904?w=800&h=400&fit=crop)

## 四、特征选择：不是越多越好

特征太多会导致维度灾难，还会引入噪声。

```python
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

# 方法1：基于统计检验
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)

# 方法2：基于模型重要性
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=feature_names)
top_features = importances.nlargest(20).index.tolist()

# 方法3：相关性过滤
corr_matrix = df.corr().abs()
high_corr = corr_matrix[corr_matrix > 0.95]  # 找到高度相关的特征对
# 保留其中一个，删除另一个
```

## 五、实战Pipeline：一键搞定

Scikit-learn的Pipeline让整个流程优雅串联：

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# 定义不同列的处理方式
numeric_features = ['age', 'salary', 'work_years']
categorical_features = ['city', 'education']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# 组合
preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# 接入模型
from sklearn.ensemble import GradientBoostingClassifier
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier())
])

# 一行代码训练
full_pipeline.fit(X_train, y_train)
score = full_pipeline.score(X_test, y_test)
print(f"准确率: {score:.4f}")
```

## 实践建议

1. **先跑baseline**：用原始数据跑一个简单模型，知道"地板"在哪，再做特征工程看提升
2. **EDA先行**：画分布图、相关性热力图，用眼睛看数据比用脑子猜靠谱
3. **避免数据泄露**：所有fit操作只能在训练集上做，transform应用到测试集
4. **版本控制数据**：用DVC或Git LFS追踪数据变更，别只管代码不管数据
5. **记录实验**：每次特征工程的尝试和效果都要记下来，避免重复踩坑

## 参考资料

- [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Feature Engineering for Machine Learning (Zheng & Casari)](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/)
- [Kaggle Feature Engineering Course](https://www.kaggle.com/learn/feature-engineering)
- [Google ML Crash Course - Data Preparation](https://developers.google.com/machine-learning/data-preparation)

---

下一期我们进入中级篇，聊聊**NLP自然语言处理**——让机器读懂人话的技术。敬请期待！

*本文由赛博阿漆AI助手自动生成*
