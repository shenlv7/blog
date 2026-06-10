---
title: "AI学习之路(第6期)：计算机视觉"
slug: ai-learning-06-computer-vision
pubDate: 2026-06-10
description: "从像素到理解，探索计算机视觉的核心技术与实战应用"
image: "https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=1200"
tags: ["AI学习", "计算机视觉", "CNN", "图像识别", "深度学习"]
series: "AI学习之路"
episode: 6
---

# AI学习之路(第6期)：计算机视觉

![计算机视觉](https://images.unsplash.com/photo-1555949963-aa79dcee981c?w=800)

> "眼睛是心灵的窗户，而计算机视觉是AI的眼睛。"

## 前言

上一期我们探索了NLP——让机器理解语言。这一期，我们转向另一个让人兴奋的方向：**计算机视觉（Computer Vision, CV）**——让机器"看懂"这个世界。

你每天都在用计算机视觉：手机人脸解锁、拍照自动美颜、自动驾驶识别路障、医学影像分析……CV已经渗透到生活的方方面面。今天，我们来拆解它背后的核心技术。

---

## 什么是计算机视觉？

计算机视觉的目标是让机器从图像或视频中提取有意义的信息。核心任务包括：

| 任务 | 描述 | 应用场景 |
|------|------|----------|
| **图像分类** | 这张图是什么？ | 相册自动分类 |
| **目标检测** | 图里有什么？在哪？ | 自动驾驶、安防监控 |
| **语义分割** | 每个像素属于什么？ | 医学影像、地图测绘 |
| **图像生成** | 创造新图像 | AI绘画、图像修复 |
| **姿态估计** | 人的动作是什么？ | 健身APP、游戏动捕 |

---

## 核心技术演进

### 1. 传统方法（深度学习之前）

在CNN大行其道之前，CV靠的是**手工特征提取**：

- **边缘检测**：Sobel、Canny算子找图像边缘
- **特征点**：SIFT、SURF提取关键点
- **HOG特征**：描述图像局部形状
- **SVM分类器**：用提取的特征做分类

```python
# 传统边缘检测示例（OpenCV）
import cv2
import numpy as np

img = cv2.imread('photo.jpg', cv2.IMREAD_GRAYSCALE)
# Canny边缘检测
edges = cv2.Canny(img, threshold1=100, threshold2=200)
cv2.imwrite('edges.jpg', edges)
```

这些方法有效但脆弱——换个角度、换个光照，特征就变了。

### 2. CNN时代：让机器自己学特征

2012年，**AlexNet**在ImageNet比赛中以巨大优势夺冠，开启了深度学习在CV领域的统治。CNN的核心思想：**让网络自动学习特征**。

#### CNN的基本组件

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 卷积层：提取局部特征
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 池化层：降低空间维度
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层：分类决策
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 10)  # 10个类别
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # 32x32 -> 16x16
        x = self.pool(self.relu(self.conv2(x)))  # 16x16 -> 8x8
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

#### 经典CNN架构一览

| 模型 | 年份 | 亮点 |
|------|------|------|
| **AlexNet** | 2012 | 开启深度学习CV时代 |
| **VGGNet** | 2014 | 证明"更深"的力量 |
| **GoogLeNet** | 2014 | Inception模块，多尺度特征 |
| **ResNet** | 2015 | 残差连接，训练超深网络（152层） |
| **EfficientNet** | 2019 | 平衡深度/宽度/分辨率 |
| **Vision Transformer** | 2020 | 将Transformer引入CV |

### 3. 迁移学习：站在巨人肩膀上

实际项目中，我们很少从头训练模型。**迁移学习**是CV工程师的杀手锏：

```python
import torchvision.models as models
import torch.nn as nn

# 加载预训练的ResNet18
model = models.resnet18(pretrained=True)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 只替换最后的分类层（假设分5类）
model.fc = nn.Linear(model.fc.in_features, 5)

# 只需训练最后的fc层，省时省力
```

**为什么迁移学习有效？** 预训练模型在ImageNet（1400万张图片）上学到的特征（边缘、纹理、形状）具有通用性，迁移到你的小数据集上依然有效。

---

## 目标检测：不只认出是什么，还要知道在哪

### 两阶段检测器

- **R-CNN系列**（R-CNN → Fast R-CNN → Faster R-CNN）
- 先找"候选区域"，再分类
- 精度高，但速度较慢

### 单阶段检测器

- **YOLO系列**（You Only Look Once）
- 一次前向传播搞定定位+分类
- 速度快，适合实时场景

```python
# 使用YOLOv8进行目标检测（Ultralytics库）
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8n.pt')

# 推理
results = model('street.jpg')

# 输出检测结果
for result in results:
    boxes = result.boxes
    for box in boxes:
        cls = int(box.cls[0])       # 类别
        conf = float(box.conf[0])   # 置信度
        xyxy = box.xyxy[0].tolist() # 边界框坐标
        print(f"检测到: {model.names[cls]}, 置信度: {conf:.2f}")
```

---

## 实战：用PyTorch做图像分类

来看一个完整的图像分类流程：

```python
import torch
import torchvision
import torchvision.transforms as transforms

# 1. 数据准备（以CIFAR-10为例）
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),   # 数据增强
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                           shuffle=True, num_workers=2)

# 2. 定义模型
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 3. 训练
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    running_loss = 0.0
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")

print("训练完成！")
```

---

## 计算机视觉的前沿

### Vision Transformer (ViT)

2020年Google提出ViT，将NLP中的Transformer架构引入CV：

- 把图像切成16×16的小块（patch）
- 每个patch当作一个"token"
- 用Self-Attention捕捉全局关系

ViT在大数据集上表现超越CNN，证明了**Transformer在CV领域的潜力**。

### 多模态融合

- **CLIP**：同时理解图像和文本
- **DALL-E / Stable Diffusion**：文本生成图像
- **GPT-4V**：理解图像内容并对话

视觉与语言的融合，正在打破模态的边界。

---

## 学习建议

1. **先掌握CNN基础**：理解卷积、池化、特征图的概念
2. **动手跑经典模型**：用PyTorch复现LeNet、AlexNet
3. **善用迁移学习**：实际项目中，ResNet/EfficientNet + 微调是最实用的方案
4. **关注目标检测**：YOLO系列是工业界最爱
5. **跟进前沿**：ViT、多模态模型代表了CV的未来方向

### 推荐学习资源

- **课程**：Stanford CS231n（计算机视觉经典课程）
- **书籍**：《深度学习》第9章（Goodfellow）
- **实践**：Kaggle图像分类竞赛
- **框架**：PyTorch + torchvision + Ultralytics

---

## 参考资料

- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.stanford.edu/)
- [PyTorch官方教程](https://pytorch.org/tutorials/)
- [YOLOv8文档](https://docs.ultralytics.com/)
- [Attention Is All You Need (ViT论文)](https://arxiv.org/abs/2010.11929)
- [CLIP: Connecting Text and Images](https://openai.com/research/clip)

---

下一期，我们将进入**强化学习**的世界——让AI学会在环境中做出最优决策，从AlphaGo到机器人控制，敬请期待！

---

*本文由赛博阿漆AI助手自动生成*
