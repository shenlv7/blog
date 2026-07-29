---
title: "AI学习之路(第15期)：神经网络架构——从MLP到ResNet的进化之路"
slug: ai-learning-15-neural-network-architectures-v2
pubDate: 2026-07-29
description: "第二季第三期！深入探索神经网络的核心架构，从最简单的多层感知机到革命性的残差网络，理解网络设计的底层逻辑"
image: "https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=1200"
series: "AI学习之路"
episode: 15
tags: ["神经网络", "MLP", "CNN", "ResNet", "网络架构", "深度学习"]
difficulty: "intermediate"
---

![神经网络架构](https://images.unsplash.com/photo-1620712943543-bcc4688e7485?w=800&h=400&fit=crop)

## 上期回顾

上期我们从感知机出发，理解了深度学习的数学基础和反向传播的工作原理。今天我们更进一步，聊聊神经网络的**架构设计**——这就像搭积木，同样的砖块（神经元），不同的搭建方式会产生截然不同的效果。

## 为什么架构很重要？

想象一下，你有一堆乐高积木。你可以搭一座简单的房子，也可以搭一艘复杂的宇宙飞船。神经网络架构就是这个"搭建蓝图"：

- **MLP**（多层感知机）→ 简单的平房，能住人，但功能有限
- **CNN**（卷积神经网络）→ 带窗户和阳台的别墅，专门处理图像
- **ResNet**（残差网络）→ 摩天大楼，解决了"楼盖太高会塌"的问题

架构决定了网络能学什么、学多好、学多快。

## 一、MLP：万能但低效的"平房"

多层感知机是最基础的神经网络架构，也叫全连接网络：

```python
import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)

# 创建一个简单的MLP
model = SimpleMLP(784, 256, 10)  # 输入784维，隐藏层256维，输出10类
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

**MLP的特点：**
- ✅ 结构简单，容易理解
- ✅ 理论上可以逼近任意函数
- ❌ 参数量巨大（每层都是全连接）
- ❌ 无法捕捉空间/时间结构信息

## 二、CNN：专治图像的"别墅"

卷积神经网络的革命性思想：**局部连接 + 权重共享**。

人类看图片时，不是一次性看所有像素，而是先看局部特征（边缘、纹理），再组合成整体。CNN模仿了这个过程：

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            # 卷积层：提取特征
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # 3通道RGB，输出32个特征图
            nn.ReLU(),
            nn.MaxPool2d(2),  # 池化：压缩空间尺寸
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
```

**卷积操作的直觉理解：**

想象一个小窗口（卷积核）在图片上滑动，每次只看一小块区域。这个窗口学会了识别"边缘"、"角点"等基本特征。更深的层把这些简单特征组合成复杂特征（眼睛、鼻子、整张脸）。

**CNN的优势：**
- ✅ 参数共享：同一个卷积核在整张图上复用
- ✅ 平移不变性：猫在左边还是右边都能认出来
- ✅ 层次化特征：低层→边缘，中层→纹理，高层→物体

## 三、ResNet：解决深度问题的"摩天大楼"

理论上，网络越深，表达能力越强。但实际训练中发现：**网络太深反而效果变差**！这不是过拟合，而是梯度消失/爆炸导致的。

2015年，何恺明团队提出了ResNet，用一个巧妙的"快捷连接"解决了这个问题：

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x  # 保存输入
        
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual  # 关键：加上原始输入！
        out = torch.relu(out)
        return out
```

**残差连接的魔法：**

传统的网络需要学习完整的映射 `H(x) = 输出`，而ResNet只需要学习残差 `F(x) = H(x) - x`，即 `H(x) = F(x) + x`。

这就像考试：传统网络要从0分考到100分，ResNet只需要学习"比及格多多少分"。学习增量比学习全部容易得多！

**ResNet的成就：**
- 2015年ImageNet冠军，错误率3.57%（超越人类的5.1%）
- 成功训练了152层的网络（之前超过20层就很难训练）
- 开创了"跳跃连接"的设计范式，影响了后续几乎所有架构

## 四、架构演进的启示

从MLP到ResNet，我们能学到什么设计原则？

| 原则 | 说明 | 代表架构 |
|------|------|----------|
| 参数共享 | 同一参数在多处复用 | CNN的卷积核 |
| 稀疏连接 | 每个神经元只连接部分输入 | CNN的局部感受野 |
| 残差学习 | 学习增量而非完整映射 | ResNet |
| 注意力机制 | 动态关注重要信息 | Transformer（下期预告！）|

## 五、动手实验：用PyTorch搭建ResNet

下面是一个完整的、可运行的ResNet实现，用于CIFAR-10图像分类：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

class ResNet9(nn.Module):
    def __init__(self):
        super(ResNet9, self).__init__()
        
        self.conv1 = self._conv_block(3, 64)
        self.conv2 = self._conv_block(64, 128, pool=True)
        self.res1 = self._residual_block(128)
        
        self.conv3 = self._conv_block(128, 256, pool=True)
        self.conv4 = self._conv_block(256, 512, pool=True)
        self.res2 = self._residual_block(512)
        
        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(512, 10)
        )
    
    def _conv_block(self, in_c, out_c, pool=False):
        layers = [
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        return nn.Sequential(*layers)
    
    def _residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out  # 残差连接
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out  # 残差连接
        out = self.classifier(out)
        return out

# 训练代码
def train_model():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2)
    
    model = ResNet9()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(10):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 100 == 99:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0
    
    print('训练完成！')
    return model

if __name__ == '__main__':
    model = train_model()
```

## 实践建议

1. **从小网络开始**：先用简单架构验证想法，再考虑加深加宽
2. **善用预训练模型**：torchvision.models里有现成的ResNet、VGG等
3. **监控梯度**：用TensorBoard观察梯度流动，诊断训练问题
4. **理解而非记忆**：重点理解设计动机，而不是背诵架构细节

## 推荐资源

- 📖 [Deep Learning Book](https://www.deeplearningbook.org/) - Ian Goodfellow的经典教材
- 🎥 [CS231n](https://cs231n.stanford.edu/) - 斯坦福计算机视觉课程
- 📱 [d2l.ai](https://d2l.ai/) - 动手学深度学习，有中文版
- 🔬 [Papers With Code](https://paperswithcode.com/) - 论文+代码+排行榜

## 下期预告

下一期我们将进入**NLP自然语言处理**的世界！从词袋模型到Word2Vec，看看计算机是如何"理解"人类语言的。

---

*本文由赛博阿漆AI助手自动生成*
