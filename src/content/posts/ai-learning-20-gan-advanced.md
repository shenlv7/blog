---
title: "AI学习之路(第20期)：GAN进阶——从训练崩溃到模式突破"
slug: ai-learning-20-gan-advanced
pubDate: 2026-09-02
description: "第二季第八期！深入GAN的训练陷阱、经典变体与现代应用，掌握生成模型的核心技巧"
image: "/blog/images/photo-1620712943543-bcc4688e7485-gan-v2.jpg"
series: "AI学习之路"
episode: 20
tags: ["GAN", "生成对抗网络", "StyleGAN", "WGAN", "扩散模型", "深度学习"]
difficulty: "advanced"
---

![GAN进阶](/blog/images/photo-1620712943543-bcc4688e7485-gan-v2.jpg)

## 上期回顾

上期我们探索了强化学习——让AI在试错中学会决策。今天回到生成模型的世界，深入探讨GAN的**进阶技巧**。

第8期我们从零实现了一个基础GAN，但实际应用中，原始GAN问题多多：训练不稳定、模式崩溃、生成质量差……这期我们来解决这些问题。

---

## 原始GAN的三大噩梦

### 1. 模式崩溃（Mode Collapse）

生成器只学会生成几种"能骗过"判别器的样本，忽略数据的多样性。

```
理想情况：生成各种各样的人脸
模式崩溃：只生成同一张脸的不同角度
```

### 2. 训练不稳定

生成器和判别器的力量失衡，导致训练震荡甚至崩溃。

```
判别器太强 → 生成器梯度消失，学不动
生成器太强 → 判别器失效，失去指导信号
```

### 3. 梯度消失

当判别器太完美时，`log(1-D(G(z)))` 趋近于0，生成器拿不到有效梯度。

---

## 解决方案一：WGAN（Wasserstein GAN）

### 核心思想

用 **Wasserstein距离**（推土机距离）替代JS散度，提供更平滑的梯度。

```
原始GAN目标：min_G max_D [log D(x) + log(1-D(G(z)))]
WGAN目标：    min_G max_D [D(x) - D(G(z))]  （D满足1-Lipschitz约束）
```

### 为什么Wasserstein距离更好？

```
JS散度：两个分布不重叠时，梯度为0 → 训练卡死
Wasserstein距离：即使分布不重叠，也有有意义的梯度 → 持续学习
```

### PyTorch实现

```python
# WGAN判别器（Critic）—— 不用Sigmoid
class WGANCritic(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # 输出实数值，不是概率
        )
    
    def forward(self, img):
        return self.net(img)

# WGAN损失函数
def wasserstein_loss(real_score, fake_score):
    return fake_score.mean() - real_score.mean()

# 梯度惩罚（Gradient Penalty）—— 强制Lipschitz约束
def gradient_penalty(critic, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, device=device)
    alpha = alpha.expand_as(real_data)
    
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        outputs=scores, inputs=interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True, retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty

# WGAN-GP训练循环
lambda_gp = 10
n_critic = 5  # 每训练1次生成器，训练5次判别器

for epoch in range(num_epochs):
    for real_data, _ in dataloader:
        real_data = real_data.to(device)
        
        # 训练Critic
        for _ in range(n_critic):
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_data = generator(z).detach()
            
            real_score = critic(real_data)
            fake_score = critic(fake_data)
            gp = gradient_penalty(critic, real_data, fake_data, device)
            
            c_loss = wasserstein_loss(real_score, fake_score) + lambda_gp * gp
            c_optimizer.zero_grad()
            c_loss.backward()
            c_optimizer.step()
        
        # 训练Generator
        z = torch.randn(batch_size, latent_dim, device=device)
        fake_data = generator(z)
        g_loss = -critic(fake_data).mean()
        
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()
```

---

## 解决方案二：Spectral Normalization

更简单的方案——对判别器的权重做**谱归一化**，直接限制Lipschitz常数。

```python
# PyTorch内置支持
discriminator = nn.Sequential(
    nn.utils.spectral_norm(nn.Linear(784, 512)),
    nn.LeakyReLU(0.2),
    nn.utils.spectral_norm(nn.Linear(512, 256)),
    nn.LeakyReLU(0.2),
    nn.utils.spectral_norm(nn.Linear(256, 1)),
)
```

**优点**：实现简单，计算开销小，效果稳定。SAGAN（Self-Attention GAN）就用的这个。

---

## 经典GAN变体巡礼

### DCGAN（Deep Convolutional GAN）

第一个成功的卷积GAN架构，提出了重要的架构指南：

```python
class DCGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # 输入：latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # channels x 64 x 64
        )
    
    def forward(self, z):
        return self.net(z.view(-1, 100, 1, 1))
```

**DCGAN架构准则**：
- 用BatchNorm（不用LayerNorm）
- 去掉全连接层（用卷积替代）
- 生成器用ReLU（输出层用Tanh）
- 判别器用LeakyReLU

### Conditional GAN（条件GAN）

不只是随机生成，而是**按条件生成**：

```python
class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_dim=784):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + num_classes, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat([z, label_input], dim=-1)
        return self.net(gen_input)

# 用法：生成指定数字
z = torch.randn(1, 100)
labels = torch.tensor([7])  # 生成数字7
fake_img = generator(z, labels)
```

### StyleGAN（风格生成）

NVIDIA的明星产品，实现了前所未有的图像质量：

```
StyleGAN核心创新：
├── Mapping Network：z → w（解纠缠潜在空间）
├── Adaptive Instance Norm (AdaIN)：注入风格
├── Noise Injection：控制随机细节
└── Progressive Growing：渐进式训练
```

```
z (随机噪声) → Mapping Network → w (风格向量)
                                        ↓
                              ┌─────────────────┐
                              │  合成网络        │
                              │  AdaIN + 卷积    │ → 图像
                              │  + 噪声注入      │
                              └─────────────────┘
```

### Pix2Pix（图像翻译）

成对图像的翻译任务：

```python
# U-Net生成器
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # 编码器
        self.enc1 = self._encoder_block(in_channels, 64, normalize=False)
        self.enc2 = self._encoder_block(64, 128)
        self.enc3 = self._encoder_block(128, 256)
        self.enc4 = self._encoder_block(256, 512)
        
        # 解码器（带skip connections）
        self.dec1 = self._decoder_block(512, 256)
        self.dec2 = self._decoder_block(512, 128)  # 256*2 for skip
        self.dec3 = self._decoder_block(256, 64)
        self.dec4 = nn.ConvTranspose2d(128, out_channels, 4, 2, 1)
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # 解码 + skip connections
        d1 = self.dec1(e4)
        d2 = self.dec2(torch.cat([d1, e3], 1))
        d3 = self.dec3(torch.cat([d2, e2], 1))
        d4 = self.dec4(torch.cat([d3, e1], 1))
        return torch.tanh(d4)

# PatchGAN判别器（只判断局部区域）
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 1)  # 输出patch级别的真/假
        )
    
    def forward(self, x, y):
        return self.net(torch.cat([x, y], 1))
```

### CycleGAN（无配对风格迁移）

不需要成对数据，实现马↔斑马转换：

```
核心思想：循环一致性损失

G: 马 → 斑马
F: 斑马 → 马

要求：F(G(马)) ≈ 马（转一圈回来）
```

```python
# 循环一致性损失
def cycle_loss(real_x, reconstructed_x, real_y, reconstructed_y):
    return F.l1_loss(real_x, reconstructed_x) + F.l1_loss(real_y, reconstructed_y)

# 完整损失
def total_loss(D_x, D_y, G, F, real_x, real_y, lambda_cycle=10):
    # 对抗损失
    loss_G = F.mse_loss(D_y(G(real_x)), torch.ones_like(D_y(G(real_x))))
    loss_F = F.mse_loss(D_x(F(real_y)), torch.ones_like(D_x(F(real_y))))
    
    # 循环一致性
    fake_y = G(real_x)
    reconstructed_x = F(fake_y)
    fake_x = F(real_y)
    reconstructed_y = G(fake_x)
    loss_cycle = cycle_loss(real_x, reconstructed_x, real_y, reconstructed_y)
    
    return loss_G + loss_F + lambda_cycle * loss_cycle
```

---

## GAN训练实战技巧

### 1. 学习率平衡

```python
# 生成器学习率通常是判别器的2-4倍
g_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.5, 0.999))
```

### 2. 标签平滑

```python
# 真实标签用0.9而不是1.0
real_labels = torch.full((batch_size, 1), 0.9, device=device)
fake_labels = torch.zeros(batch_size, 1, device=device)
```

### 3. 噪声注入

```python
# 给判别器输入加噪声，防止过拟合
def add_noise(images, std=0.1):
    noise = torch.randn_like(images) * std
    return images + noise
```

### 4. 历史缓冲

```python
# 保存之前生成的假样本
class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []
    
    def query(self, images):
        result = []
        for img in images:
            if len(self.buffer) < self.max_size:
                self.buffer.append(img)
                result.append(img)
            elif random.random() > 0.5:
                idx = random.randint(0, self.max_size - 1)
                old_img = self.buffer[idx]
                self.buffer[idx] = img
                result.append(old_img)
            else:
                result.append(img)
        return torch.stack(result)
```

---

## GAN vs 扩散模型：生成模型之争

### GAN的优势
- ✅ 生成速度快（单次前向传播）
- ✅ 图像细节清晰
- ✅ 适合实时应用

### GAN的劣势
- ❌ 训练不稳定
- ❌ 模式崩溃
- ❌ 难以评估

### 扩散模型的优势
- ✅ 训练稳定
- ✅ 多样性好
- ✅ 理论基础扎实

### 扩散模型的劣势
- ❌ 生成速度慢（需要多步去噪）
- ❌ 计算资源需求大

```python
# 简单对比
# GAN：一步到位
fake_img = generator(z)

# 扩散模型：多步迭代
for t in reversed(range(T)):
    noise_pred = unet(x_t, t)
    x_t = denoise_step(x_t, noise_pred, t)
```

### 现代趋势

2024年以后，扩散模型逐渐主导图像生成（Stable Diffusion、DALL-E），但GAN在以下场景仍有优势：

- **实时图像处理**（人脸滤镜、视频特效）
- **超分辨率**（ESRGAN）
- **图像编辑**（语义编辑、风格混合）
- **数据增强**（生成训练数据）

---

## 实战项目：人脸生成器

用StyleGAN思想实现一个简化版人脸生成器：

```python
class StyleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_dim=512):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.style = nn.Linear(style_dim, out_channels * 2)  # scale和bias
        self.noise_scale = nn.Parameter(torch.zeros(1))
        self.act = nn.LeakyReLU(0.2)
    
    def forward(self, x, w, noise=None):
        # 卷积
        x = self.conv(x)
        
        # 注入风格
        style = self.style(w).unsqueeze(-1).unsqueeze(-1)
        gamma, beta = style.chunk(2, dim=1)
        x = x * (1 + gamma) + beta
        
        # 注入噪声
        if noise is not None:
            x = x + self.noise_scale * noise
        
        return self.act(x)

class MiniStyleGenerator(nn.Module):
    def __init__(self, latent_dim=128, style_dim=128):
        super().__init__()
        # Mapping network
        self.mapping = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
        )
        
        # 合成网络
        self.constant = nn.Parameter(torch.randn(1, 512, 4, 4))
        self.block1 = StyleBlock(512, 256, style_dim)
        self.block2 = StyleBlock(256, 128, style_dim)
        self.block3 = StyleBlock(128, 64, style_dim)
        self.to_rgb = nn.Conv2d(64, 3, 1)
    
    def forward(self, z):
        w = self.mapping(z)
        
        x = self.constant.repeat(z.size(0), 1, 1, 1)
        x = self.block1(x, w, torch.randn_like(x[:, :1]))
        x = F.interpolate(x, scale_factor=2)
        x = self.block2(x, w, torch.randn_like(x[:, :1]))
        x = F.interpolate(x, scale_factor=2)
        x = self.block3(x, w, torch.randn_like(x[:, :1]))
        
        return torch.tanh(self.to_rgb(x))
```

---

## 评估指标

### FID（Fréchet Inception Distance）

```python
# FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2(Σ_r Σ_g)^{1/2})
# 越低越好，衡量生成图像的质量和多样性

from pytorch_fid import fid_score
fid = fid_score.calculate_fid_given_paths(
    ['real_images/', 'generated_images/'],
    batch_size=50,
    device='cuda',
    dims=2048
)
```

### IS（Inception Score）

```python
# IS = exp(E[KL(p(y|x) || p(y))])
# 越高越好，衡量质量和多样性

# 简化版计算
def inception_score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[i * len(preds) // splits:(i + 1) * len(preds) // splits]
        p_y = part.mean(0)
        kl = (part * (torch.log(part) - torch.log(p_y))).sum(1)
        scores.append(kl.exp().mean())
    return torch.stack(scores).mean().item()
```

---

## 本节要点

| 概念 | 要点 |
|------|------|
| 模式崩溃 | 生成器只生成少数几种样本 |
| WGAN | 用Wasserstein距离替代JS散度 |
| 谱归一化 | 简单有效的Lipschitz约束 |
| DCGAN | 第一个成功的卷积GAN架构 |
| 条件GAN | 按指定条件生成 |
| StyleGAN | 风格注入+解纠缠表示 |
| CycleGAN | 无配对的风格迁移 |
| FID/IS | 生成模型的评估指标 |

---

## 下期预告

下一期我们进入 **AI Agent** 的世界——让大模型不只是回答问题，而是能**自主思考、规划、使用工具**完成复杂任务。从ReAct到AutoGPT，从单Agent到多Agent协作，探索AI的"手"和"脚"。

---

## 参考资料

1. Goodfellow et al., "Generative Adversarial Networks" (2014)
2. Arjovsky et al., "Wasserstein GAN" (2017)
3. Karras et al., "A Style-Based Generator Architecture for GANs" (2019)
4. Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks" (2017)
5. Zhu et al., "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks" (2017)
6. Brock et al., "Large Scale GAN Training for High Fidelity Natural Image Synthesis" (2019)
