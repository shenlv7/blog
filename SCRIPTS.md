# 🔫 博客脚本工具

## 快速开始

```bash
# 1. 初始化项目（首次使用）
./setup.sh

# 2. 本地预览
./preview.sh

# 3. 新建文章
./new-post.sh "文章标题" "标签1,标签2"

# 4. 一键部署
./deploy.sh "post: 文章标题"
```

## 脚本说明

### setup.sh
初始化项目，安装依赖。

**用法：**
```bash
./setup.sh
```

### preview.sh
启动本地开发服务器预览博客。

**用法：**
```bash
./preview.sh
```

**访问地址：** http://localhost:4321

### new-post.sh
快速创建新文章模板。

**用法：**
```bash
./new-post.sh "文章标题" "标签1,标签2"
```

**示例：**
```bash
./new-post.sh "机器学习入门" "机器学习,Python"
```

**生成的文件：** `src/content/posts/机器学习入门.md`

### deploy.sh
一键提交并部署博客到 GitHub Pages。

**用法：**
```bash
./deploy.sh "commit message"
```

**示例：**
```bash
./deploy.sh "post: 添加机器学习入门文章"
```

**功能：**
- 自动检测更改
- 添加所有文件
- 提交更改
- 推送到 GitHub
- 触发 GitHub Actions 自动部署

## 工作流程

### 写新文章

1. **创建文章模板**
   ```bash
   ./new-post.sh "AI学习之路(第2期)" "深度学习,神经网络"
   ```

2. **编辑文章内容**
   - 打开生成的 `.md` 文件
   - 填写标题、描述、标签
   - 编写正文内容
   - 添加配图（Unsplash URL）

3. **本地预览**
   ```bash
   ./preview.sh
   ```
   访问 http://localhost:4321 查看效果

4. **部署上线**
   ```bash
   ./deploy.sh "post: AI学习之路(第2期)"
   ```

5. **等待部署**
   - GitHub Actions 自动构建（1-2分钟）
   - 访问 https://shenlv7.github.io/blog/ 查看

### 修改现有文章

1. **编辑文件**
   - 修改 `src/content/posts/` 下的 `.md` 文件

2. **预览更改**
   ```bash
   ./preview.sh
   ```

3. **部署更改**
   ```bash
   ./deploy.sh "update: 修改文章标题"
   ```

## 文章格式

### Frontmatter

```yaml
---
title: "文章标题"
description: "文章描述（100-150字）"
pubDate: 2026-05-29
tags: ["标签1", "标签2"]
---
```

### 正文结构

```markdown
## 引言

简要介绍主题...

## 核心概念

详细讲解...

## 代码示例

```python
# 代码示例
```

## 实践建议

实际应用建议...

## 总结

要点回顾...

## 参考资料

- [链接1](https://example.com)
- [链接2](https://example.com)
```

## 配图规范

### 图片来源
- Unsplash: https://unsplash.com
- Pexels: https://pexels.com

### 图片格式
```markdown
![描述](https://images.unsplash.com/photo-xxx?w=800&h=400&fit=crop)
```

### 图片数量
- 每篇文章 3-5 张配图
- 关键概念处插入
- 避免大段文字无图

## 标签规范

### 常用标签
- 机器学习
- 深度学习
- 神经网络
- 自然语言处理
- 计算机视觉
- 强化学习
- 生成对抗网络
- Transformer
- 大语言模型
- Python
- TensorFlow
- PyTorch

### 标签命名
- 使用中文
- 保持简洁（2-4个字）
- 避免重复标签

## 故障排除

### 依赖安装失败
```bash
rm -rf node_modules package-lock.json
npm install
```

### 本地预览报错
```bash
npm run dev
```
查看错误信息，通常是语法错误或缺少依赖。

### 推送失败
检查 GitHub Token 是否过期：
```bash
git remote set-url origin https://shenlv7:<TOKEN>@github.com/shenlv7/blog.git
```

### GitHub Actions 构建失败
1. 查看 Actions 页面错误信息
2. 检查 `astro.config.mjs` 配置
3. 确认 `package.json` 依赖版本

## 联系方式

- GitHub: https://github.com/shenlv7
- 邮箱: 2056440151@qq.com
