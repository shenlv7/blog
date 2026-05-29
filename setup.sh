#!/bin/bash

# 博客项目初始化脚本
# 用法: ./setup.sh

set -e

echo "🔫 初始化博客项目..."
echo ""

if ! command -v node &> /dev/null; then
    echo "❌ 未找到 Node.js，请先安装 Node.js 18+"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo "❌ 未找到 npm，请先安装 npm"
    exit 1
fi

echo "✅ Node.js 版本: $(node -v)"
echo "✅ npm 版本: $(npm -v)"

echo ""
echo "📦 安装依赖..."
npm install

echo ""
echo "✅ 依赖安装完成！"
echo ""
echo "🚀 可用命令:"
echo "  ./preview.sh    - 本地预览"
echo "  ./new-post.sh   - 新建文章"
echo "  ./deploy.sh     - 一键部署"
echo ""
echo "📝 开始写文章:"
echo "  ./new-post.sh \"我的第一篇文章\" \"学习,笔记\""
echo ""
echo "🌐 本地预览:"
echo "  ./preview.sh"
