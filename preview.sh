#!/bin/bash

# 博客本地预览脚本
# 用法: ./preview.sh

set -e

echo "🔫 启动博客本地预览..."
echo ""

if [ ! -d "node_modules" ]; then
    echo "📦 安装依赖..."
    npm install
fi

echo "🌐 启动开发服务器..."
echo "📍 访问地址: http://localhost:4321"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

npm run dev
