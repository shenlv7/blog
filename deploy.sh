#!/bin/bash

# 一键提交博客文章脚本（Gitee版）
# 用法: ./deploy.sh [commit message]

set -e

if [ -n "$1" ]; then
    commit_msg="$1"
else
    commit_msg="post: $(date '+%Y-%m-%d') 更新文章"
fi

echo "🔫 开始部署博客..."
echo ""

if [ -z "$(git status --porcelain)" ]; then
    echo "✅ 没有新的更改需要提交"
    exit 0
fi

echo "📁 将要提交的文件:"
git status --short
echo ""

echo "📦 添加文件..."
git add -A

echo "💾 提交: $commit_msg"
git commit -m "$commit_msg"

echo "🚀 推送到 Gitee..."
git push origin main

echo ""
echo "✅ 部署完成！"
echo "📝 正在更新 Gitee Pages..."
echo "🌐 博客地址: https://shenlv77.gitee.io/blog"
echo ""
echo "💡 提示: 1-2分钟后刷新页面查看更新"
