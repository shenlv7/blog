#!/bin/bash

# 快速添加新文章脚本
# 用法: ./new-post.sh "文章标题" "标签1,标签2"

set -e

if [ -z "$1" ]; then
    echo "❌ 请提供文章标题"
    echo "用法: ./new-post.sh \"文章标题\" \"标签1,标签2\""
    exit 1
fi

title="$1"
tags="${2:-}"
date=$(date '+%Y-%m-%d')
slug=$(echo "$title" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | sed 's/^-//' | sed 's/-$//')
filename="src/content/posts/${slug}.md"

if [ -n "$tags" ]; then
    tags_section=$(echo "$tags" | tr ',' '\n' | sed 's/^/  - /' | tr '\n' '\n')
else
    tags_section="  - 未分类"
fi

echo "📝 创建新文章: $title"
echo "📁 文件: $filename"

cat > "$filename" << EOF
---
title: "$title"
description: "请填写文章描述"
pubDate: $date
tags:
$tags_section
---

## 引言

请填写引言内容...

## 核心概念

请填写主要内容...

## 代码示例

\`\`\`python
# 在这里添加代码示例
\`\`\`

## 实践建议

请填写实践建议...

## 总结

请填写总结...

## 参考资料

- [参考1](https://example.com)
- [参考2](https://example.com)
EOF

echo "✅ 文章创建完成！"
echo ""
echo "📄 文件: $filename"
echo "✏️  请编辑文章内容"
echo ""
echo "🚀 编辑完成后运行: ./deploy.sh \"post: $title\""
