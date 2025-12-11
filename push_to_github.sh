#!/bin/bash
# 推送CRAFT框架到GitHub的脚本
# 使用方法：./push_to_github.sh <你的GitHub用户名> <仓库名>

if [ $# -lt 1 ]; then
    echo "使用方法: $0 <GitHub用户名> [仓库名]"
    echo "示例: $0 kalcy craft"
    exit 1
fi

GITHUB_USER=$1
REPO_NAME=${2:-craft}

echo "准备推送到: https://github.com/${GITHUB_USER}/${REPO_NAME}.git"

# 添加远程仓库
git remote add origin https://github.com/${GITHUB_USER}/${REPO_NAME}.git 2>/dev/null || \
git remote set-url origin https://github.com/${GITHUB_USER}/${REPO_NAME}.git

# 推送代码
echo "正在推送代码..."
git push -u origin main

echo "完成！"
echo "你的代码已推送到: https://github.com/${GITHUB_USER}/${REPO_NAME}"



