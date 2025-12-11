#!/bin/bash
# 使用代理推送CRAFT到GitHub的脚本

# 设置代理（根据你的clash配置修改）
PROXY_HOST="127.0.0.1"
PROXY_PORT="7890"

echo "设置代理: http://${PROXY_HOST}:${PROXY_PORT}"

# 设置环境变量
export http_proxy=http://${PROXY_HOST}:${PROXY_PORT}
export https_proxy=http://${PROXY_HOST}:${PROXY_PORT}
export HTTP_PROXY=http://${PROXY_HOST}:${PROXY_PORT}
export HTTPS_PROXY=http://${PROXY_HOST}:${PROXY_PORT}

# 配置git代理
git config --global http.proxy http://${PROXY_HOST}:${PROXY_PORT}
git config --global https.proxy http://${PROXY_HOST}:${PROXY_PORT}

# 切换到craft目录
cd /home/fdse/zzy/craft

# 检查远程仓库
echo "检查远程仓库配置..."
git remote -v

# 尝试推送
echo "开始推送代码到GitHub..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo "✅ 推送成功！"
    echo "仓库地址: https://github.com/Leo0408/craft"
else
    echo "❌ 推送失败，可能的原因："
    echo "1. 代理配置不正确，请检查clash是否运行在端口 ${PROXY_PORT}"
    echo "2. 网络连接问题"
    echo "3. 需要GitHub认证（Personal Access Token）"
    echo ""
    echo "如果使用HTTPS方式，可能需要："
    echo "1. 在GitHub设置中创建Personal Access Token"
    echo "2. 推送时使用token作为密码"
    echo ""
    echo "或者尝试使用SSH方式（需要配置SSH密钥）："
    echo "git remote set-url origin git@github.com:Leo0408/craft.git"
fi

