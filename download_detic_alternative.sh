#!/bin/bash
# DETIC 模型下载脚本（处理 403 错误）

MODEL_URL="https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R50_1x.pth"
OUTPUT_FILE="detic_LCOCOI21k_CLIP_R50_1x.pth"

echo "=" ========================================
echo "DETIC 模型下载脚本"
echo "=" ========================================
echo ""

# 方法 1: 使用 curl with User-Agent
echo "方法 1: 使用 curl (推荐)"
curl -L -A "Mozilla/5.0" -o "$OUTPUT_FILE" "$MODEL_URL" && echo "✅ 下载成功" || echo "❌ 下载失败"

# 如果方法 1 失败，尝试方法 2
if [ ! -f "$OUTPUT_FILE" ] || [ ! -s "$OUTPUT_FILE" ]; then
    echo ""
    echo "方法 2: 使用 wget with User-Agent"
    wget --user-agent="Mozilla/5.0" -O "$OUTPUT_FILE" "$MODEL_URL" && echo "✅ 下载成功" || echo "❌ 下载失败"
fi

# 如果都失败，提供手动下载说明
if [ ! -f "$OUTPUT_FILE" ] || [ ! -s "$OUTPUT_FILE" ]; then
    echo ""
    echo "=" ========================================
    echo "自动下载失败，请手动下载："
    echo "=" ========================================
    echo "1. 在浏览器中打开以下 URL:"
    echo "   $MODEL_URL"
    echo ""
    echo "2. 下载文件并保存为: $OUTPUT_FILE"
    echo ""
    echo "3. 或者使用以下命令（如果安装了 aria2）:"
    echo "   aria2c -x 16 -s 16 $MODEL_URL -o $OUTPUT_FILE"
fi
