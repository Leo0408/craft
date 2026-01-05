#!/bin/bash
# DETIC自定义词汇表测试脚本

cd /home/fdse/zzy/craft/Detic

echo "=========================================="
echo "DETIC自定义词汇表检测测试"
echo "=========================================="
echo ""

# 测试1: 检测厨房相关物体（基础版）
echo "测试1: 基础厨房物体检测"
python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_custom_basic.jpg \
    --vocabulary custom \
    --custom-vocabulary "coffee maker,cup,mug,sink,faucet,kettle,pot,stove" \
    --cpu \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth

echo ""
echo "=========================================="

# 测试2: 检测厨房相关物体（扩展版，包含同义词）
echo "测试2: 扩展版厨房物体检测（包含同义词）"
python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_custom_extended.jpg \
    --vocabulary custom \
    --custom-vocabulary "coffee maker,coffee machine,espresso machine,cup,mug,coffee cup,drinking cup,sink,kitchen sink,basin,faucet,tap,water tap,kettle,electric kettle,teapot,pot,cooking pot,saucepan,stove,range,cooktop" \
    --cpu \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth

echo ""
echo "=========================================="
echo "测试完成！"
echo "输出文件:"
echo "  - out_custom_basic.jpg (基础版)"
echo "  - out_custom_extended.jpg (扩展版)"
echo "=========================================="

