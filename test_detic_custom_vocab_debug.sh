#!/bin/bash
# DETIC自定义词汇表调试脚本
# 测试不同阈值和物体组合

cd /home/fdse/zzy/craft/Detic

echo "=========================================="
echo "DETIC自定义词汇表调试测试"
echo "=========================================="
echo ""

# 测试1：最基础的物体，阈值0.05
echo "测试1: 基础物体 (cup,table,sink)，阈值0.05"
python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_debug1.jpg \
    --vocabulary custom \
    --custom_vocabulary "cup,table,sink" \
    --cpu \
    --confidence-threshold 0.05 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth 2>&1 | tail -3
echo ""

# 测试2：添加coffee machine，阈值0.05
echo "测试2: 添加coffee machine，阈值0.05"
python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_debug2.jpg \
    --vocabulary custom \
    --custom_vocabulary "coffee machine,cup,table,sink" \
    --cpu \
    --confidence-threshold 0.05 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth 2>&1 | tail -3
echo ""

# 测试3：完整列表，阈值0.01
echo "测试3: 完整列表（包含颜色属性），阈值0.01"
python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_debug3.jpg \
    --vocabulary custom \
    --custom_vocabulary "coffee machine,purple cup,blue cup with handle,cup,table,sink" \
    --cpu \
    --confidence-threshold 0.01 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth 2>&1 | tail -3
echo ""

# 测试4：只使用基础名称（不含颜色属性），阈值0.05
echo "测试4: 只使用基础名称（不含颜色属性），阈值0.05"
python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_debug4.jpg \
    --vocabulary custom \
    --custom_vocabulary "coffee machine,cup,table,sink" \
    --cpu \
    --confidence-threshold 0.05 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth 2>&1 | tail -3
echo ""

echo "=========================================="
echo "测试完成！"
echo "检查输出图像："
echo "  - out_debug1.jpg (基础物体)"
echo "  - out_debug2.jpg (添加coffee machine)"
echo "  - out_debug3.jpg (完整列表，低阈值)"
echo "  - out_debug4.jpg (基础名称)"
echo "=========================================="

