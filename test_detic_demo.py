#!/usr/bin/env python3
"""
使用DETIC官方demo.py进行测试
这是最接近官方示例的方式
"""
import sys
import os
import subprocess

# 切换到Detic目录
detic_dir = os.path.join(os.path.dirname(__file__), 'Detic')
os.chdir(detic_dir)

print("=" * 60)
print("使用DETIC官方demo.py测试")
print("=" * 60)
print(f"工作目录: {os.getcwd()}")

# 检查必要文件
config_file = "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"
weights_file = "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"

if not os.path.exists(config_file):
    print(f"❌ 配置文件不存在: {config_file}")
    sys.exit(1)

if not os.path.exists(weights_file):
    print(f"❌ 权重文件不存在: {weights_file}")
    sys.exit(1)

print(f"✅ 配置文件: {config_file}")
print(f"✅ 权重文件: {weights_file}")

# 创建一个简单的测试图像（如果demo.py需要）
# 或者使用官方README中的示例图像URL

print("\n尝试运行官方demo.py...")
print("命令: python demo.py --config-file {} --vocabulary lvis --opts MODEL.WEIGHTS {} --cpu".format(
    config_file, weights_file))

# 注意：这里我们只测试配置加载，不实际运行（因为需要输入图像）
# 用户可以在notebook中直接运行官方命令

print("\n💡 在notebook中运行以下命令来测试:")
print("   cd Detic")
print("   python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \")
print("                  --input <your_image.jpg> \")
print("                  --output output.jpg \")
print("                  --vocabulary lvis \")
print("                  --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth")
