#!/usr/bin/env python3
"""
最简单的DETIC官方示例测试
直接使用官方demo.py的命令行方式
"""
import sys
import os
import subprocess

# 切换到Detic目录
detic_dir = os.path.join(os.path.dirname(__file__), 'Detic')
original_dir = os.getcwd()

print("=" * 60)
print("DETIC官方demo.py测试")
print("=" * 60)

# 检查必要文件
config_file = "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"
weights_file = "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"

os.chdir(detic_dir)

if not os.path.exists(config_file):
    print(f"❌ 配置文件不存在: {config_file}")
    sys.exit(1)

if not os.path.exists(weights_file):
    print(f"❌ 权重文件不存在: {weights_file}")
    sys.exit(1)

print(f"✅ 配置文件: {config_file}")
print(f"✅ 权重文件: {weights_file}")
print(f"工作目录: {os.getcwd()}")

# 下载测试图像（如果不存在）
test_image = "desk.jpg"
if not os.path.exists(test_image):
    print(f"\n下载测试图像...")
    try:
        import urllib.request
        urllib.request.urlretrieve(
            'https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg',
            test_image
        )
        print(f"✅ 已下载: {test_image}")
    except Exception as e:
        print(f"⚠️  下载失败: {e}")
        print("   请手动下载测试图像或使用您自己的图像")

# 构建命令
cmd = [
    "python", "demo.py",
    "--config-file", config_file,
    "--input", test_image if os.path.exists(test_image) else "test.jpg",  # 如果图像不存在，使用占位符
    "--output", "output_test.jpg",
    "--vocabulary", "lvis",
    "--confidence-threshold", "0.3",
    "--opts", f"MODEL.WEIGHTS={weights_file}"
]

print("\n" + "=" * 60)
print("运行命令:")
print(" ".join(cmd))
print("=" * 60)

print("\n💡 如果遇到'CenterNet未注册'错误，说明DETIC安装不完整")
print("   需要运行: cd Detic && pip install -e .")
print("   和: cd Detic/third_party/CenterNet2 && pip install -e .")

# 注意：这里不实际运行，因为可能需要交互
# 用户可以在notebook中运行
print("\n💡 在notebook中运行:")
print("   cd Detic")
print("   !python demo.py --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \\")
print("                  --input desk.jpg \\")
print("                  --output out.jpg \\")
print("                  --vocabulary lvis \\")
print("                  --confidence-threshold 0.3 \\")
print("                  --opts MODEL.WEIGHTS=models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth")

os.chdir(original_dir)
