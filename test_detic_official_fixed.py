# ============================================================
# DETIC官方demo测试（修复路径问题版本）
# ============================================================

import os
import sys
import ssl
import urllib.request
import subprocess
from PIL import Image
import matplotlib.pyplot as plt

# 修复SSL证书问题
ssl._create_default_https_context = ssl._create_unverified_context

# 1. 找到Detic目录（支持多种路径）
original_dir = os.getcwd()
print(f"当前目录: {original_dir}")

# 尝试找到Detic目录
detic_dir = None
possible_paths = [
    'Detic',  # 相对路径
    './Detic',
    os.path.join(os.path.dirname(os.path.abspath('__file__')), 'Detic') if '__file__' in globals() else None,
    os.path.join(os.getcwd(), 'Detic'),
    os.path.join(os.path.dirname(os.getcwd()), 'Detic'),
]

# 检查当前目录的父目录（notebook可能在子目录中运行）
if os.path.basename(os.getcwd()) != 'craft':
    # 尝试向上查找craft目录
    current = os.getcwd()
    for _ in range(3):  # 最多向上3级
        parent = os.path.dirname(current)
        possible_path = os.path.join(parent, 'Detic')
        possible_paths.append(possible_path)
        if os.path.basename(parent) == 'craft':
            break
        current = parent

for path in possible_paths:
    if path and os.path.exists(path) and os.path.isdir(path):
        detic_dir = os.path.abspath(path)
        print(f"✅ 找到Detic目录: {detic_dir}")
        break

if not detic_dir:
    print("❌ 未找到Detic目录")
    print("   请确保在正确的目录运行，或者手动设置detic_dir变量")
    print(f"   当前目录: {os.getcwd()}")
    print("   尝试查找的路径:")
    for path in possible_paths:
        if path:
            print(f"     - {path}")
    sys.exit(1)

# 切换到Detic目录
os.chdir(detic_dir)
print(f"工作目录: {os.getcwd()}")

# 2. 下载测试图像
test_image = 'desk.jpg'
if not os.path.exists(test_image):
    print("\n下载测试图像...")
    try:
        urllib.request.urlretrieve(
            'https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg', 
            test_image
        )
        print(f"✅ 已下载: {test_image}")
    except Exception as e:
        print(f"⚠️  下载失败: {e}")
        print("   💡 可以使用您自己的测试图像")
        if not os.path.exists(test_image):
            print("   ⚠️  没有测试图像，demo可能失败")

# 3. 检查必要文件
config_file = "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"
weights_file = "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"

if not os.path.exists(config_file):
    print(f"❌ 配置文件不存在: {config_file}")
    os.chdir(original_dir)
    sys.exit(1)
    
if not os.path.exists(weights_file):
    print(f"❌ 权重文件不存在: {weights_file}")
    os.chdir(original_dir)
    sys.exit(1)

print(f"✅ 配置文件: {config_file}")
print(f"✅ 权重文件: {weights_file}")

# 4. 运行官方demo
print("\n" + "=" * 60)
print("运行DETIC官方demo...")
print("=" * 60)

output_file = "out_official.jpg"

cmd = [
    "python", "demo.py",
    "--config-file", config_file,
    "--input", test_image if os.path.exists(test_image) else "test.jpg",
    "--output", output_file,
    "--vocabulary", "lvis",
    "--confidence-threshold", "0.3",
    "--cpu",
    "--opts", f"MODEL.WEIGHTS={weights_file}"
]

print("命令:", " ".join(cmd))
print()

# 运行命令
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=detic_dir)
    
    if result.returncode == 0:
        print("✅ 官方demo运行成功！")
        if result.stdout:
            print("\n最后输出:")
            print(result.stdout[-1000:])
    else:
        print("❌ 官方demo运行失败")
        if result.stderr:
            print("\n错误:")
            print(result.stderr[-1000:])
        if result.stdout:
            print("\n输出:")
            print(result.stdout[-1000:])
except subprocess.TimeoutExpired:
    print("⚠️  运行超时")
except Exception as e:
    print(f"❌ 运行出错: {e}")

# 5. 显示结果
os.chdir(original_dir)

output_path = os.path.join(detic_dir, output_file)
if os.path.exists(output_path):
    print("\n" + "=" * 60)
    print("显示结果")
    print("=" * 60)
    result_img = Image.open(output_path)
    plt.figure(figsize=(15, 10))
    plt.imshow(result_img)
    plt.axis('off')
    plt.title("DETIC官方demo输出结果", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✅ 结果已显示")
else:
    print(f"⚠️  输出文件不存在: {output_path}")

