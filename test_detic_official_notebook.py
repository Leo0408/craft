# ============================================================
# DETIC官方demo测试（Notebook版本 - 修复SSL问题）
# ============================================================

import os
import subprocess
from PIL import Image
import matplotlib.pyplot as plt

# 1. 切换到Detic目录
original_dir = os.getcwd()
try:
    os.chdir('Detic')
    print(f"工作目录: {os.getcwd()}")
except:
    print("⚠️  无法切换到Detic目录，请确保在正确的路径运行")
    os.chdir(original_dir)

# 2. 下载测试图像（如果不存在）- 修复SSL问题
test_image = 'desk.jpg'
if not os.path.exists(test_image):
    print("下载官方测试图像...")
    try:
        # 方法1: 绕过SSL验证
        import ssl
        import urllib.request
        ssl._create_default_https_context = ssl._create_unverified_context
        
        urllib.request.urlretrieve(
            'https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg', 
            test_image
        )
        print(f"✅ 已下载: {test_image}")
    except Exception as e1:
        print(f"⚠️  urllib下载失败: {e1}")
        # 方法2: 使用requests库
        try:
            import requests
            print("   尝试使用requests库下载...")
            response = requests.get('https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg', verify=False, timeout=10)
            if response.status_code == 200:
                with open(test_image, 'wb') as f:
                    f.write(response.content)
                print(f"✅ 使用requests下载成功: {test_image}")
            else:
                print(f"⚠️  requests下载失败，状态码: {response.status_code}")
        except Exception as e2:
            print(f"⚠️  requests下载也失败: {e2}")
            print("   💡 请手动下载测试图像或使用您自己的图像")
            print("   命令: wget https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg")
            print("   或者: curl -k https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg -o desk.jpg")

# 3. 检查必要文件
config_file = "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"
weights_file = "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"

if not os.path.exists(config_file):
    print(f"❌ 配置文件不存在: {config_file}")
    os.chdir(original_dir)
    
if not os.path.exists(weights_file):
    print(f"❌ 权重文件不存在: {weights_file}")
    os.chdir(original_dir)

# 4. 运行官方demo
print("\n" + "=" * 60)
print("运行DETIC官方demo...")
print("=" * 60)

output_file = "out_official.jpg"

# 构建命令（使用列表格式，更清晰）
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

try:
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([
        os.path.abspath('.'),
        os.path.abspath('third_party/CenterNet2'),
        env.get('PYTHONPATH', '')
    ])
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env, cwd=os.getcwd())
    
    if result.returncode == 0:
        print("✅ 官方demo运行成功！")
        if result.stdout:
            print("\n最后100行输出:")
            print(result.stdout[-2000:])  # 显示最后2000字符
    else:
        print("❌ 官方demo运行失败")
        if result.stderr:
            print("\n错误输出:")
            print(result.stderr[-2000:])
        if result.stdout:
            print("\n标准输出:")
            print(result.stdout[-2000:])
except subprocess.TimeoutExpired:
    print("⚠️  运行超时（可能需要更长时间，特别是第一次运行）")
except Exception as e:
    print(f"❌ 运行出错: {e}")
    import traceback
    traceback.print_exc()

# 5. 显示结果
os.chdir(original_dir)

output_path = f'Detic/{output_file}'
if os.path.exists(output_path):
    print("\n" + "=" * 60)
    print("显示结果")
    print("=" * 60)
    try:
        result_img = Image.open(output_path)
        plt.figure(figsize=(15, 10))
        plt.imshow(result_img)
        plt.axis('off')
        plt.title("DETIC官方demo输出结果", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        print("✅ 结果已显示")
    except Exception as e:
        print(f"⚠️  显示图像失败: {e}")
else:
    print(f"⚠️  输出文件不存在: {output_path}")
    print("   demo可能失败或还在运行中")

