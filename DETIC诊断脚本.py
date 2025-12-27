
# ============================================================================
# DETIC 加载问题诊断和修复脚本
# 在 Jupyter Notebook 中运行此脚本
# ============================================================================

import sys
import os

print("=" * 60)
print("DETIC 加载问题诊断")
print("=" * 60)

# 1. 检查当前环境
print(f"
1. 当前 Python 环境:")
print(f"   Python 路径: {sys.executable}")
print(f"   Python 版本: {sys.version.split()[0]}")

# 2. 检查 NumPy
print(f"
2. NumPy 状态:")
try:
    import numpy as np
    print(f"   ✅ NumPy 版本: {np.__version__}")
    print(f"   NumPy 路径: {np.__file__}")
    if np.__version__.startswith('2.'):
        print("   ⚠️  NumPy 仍然是 2.x，需要降级")
        print("   💡 运行以下命令降级:")
        print(f"      !{sys.executable} -m pip install 'numpy<2.0'")
    else:
        print("   ✅ NumPy 版本正确（1.x）")
except ImportError:
    print("   ❌ NumPy 未安装")

# 3. 检查 detectron2
print(f"
3. detectron2 状态:")
try:
    import detectron2
    print(f"   ✅ detectron2 已安装: {detectron2.__version__}")
except ImportError:
    print("   ❌ detectron2 未安装")
    print("   💡 运行以下命令安装:")
    print(f"      !{sys.executable} -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch1.10/index.html")

# 4. 检查 DETIC 模块
print(f"
4. DETIC 模块状态:")
detic_path = "/home/fdse/zzy/craft/Detic"
if os.path.exists(detic_path):
    print(f"   ✅ Detic 目录存在: {detic_path}")
    sys.path.insert(0, detic_path)
    try:
        from detic import add_detic_config
        print("   ✅ DETIC 模块可以导入")
    except Exception as e:
        print(f"   ❌ DETIC 导入失败: {type(e).__name__}: {e}")
        if "NumPy" in str(e) or "_ARRAY_API" in str(e):
            print("      ⚠️  NumPy 兼容性问题！需要降级 NumPy 并重启 kernel")
else:
    print(f"   ❌ Detic 目录不存在: {detic_path}")

# 5. 检查权重和配置文件
print(f"
5. DETIC 权重和配置:")
weights_path = "/home/fdse/zzy/craft/Detic/models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"
config_path = "/home/fdse/zzy/craft/Detic/configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"

if os.path.exists(weights_path):
    size_mb = os.path.getsize(weights_path) / (1024*1024)
    print(f"   ✅ 权重文件存在: {size_mb:.1f} MB")
else:
    print(f"   ❌ 权重文件不存在")

if os.path.exists(config_path):
    print(f"   ✅ 配置文件存在")
else:
    print(f"   ❌ 配置文件不存在")

print("
" + "=" * 60)
print("诊断完成")
print("=" * 60)
print("
💡 如果 NumPy 是 2.x 或 detectron2 未安装，请:")
print("   1. 运行上面的安装命令")
print("   2. 重启 kernel (Kernel → Restart Kernel)")
print("   3. 重新运行此脚本验证")
print("   4. 重新运行 Step 4 初始化检测器")
