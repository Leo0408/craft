
# 快速检查 DETIC 导入状态
import sys
import os

print("=" * 60)
print("快速检查 DETIC 状态")
print("=" * 60)

# 检查 NumPy
import numpy as np
print(f"NumPy 版本: {np.__version__}")
if np.__version__.startswith('2.'):
    print("⚠️  NumPy 仍然是 2.x，请重启 kernel")
else:
    print("✅ NumPy 版本正确")

# 检查 DETIC
detic_path = "/home/fdse/zzy/craft/Detic"
if os.path.exists(detic_path):
    sys.path.insert(0, detic_path)
    try:
        from detic import add_detic_config
        print("✅ DETIC 模块可以导入")
    except Exception as e:
        print(f"❌ DETIC 导入失败: {type(e).__name__}")
        if "NumPy" in str(e) or "_ARRAY_API" in str(e):
            print("   ⚠️  NumPy 兼容性问题，请重启 kernel")
else:
    print("❌ Detic 目录不存在")

# 检查检测器状态
try:
    if 'detector' in globals():
        print(f"
检测器类型: {type(detector).__name__}")
        if hasattr(detector, 'detic_available'):
            print(f"DETIC 可用: {detector.detic_available}")
        if hasattr(detector, 'detic_model'):
            print(f"DETIC 模型: {'已加载' if detector.detic_model is not None else '未加载'}")
    else:
        print("
⚠️  检测器未初始化，请先运行 Step 4")
except Exception as e:
    print(f"
检查检测器时出错: {e}")
