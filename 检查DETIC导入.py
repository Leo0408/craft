#!/usr/bin/env python3
"""
检查 DETIC 模块是否能正常导入
在重启 kernel 后运行此脚本验证
"""

import sys
import os

print("=" * 60)
print("DETIC 导入检查")
print("=" * 60)

# 检查 NumPy 版本
try:
    import numpy as np
    print(f"✅ NumPy 版本: {np.__version__}")
    if np.__version__.startswith('2.'):
        print("⚠️  警告: NumPy 仍然是 2.x，可能需要重启 kernel")
    else:
        print("✅ NumPy 版本正确（1.x）")
except ImportError:
    print("❌ NumPy 未安装")

# 检查 detectron2
print("\n" + "=" * 60)
print("detectron2 检查")
print("=" * 60)

try:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    print("✅ detectron2 可用")
    print(f"   版本: {detectron2.__version__}")
except ImportError as e:
    print(f"❌ detectron2 不可用: {e}")
    print("   安装: pip install detectron2")

# 检查 DETIC 模块
print("\n" + "=" * 60)
print("DETIC 模块检查")
print("=" * 60)

detic_path = "/home/fdse/zzy/craft/Detic"
if os.path.exists(detic_path):
    sys.path.insert(0, detic_path)
    print(f"✅ Detic 目录存在: {detic_path}")
    
    try:
        from detic import add_detic_config
        print("✅ 成功导入: from detic import add_detic_config")
    except Exception as e:
        print(f"❌ 导入失败: {type(e).__name__}: {e}")
        if "NumPy" in str(e) or "_ARRAY_API" in str(e):
            print("   ⚠️  NumPy 兼容性问题，请重启 kernel 后重试")
    
    try:
        from detic.modeling.utils import reset_cls_test
        print("✅ 成功导入: from detic.modeling.utils import reset_cls_test")
    except Exception as e:
        print(f"❌ 导入失败: {type(e).__name__}: {e}")
else:
    print(f"❌ Detic 目录不存在: {detic_path}")

# 检查 DeticClipDetector
print("\n" + "=" * 60)
print("DeticClipDetector 检查")
print("=" * 60)

try:
    sys.path.insert(0, '/home/fdse/zzy/craft')
    from perception.detic_clip_detector import DeticClipDetector
    print("✅ 成功导入 DeticClipDetector")
    
    # 尝试初始化（不加载模型，只检查类定义）
    print("\n尝试初始化检测器（仅检查，不加载模型）...")
    try:
        detector = DeticClipDetector(
            device="cpu",
            detic_threshold=0.3,
            clip_threshold=0.25,
            use_tracking=False  # 跳过 ByteTrack
        )
        print("✅ 检测器初始化成功")
        print(f"   DETIC 可用: {detector.detic_available}")
        print(f"   CLIP 可用: {detector.clip_available}")
        print(f"   DETIC 模型: {'已加载' if detector.detic_model is not None else '未加载'}")
    except Exception as e:
        print(f"⚠️  初始化时出错: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)
print("\n💡 如果看到错误，请：")
print("   1. 重启 Jupyter kernel")
print("   2. 重新运行 Step 4（初始化检测器）")
print("   3. 重新运行 Step 6（生成 scene graph）")

