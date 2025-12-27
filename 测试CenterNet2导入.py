# ============================================================================
# 测试 CenterNet2 导入（在 Jupyter Notebook 的 conda reflect_env 中运行）
# 这个脚本会测试导入，但不会触发注册冲突
# ============================================================================

import sys
import os

print("=" * 60)
print("测试 CenterNet2 导入（避免注册冲突）")
print("=" * 60)

centernet_path = "/home/fdse/zzy/craft/Detic/third_party/CenterNet2"
detic_path = "/home/fdse/zzy/craft/Detic"

sys.path.insert(0, centernet_path)
sys.path.insert(0, detic_path)

# 1. 测试 config 导入
print(f"\n1. 测试 config 导入:")
try:
    from centernet.config import add_centernet_config, get_cfg
    print("   ✅ from centernet.config import add_centernet_config, get_cfg 成功")
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 2. 测试 backbone 模块导入（关键测试）
print(f"\n2. 测试 backbone 模块导入:")
print("   注意：这可能会触发注册，如果看到 FCOS 重复注册错误，")
print("   说明之前已经导入过 adet，需要重启 kernel")

try:
    # 直接导入 fpn_p5（不通过 __init__.py）
    from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
    print("   ✅ LastLevelP6P7_P5 可以导入")
    
    # 导入 BiFPN
    from centernet.modeling.backbone.bifpn import BiFPN
    print("   ✅ BiFPN 可以导入")
    
    # 通过 __init__.py 导入
    from centernet.modeling.backbone import LastLevelP6P7_P5 as L1, BiFPN as B1
    print("   ✅ 通过 __init__.py 导入成功")
    
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n   ⚠️  注册冲突错误！")
        print("   💡 解决方案:")
        print("      1. 重启 kernel (Kernel → Restart Kernel)")
        print("      2. 确保没有在其他地方导入 adet")
        print("      3. 重新运行此脚本")
    import traceback
    traceback.print_exc()

# 3. 测试 DETIC 导入（完整测试）
print(f"\n3. 测试 DETIC 导入:")
try:
    from detic import add_detic_config
    print("   ✅ from detic import add_detic_config 成功")
    
    # 测试配置
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_detic_config(cfg)
    print("   ✅ add_detic_config 可以调用")
    
    print("\n" + "=" * 60)
    print("✅ 所有导入测试成功！")
    print("=" * 60)
    print("\n💡 下一步:")
    print("   重新运行 Step 4 (初始化 DETIC + CLIP 检测器)")
    print("   应该看到: ✅ DETIC model loaded")
    
except Exception as e:
    print(f"   ❌ DETIC 导入失败: {type(e).__name__}: {e}")
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n   ⚠️  注册冲突错误！")
        print("   💡 这通常意味着:")
        print("      1. adet 包已经被导入过（可能在其他地方）")
        print("      2. 需要重启 kernel 清除已注册的模块")
        print("      3. 确保没有重复导入")
    import traceback
    traceback.print_exc()

