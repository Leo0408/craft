# ============================================================================
# 最终解决方案：避免注册冲突
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import sys
import os

print("=" * 60)
print("最终解决方案：避免注册冲突")
print("=" * 60)

print("\n💡 关键点：")
print("   - centernet/config/__init__.py 使用 importlib 导入，避免触发 adet/__init__.py")
print("   - centernet/__init__.py 和 centernet/modeling/__init__.py 都是空的")
print("   - 只在需要时导入 backbone 模块")

centernet_path = "/home/fdse/zzy/craft/Detic/third_party/CenterNet2"
detic_path = "/home/fdse/zzy/craft/Detic"

sys.path.insert(0, centernet_path)
sys.path.insert(0, detic_path)

# 1. 验证 config 导入（不应该触发注册）
print(f"\n1. 验证 config 导入:")
try:
    from centernet.config import add_centernet_config, get_cfg
    print("   ✅ from centernet.config import add_centernet_config, get_cfg 成功")
    
    # 测试配置
    from detectron2.config import get_cfg as d2_get_cfg
    cfg = d2_get_cfg()
    add_centernet_config(cfg)
    print("   ✅ add_centernet_config 可以调用")
    
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 2. 验证 backbone 导入（关键测试）
print(f"\n2. 验证 backbone 导入:")
print("   ⚠️  如果看到 'FCOS already registered' 错误，说明之前导入过 adet")
print("   💡 解决方案：重启 kernel")

try:
    # 直接导入 fpn_p5（不通过 __init__.py）
    from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
    print("   ✅ LastLevelP6P7_P5 可以导入")
    
    # 导入 BiFPN
    from centernet.modeling.backbone.bifpn import BiFPN
    print("   ✅ BiFPN 可以导入")
    
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n   ⚠️  注册冲突！")
        print("   💡 必须重启 kernel (Kernel → Restart Kernel)")
        print("      然后重新运行此脚本")
    import traceback
    traceback.print_exc()

# 3. 验证 DETIC 导入（完整测试）
print(f"\n3. 验证 DETIC 导入:")
try:
    from detic import add_detic_config
    print("   ✅ from detic import add_detic_config 成功")
    
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_detic_config(cfg)
    print("   ✅ add_detic_config 可以调用")
    
    print("\n" + "=" * 60)
    print("✅ 所有导入验证成功！")
    print("=" * 60)
    print("\n💡 下一步:")
    print("   重新运行 Step 4 (初始化 DETIC + CLIP 检测器)")
    print("   应该看到: ✅ DETIC model loaded")
    
except Exception as e:
    print(f"   ❌ DETIC 导入失败: {type(e).__name__}: {e}")
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n   ⚠️  注册冲突错误！")
        print("   💡 必须重启 kernel (Kernel → Restart Kernel)")
        print("      然后重新运行 Step 4")
    import traceback
    traceback.print_exc()

