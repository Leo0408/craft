# ============================================================================
# 简化验证脚本（按照 DETIC 官方方式）
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import sys
import os

print("=" * 60)
print("按照 DETIC 官方方式验证")
print("=" * 60)

# 按照官方 demo.py 的方式
detic_path = "/home/fdse/zzy/craft/Detic"
os.chdir(detic_path)
sys.path.insert(0, 'third_party/CenterNet2/')

print(f"\n1. 路径设置:")
print(f"   当前目录: {os.getcwd()}")
print(f"   添加路径: third_party/CenterNet2/")

# 验证 CenterNet2 config
print(f"\n2. 验证 CenterNet2 config:")
try:
    from centernet.config import add_centernet_config
    print("   ✅ from centernet.config import add_centernet_config 成功")
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 验证 backbone（关键）
print(f"\n3. 验证 backbone 模块:")
try:
    from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
    print("   ✅ LastLevelP6P7_P5 可以导入")
    
    from centernet.modeling.backbone.bifpn import BiFPN
    print("   ✅ BiFPN 可以导入")
    
    # 通过 __init__.py 导入
    from centernet.modeling.backbone import LastLevelP6P7_P5 as L1, BiFPN as B1
    print("   ✅ 通过 __init__.py 导入成功")
    
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n   ⚠️  注册冲突！必须重启 kernel")
    import traceback
    traceback.print_exc()

# 验证 DETIC
print(f"\n4. 验证 DETIC:")
try:
    from detic import add_detic_config
    print("   ✅ from detic import add_detic_config 成功")
    
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_detic_config(cfg)
    print("   ✅ add_detic_config 可以调用")
    
    print("\n" + "=" * 60)
    print("✅ 所有验证成功！")
    print("=" * 60)
    print("\n💡 下一步:")
    print("   重新运行 Step 4 (初始化 DETIC + CLIP 检测器)")
    print("   应该看到: ✅ DETIC model loaded")
    
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n   ⚠️  注册冲突！必须重启 kernel")
    import traceback
    traceback.print_exc()

