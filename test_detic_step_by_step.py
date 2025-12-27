#!/usr/bin/env python
"""
分步测试DETIC导入，找出问题所在
"""

import sys
import os

print("=" * 70)
print("DETIC分步诊断")
print("=" * 70)

# 设置路径
detic_root = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_root, "third_party", "CenterNet2")

if detic_root not in sys.path:
    sys.path.insert(0, detic_root)
if centernet_path not in sys.path:
    sys.path.insert(0, centernet_path)

print("\n1. 检查基本路径...")
print(f"   DETIC: {os.path.exists(detic_root)}")
print(f"   CenterNet2: {os.path.exists(centernet_path)}")

print("\n2. 清理模块缓存...")
modules_to_clear = [k for k in list(sys.modules.keys()) if any(x in k for x in ['detic', 'centernet', 'adet'])]
for mod in modules_to_clear:
    del sys.modules[mod]
print(f"   清理了 {len(modules_to_clear)} 个模块")

print("\n3. 导入detectron2...")
try:
    import detectron2
    print(f"   ✅ detectron2版本: {detectron2.__version__}")
except Exception as e:
    print(f"   ❌ detectron2导入失败: {e}")
    sys.exit(1)

print("\n4. 检查注册表（导入前）...")
try:
    from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
    backbones = list(BACKBONE_REGISTRY._obj_map.keys())
    print(f"   已注册的backbone: {len(backbones)} 个")
    if 'build_mnv2_backbone' in backbones:
        print(f"   ⚠️  build_mnv2_backbone已经在注册表中！")
        print(f"   这是导致冲突的原因")
    else:
        print(f"   ✅ build_mnv2_backbone不在注册表中")
except Exception as e:
    print(f"   ⚠️  无法检查注册表: {e}")

print("\n5. 导入adet.modeling...")
try:
    import adet.modeling
    print("   ✅ adet.modeling导入成功")
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ⚠️  注册冲突（已忽略）")
    else:
        print(f"   ❌ AssertionError: {e}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n6. 检查注册表（导入adet后）...")
try:
    from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
    if 'build_mnv2_backbone' in BACKBONE_REGISTRY._obj_map:
        print(f"   ⚠️  build_mnv2_backbone现在在注册表中（由adet.modeling注册）")
except Exception as e:
    print(f"   ⚠️  无法检查注册表: {e}")

print("\n7. 导入centernet.config...")
try:
    from centernet.config import add_centernet_config
    print("   ✅ centernet.config导入成功")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n8. 导入detic.config（关键步骤）...")
try:
    from detic.config import add_detic_config
    print("   ✅ detic.config导入成功（无冲突）")
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ❌ 注册冲突发生！")
        print(f"   错误: {str(e)[:200]}")
        print(f"\n   分析:")
        print(f"   - build_mnv2_backbone已经在注册表中")
        print(f"   - detic.config导入时尝试再次注册它")
        print(f"   - 这导致了冲突")
        print(f"\n   根本原因:")
        print(f"   - adet.modeling已经注册了build_mnv2_backbone")
        print(f"   - detic/__init__.py导入.modeling.backbone时，")
        print(f"     又会导入centernet.modeling.backbone，")
        print(f"     尝试再次注册build_mnv2_backbone")
        print(f"\n   解决方案:")
        print(f"   1. 不要在导入detic.config之前导入adet.modeling")
        print(f"   2. 或者修改代码，使用直接文件加载的方式导入detic.config")
        print(f"   3. 或者在Jupyter中重启kernel，确保模块只导入一次")
        sys.exit(1)
    else:
        print(f"   ❌ AssertionError: {e}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 其他错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n9. 测试配置创建...")
try:
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    print("   ✅ 配置创建成功")
except Exception as e:
    print(f"   ❌ 配置创建失败: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有步骤通过！DETIC可以正常导入")
print("=" * 70)

