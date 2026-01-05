#!/usr/bin/env python
"""
测试CenterNet注册是否正确工作
"""

import sys
import os

# 设置路径
detic_root = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_root, "third_party", "CenterNet2")

if detic_root not in sys.path:
    sys.path.insert(0, detic_root)
if centernet_path not in sys.path:
    sys.path.insert(0, centernet_path)

print("=" * 70)
print("测试CenterNet注册流程")
print("=" * 70)

# 清理模块缓存
print("\n1. 清理模块缓存...")
modules_to_clear = [k for k in list(sys.modules.keys()) if any(x in k for x in ['detic', 'centernet', 'adet'])]
for mod in modules_to_clear:
    del sys.modules[mod]
print(f"   清理了 {len(modules_to_clear)} 个模块")

# 按照正确的顺序导入
print("\n2. 导入centernet.config...")
try:
    from centernet.config import add_centernet_config
    print("   ✅ centernet.config导入成功")
except Exception as e:
    print(f"   ❌ 失败: {e}")
    sys.exit(1)

print("\n3. 导入detic.config...")
try:
    from detic.config import add_detic_config
    print("   ✅ detic.config导入成功")
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ⚠️  注册冲突: {e}")
        print("   尝试从缓存获取...")
        if 'detic.config' in sys.modules:
            add_detic_config = sys.modules['detic.config'].add_detic_config
            print("   ✅ 从缓存获取成功")
        else:
            print("   ❌ 缓存中也没有")
            sys.exit(1)
    else:
        print(f"   ❌ AssertionError: {e}")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 失败: {e}")
    sys.exit(1)

print("\n4. 导入adet.modeling...")
adet_imported = False
try:
    if 'adet.modeling' not in sys.modules:
        import adet.modeling
        print("   ✅ adet.modeling导入成功")
        adet_imported = True
    else:
        print("   ℹ️  adet.modeling已导入")
        adet_imported = True
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ⚠️  注册冲突（已忽略）: {e}")
        # 检查FCOS是否可用
        try:
            from adet.modeling.fcos import FCOS
            print("   ✅ FCOS仍然可用")
            adet_imported = True
        except:
            print("   ❌ FCOS不可用")
    else:
        print(f"   ❌ AssertionError: {e}")
except Exception as e:
    print(f"   ❌ 失败: {e}")

if not adet_imported:
    print("   ❌ adet.modeling未成功导入，无法注册CenterNet")
    sys.exit(1)

print("\n5. 注册CenterNet...")
try:
    from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
    
    if "CenterNet" not in PROPOSAL_GENERATOR_REGISTRY._obj_map:
        from adet.modeling.fcos import FCOS
        
        class CenterNet(FCOS):
            pass
        CenterNet.__name__ = "CenterNet"
        PROPOSAL_GENERATOR_REGISTRY.register(CenterNet)
        PROPOSAL_GENERATOR_REGISTRY._obj_map["CenterNet"] = FCOS
        print("   ✅ CenterNet注册成功")
        
        # 验证
        if "CenterNet" in PROPOSAL_GENERATOR_REGISTRY._obj_map:
            print("   ✅ 验证：CenterNet在注册表中")
        else:
            print("   ❌ 验证失败：CenterNet不在注册表中")
    else:
        print("   ℹ️  CenterNet已注册")
        
except Exception as e:
    print(f"   ❌ 注册失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ 所有步骤通过！CenterNet注册成功")
print("=" * 70)

