#!/usr/bin/env python
"""
在终端测试DETIC加载，模拟Jupyter notebook环境
用于调试注册冲突问题

使用方法：
    source /home/fdse/anaconda3/etc/profile.d/conda.sh
    conda activate reflect_env
    cd /home/fdse/zzy/craft
    python test_detic_notebook_fix.py
"""

import sys
import os

print("=" * 60)
print("DETIC加载测试（模拟Notebook环境）")
print("=" * 60)

# 设置路径（模拟notebook中的环境）
detic_root = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_root, "third_party", "CenterNet2")

if detic_root not in sys.path:
    sys.path.insert(0, detic_root)
if centernet_path not in sys.path:
    sys.path.insert(0, centernet_path)

print(f"\n1. 路径设置:")
print(f"   DETIC: {detic_root}")
print(f"   CenterNet2: {centernet_path}")

# 2. 清理模块缓存（模拟notebook重启kernel）
print("\n2. 清理模块缓存...")
modules_to_clear = [
    'detic', 'centernet', 'adet',
    'detic.config', 'detic.modeling',
    'centernet.config', 'centernet.modeling',
    'adet.modeling',
    'craft.perception.detic_clip_detector'
]
cleared = []
for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]
        cleared.append(mod)

if cleared:
    print(f"   ✅ 已清理 {len(cleared)} 个模块")
else:
    print("   ℹ️  没有需要清理的模块")

# 3. 导入adet.modeling（可能触发注册冲突）
print("\n3. 导入adet.modeling...")
try:
    import adet.modeling
    print("   ✅ adet.modeling导入成功")
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ℹ️  注册冲突（已忽略）: {str(e)[:80]}...")
        # 模块应该已经加载
        if 'adet.modeling' in sys.modules:
            print("   ✅ adet.modeling已在sys.modules中")
    else:
        print(f"   ❌ 意外的AssertionError: {e}")
        sys.exit(1)

# 4. 导入centernet.config
print("\n4. 导入centernet.config...")
try:
    from centernet.config import add_centernet_config
    print("   ✅ centernet.config导入成功")
except Exception as e:
    print(f"   ❌ centernet.config导入失败: {e}")
    sys.exit(1)

# 5. 导入detic.config（关键步骤 - 这里会触发注册冲突）
print("\n5. 导入detic.config（可能触发注册冲突）...")
add_detic_config = None
try:
    from detic.config import add_detic_config
    print("   ✅ detic.config导入成功（无冲突）")
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ℹ️  注册冲突: build_mnv2_backbone已经注册")
        print("   ⚠️  这通常发生在notebook环境中，因为模块已经加载过")
        
        # 检查detic.config是否在sys.modules中
        if 'detic.config' in sys.modules:
            print("   ✅ detic.config在sys.modules中，尝试获取函数...")
            try:
                add_detic_config = sys.modules['detic.config'].add_detic_config
                print("   ✅ 成功从缓存获取add_detic_config")
            except AttributeError:
                print("   ❌ detic.config模块中没有add_detic_config属性")
                print("   💡 建议：重启Jupyter kernel，然后重新运行所有cells")
                sys.exit(1)
        else:
            print("   ❌ detic.config不在sys.modules中")
            print("   💡 这意味着导入失败，模块未加载")
            print("   💡 建议：重启Jupyter kernel，然后重新运行所有cells")
            sys.exit(1)
    else:
        print(f"   ❌ 意外的AssertionError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
except Exception as e:
    print(f"   ❌ 其他错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

if add_detic_config is None:
    print("   ❌ 无法获取add_detic_config函数")
    sys.exit(1)

# 6. 注册CenterNet
print("\n6. 注册CenterNet proposal generator...")
try:
    from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
    from adet.modeling.fcos import FCOS
    
    if "CenterNet" not in PROPOSAL_GENERATOR_REGISTRY._obj_map:
        class CenterNet(FCOS):
            pass
        CenterNet.__name__ = "CenterNet"
        PROPOSAL_GENERATOR_REGISTRY.register(CenterNet)
        PROPOSAL_GENERATOR_REGISTRY._obj_map["CenterNet"] = FCOS
        print("   ✅ CenterNet注册成功")
    else:
        print("   ℹ️  CenterNet已注册")
except Exception as e:
    print(f"   ⚠️  CenterNet注册失败: {e}")

# 7. 导入detic.modeling（注册CustomRCNN）
print("\n7. 导入detic.modeling（注册CustomRCNN）...")
try:
    from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
    
    if "CustomRCNN" not in META_ARCH_REGISTRY._obj_map:
        try:
            import detic.modeling.meta_arch.custom_rcnn
            print("   ✅ detic.modeling.meta_arch.custom_rcnn导入成功")
        except AssertionError as e:
            if 'already registered' in str(e):
                print(f"   ℹ️  注册冲突（已忽略）")
                # 检查是否已注册
                if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
                    print("   ✅ CustomRCNN已注册")
                else:
                    print("   ⚠️  CustomRCNN未注册，尝试完整导入...")
                    import detic.modeling
                    if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
                        print("   ✅ CustomRCNN已注册（通过完整导入）")
                    else:
                        print("   ❌ CustomRCNN仍未注册")
            else:
                raise
        except Exception as e:
            print(f"   ⚠️  导入失败: {type(e).__name__}: {e}")
    else:
        print("   ℹ️  CustomRCNN已注册")
        
    # 验证
    if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
        print("   ✅ CustomRCNN验证成功")
    else:
        print("   ❌ CustomRCNN未注册，DETIC可能无法正常工作")
except Exception as e:
    print(f"   ⚠️  错误: {type(e).__name__}: {e}")

# 8. 测试DeticClipDetector初始化
print("\n8. 测试DeticClipDetector初始化...")
try:
    # 添加craft路径
    craft_root = "/home/fdse/zzy/craft"
    if craft_root not in sys.path:
        sys.path.insert(0, craft_root)
    
    # 清理可能的缓存
    if 'craft.perception.detic_clip_detector' in sys.modules:
        del sys.modules['craft.perception.detic_clip_detector']
    
    from craft.perception.detic_clip_detector import DeticClipDetector
    
    print("   ✅ DeticClipDetector导入成功")
    print("   正在初始化（可能需要一些时间）...")
    
    detector = DeticClipDetector(
        device="cpu",
        detic_threshold=0.3,
        clip_threshold=0.25,
        use_tracking=False
    )
    
    if detector.detic_model is not None:
        print("   ✅✅✅ DETIC模型加载成功！")
        print("   ✅ DETIC可以正常使用！")
    elif detector.clip_model is not None:
        print("   ⚠️  DETIC未加载，但CLIP-only模式可用")
        print("   ⚠️  精度会略有下降，但仍可检测对象")
    else:
        print("   ❌ DETIC和CLIP都未加载")
        
except Exception as e:
    print(f"   ❌ 初始化失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
print("\n💡 如果在notebook中仍然遇到问题：")
print("   1. 重启Jupyter kernel (Kernel → Restart Kernel)")
print("   2. 重新运行所有cells（按顺序）")
print("   3. 确保Cell 5 (Step 4)中的DETECTION_METHOD设置为'detic_clip'")
print("\n💡 如果问题仍然存在，可能是notebook环境的模块缓存问题")
print("   建议：完全关闭Jupyter，重新打开notebook")

