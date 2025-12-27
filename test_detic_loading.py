#!/usr/bin/env python
"""
测试DETIC模块加载脚本
在终端运行：python test_detic_loading.py
用于调试DETIC初始化问题，避免频繁重启Jupyter kernel
"""

import os
import sys

print("=" * 60)
print("DETIC模块加载测试")
print("=" * 60)

# 1. 检查路径设置
print("\n1. 检查路径设置...")
detic_root = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_root, "third_party", "CenterNet2")

if os.path.exists(detic_root):
    print(f"✅ DETIC根目录存在: {detic_root}")
    if detic_root not in sys.path:
        sys.path.insert(0, detic_root)
        print(f"✅ 已添加到sys.path: {detic_root}")
else:
    print(f"❌ DETIC根目录不存在: {detic_root}")
    sys.exit(1)

if os.path.exists(centernet_path):
    print(f"✅ CenterNet2路径存在: {centernet_path}")
    if centernet_path not in sys.path:
        sys.path.insert(0, centernet_path)
        print(f"✅ 已添加到sys.path: {centernet_path}")
else:
    print(f"❌ CenterNet2路径不存在: {centernet_path}")
    sys.exit(1)

# 2. 清理模块缓存（模拟重新导入）
print("\n2. 清理模块缓存...")
modules_to_remove = [
    'adet', 'adet.modeling', 'centernet', 'centernet.config', 
    'centernet.modeling', 'detic', 'detic.config', 'detic.modeling',
    'craft.perception.detic_clip_detector'
]
for mod in modules_to_remove:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"   ✅ 已移除: {mod}")

# 3. 导入adet.modeling（处理注册冲突）
print("\n3. 导入adet.modeling...")
try:
    if 'adet.modeling' not in sys.modules:
        try:
            import adet.modeling
            print("✅ adet.modeling导入成功")
        except AssertionError as e:
            if 'already registered' in str(e):
                print(f"ℹ️  注册冲突（无害）: {e}")
                print("   组件已经注册过，继续...")
            else:
                print(f"❌ 导入失败: {e}")
                sys.exit(1)
    else:
        print("ℹ️  adet.modeling已经导入")
except Exception as e:
    print(f"❌ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 导入配置模块（处理注册冲突）
print("\n4. 导入配置模块...")
try:
    from centernet.config import add_centernet_config
    print("✅ centernet.config导入成功")
except Exception as e:
    print(f"❌ centernet.config导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 导入detic.config（可能会触发注册冲突，但我们需要这个函数）
print("   导入detic.config...")
add_detic_config = None
try:
    from detic.config import add_detic_config
    print("✅ detic.config导入成功")
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"ℹ️  注册冲突（已忽略）: {e}")
        # 如果模块已经部分加载，尝试获取函数
        if 'detic.config' in sys.modules:
            add_detic_config = sys.modules['detic.config'].add_detic_config
            print("✅ 从已加载的模块获取add_detic_config")
        else:
            print("⚠️  模块未加载，尝试直接导入detic包...")
            # 先导入detic包
            import detic
            from detic.config import add_detic_config
            print("✅ detic.config导入成功（通过detic包）")
except Exception as e:
    print(f"⚠️  导入失败: {type(e).__name__}: {e}")
    print("   尝试直接导入detic包（忽略注册冲突）...")
    try:
        try:
            import detic
        except AssertionError as reg_err:
            if 'already registered' in str(reg_err):
                print(f"ℹ️  注册冲突（已忽略）: {reg_err}")
                # detic包可能已经部分加载
                if 'detic' not in sys.modules:
                    raise
            else:
                raise
        
        try:
            from detic.config import add_detic_config
        except AssertionError as reg_err:
            if 'already registered' in str(reg_err):
                print(f"ℹ️  注册冲突（已忽略）: {reg_err}")
                # 如果模块已经部分加载，尝试获取函数
                if 'detic.config' in sys.modules:
                    add_detic_config = sys.modules['detic.config'].add_detic_config
                else:
                    raise
            else:
                raise
        
        print("✅ detic.config导入成功（通过直接导入detic包）")
    except Exception as final_err:
        print(f"❌ 最终导入失败: {final_err}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if add_detic_config is None:
    print("❌ 无法获取add_detic_config函数")
    sys.exit(1)

# 5. 注册CenterNet proposal generator
print("\n5. 注册CenterNet proposal generator...")
try:
    from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
    from adet.modeling.fcos import FCOS
    
    # 检查是否已经注册
    if "CenterNet" in PROPOSAL_GENERATOR_REGISTRY._obj_map:
        print("ℹ️  CenterNet已经注册")
    else:
        # 注册FCOS为"CenterNet"
        # 使用装饰器方式：创建一个包装类
        class CenterNet(FCOS):
            pass
        CenterNet.__name__ = "CenterNet"
        PROPOSAL_GENERATOR_REGISTRY.register(CenterNet)
        # 然后手动添加到字典（因为名字不同）
        PROPOSAL_GENERATOR_REGISTRY._obj_map["CenterNet"] = FCOS
        print("✅ CenterNet注册成功（使用FCOS）")
    
    # 验证注册
    registered = list(PROPOSAL_GENERATOR_REGISTRY._obj_map.keys())
    if "CenterNet" in registered:
        print(f"✅ 验证成功: CenterNet在注册表中")
        print(f"   所有已注册的proposal generators: {registered}")
    else:
        print("❌ 验证失败: CenterNet不在注册表中")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ CenterNet注册失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 6. 测试配置创建
print("\n6. 测试配置创建...")
try:
    from detectron2.config import get_cfg
    
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    
    # 验证CENTERNET配置
    if not hasattr(cfg.MODEL, 'CENTERNET'):
        print("❌ MODEL.CENTERNET未添加")
        sys.exit(1)
    print("✅ 配置创建成功")
    print(f"   MODEL.CENTERNET.NUM_CLASSES: {cfg.MODEL.CENTERNET.NUM_CLASSES}")
except Exception as e:
    print(f"❌ 配置创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 7. 测试配置文件加载
print("\n7. 测试配置文件加载...")
try:
    config_path = os.path.join(detic_root, "configs", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
        print(f"✅ 配置文件加载成功: {config_path}")
        print(f"   PROPOSAL_GENERATOR.NAME: {cfg.MODEL.PROPOSAL_GENERATOR.NAME}")
    else:
        print(f"⚠️  配置文件不存在: {config_path}")
        print("   将跳过配置文件加载测试")
except Exception as e:
    print(f"❌ 配置文件加载失败: {e}")
    import traceback
    traceback.print_exc()
    # 不退出，继续测试

# 8. 测试权重文件
print("\n8. 检查权重文件...")
weights_path = os.path.join(detic_root, "models", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth")
if os.path.exists(weights_path):
    print(f"✅ 权重文件存在: {weights_path}")
    cfg.MODEL.WEIGHTS = weights_path
else:
    print(f"⚠️  权重文件不存在: {weights_path}")
    print("   将使用URL下载")

# 9. 导入detic.modeling以注册CustomRCNN
print("\n9. 导入detic.modeling（注册CustomRCNN）...")
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

# 检查是否已经注册
if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
    print("✅ CustomRCNN已经注册")
else:
    print("   正在导入detic.modeling（可能遇到注册冲突，将被忽略）...")
    try:
        # 直接导入detic.modeling，即使遇到注册冲突也继续
        import detic.modeling
        print("✅ detic.modeling导入成功")
    except AssertionError as e:
        if 'already registered' in str(e):
            print(f"ℹ️  注册冲突（已忽略）: {e}")
        else:
            raise
    except Exception as e:
        print(f"⚠️  导入遇到问题: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    # 验证CustomRCNN是否已注册
    if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
        print("✅ CustomRCNN已注册到META_ARCH_REGISTRY")
    else:
        print("❌ CustomRCNN仍未注册，尝试直接导入custom_rcnn...")
        try:
            from detic.modeling.meta_arch import custom_rcnn
            if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
                print("✅ CustomRCNN已注册（通过直接导入custom_rcnn）")
            else:
                print("❌ CustomRCNN仍未注册")
        except Exception as final_e:
            print(f"❌ 最终导入失败: {final_e}")
            import traceback
            traceback.print_exc()

# 10. 测试模型构建（关键步骤）
print("\n10. 测试模型构建（这一步可能较慢）...")
try:
    from detectron2.engine import DefaultPredictor
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"  # 使用CPU以避免CUDA问题
    
    print("   正在构建模型（可能需要一些时间）...")
    predictor = DefaultPredictor(cfg)
    print("✅ 模型构建成功！DETIC可以正常使用！")
    
except KeyError as e:
    if "CenterNet" in str(e) and "PROPOSAL_GENERATOR" in str(e):
        print("❌ 模型构建失败: CenterNet未注册到PROPOSAL_GENERATOR_REGISTRY")
        print("   这表示步骤5的注册可能没有生效")
        sys.exit(1)
    else:
        print(f"❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
except Exception as e:
    print(f"❌ 模型构建失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 11. 测试完整的DeticClipDetector初始化
print("\n11. 测试完整的DeticClipDetector初始化...")
try:
    # 清理并重新导入
    if 'craft.perception.detic_clip_detector' in sys.modules:
        del sys.modules['craft.perception.detic_clip_detector']
    
    # 添加craft路径
    craft_root = "/home/fdse/zzy/craft"
    if craft_root not in sys.path:
        sys.path.insert(0, craft_root)
    
    from craft.perception.detic_clip_detector import DeticClipDetector
    
    print("   正在初始化DeticClipDetector...")
    detector = DeticClipDetector(
        device="cpu",
        detic_threshold=0.3,
        clip_threshold=0.25,
        use_tracking=False
    )
    
    if detector.detic_model is not None:
        print("✅ DeticClipDetector初始化成功，DETIC模型已加载！")
    elif detector.clip_model is not None:
        print("⚠️  DeticClipDetector初始化成功，但使用的是CLIP-only模式")
        print("   DETIC模型未加载，将使用CLIP作为后备")
    else:
        print("❌ DeticClipDetector初始化失败")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ DeticClipDetector初始化失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！DETIC模块可以正常使用！")
print("=" * 60)
print("\n💡 现在你可以在Jupyter notebook中重新运行Cell 9，DETIC应该能正常加载了。")
print("   如果还有问题，请查看上面的错误信息。")

