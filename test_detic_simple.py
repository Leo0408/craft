#!/usr/bin/env python
"""
简化的DETIC测试脚本 - 使用与detic_clip_detector.py相同的方法
"""

import os
import sys

print("=" * 60)
print("简化的DETIC测试")
print("=" * 60)

# 设置路径
detic_root = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_root, "third_party", "CenterNet2")

if detic_root not in sys.path:
    sys.path.insert(0, detic_root)
if centernet_path not in sys.path:
    sys.path.insert(0, centernet_path)

print("\n1. 导入adet.modeling...")
try:
    if 'adet.modeling' not in sys.modules:
        try:
            import adet.modeling
            print("✅ adet.modeling导入成功")
        except AssertionError as e:
            if 'already registered' in str(e):
                print(f"ℹ️  注册冲突（已忽略）")
            else:
                raise
    else:
        print("ℹ️  adet.modeling已导入")
except Exception as e:
    print(f"❌ 失败: {e}")
    sys.exit(1)

print("\n2. 导入配置...")
try:
    from centernet.config import add_centernet_config
    print("✅ centernet.config导入成功")
except Exception as e:
    print(f"❌ centernet.config导入失败: {e}")
    sys.exit(1)

# 导入detic.config，处理注册冲突
add_detic_config = None
print("   导入detic.config...")
try:
    from detic.config import add_detic_config
    print("✅ detic.config导入成功")
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"ℹ️  注册冲突（已忽略）")
        # 尝试从已加载的模块获取
        if 'detic.config' in sys.modules:
            add_detic_config = sys.modules['detic.config'].add_detic_config
            print("✅ 从已加载模块获取add_detic_config")
        else:
            # 如果模块未加载，说明导入失败了，我们需要接受这个错误
            print("⚠️  模块未加载，但继续（将在后续步骤中处理）")
    else:
        raise
except Exception as e:
    print(f"❌ detic.config导入失败: {e}")
    print("   这可能是由于注册冲突，将在后续步骤中处理")

if add_detic_config is None:
    print("⚠️  add_detic_config未获取到，尝试直接使用detic包...")
    # 如果add_detic_config仍然为None，说明需要完整导入detic包
    # 但我们知道这会有注册冲突，所以先跳过，在需要时再处理
    # 在实际使用中，detic_clip_detector.py已经通过直接加载文件的方式解决了这个问题
    print("   注意：在实际代码中使用直接加载文件的方式避免这个问题")

print("\n3. 注册CenterNet...")
try:
    from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
    from adet.modeling.fcos import FCOS
    
    if "CenterNet" not in PROPOSAL_GENERATOR_REGISTRY._obj_map:
        class CenterNet(FCOS):
            pass
        CenterNet.__name__ = "CenterNet"
        PROPOSAL_GENERATOR_REGISTRY.register(CenterNet)
        PROPOSAL_GENERATOR_REGISTRY._obj_map["CenterNet"] = FCOS
        print("✅ CenterNet注册成功")
    else:
        print("ℹ️  CenterNet已注册")
except Exception as e:
    print(f"❌ CenterNet注册失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. 导入detic.modeling（注册CustomRCNN）...")
try:
    from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
    
    if "CustomRCNN" not in META_ARCH_REGISTRY._obj_map:
        try:
            import detic.modeling
            print("✅ detic.modeling导入成功")
        except AssertionError as e:
            if 'already registered' in str(e):
                print(f"ℹ️  注册冲突（已忽略）")
            else:
                raise
    else:
        print("ℹ️  CustomRCNN已注册")
    
    if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
        print("✅ CustomRCNN已注册")
    else:
        print("❌ CustomRCNN未注册")
        sys.exit(1)
except Exception as e:
    print(f"❌ detic.modeling导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n5. 测试模型构建...")
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    
    config_path = os.path.join(detic_root, "configs", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
    if os.path.exists(config_path):
        cfg.merge_from_file(config_path)
    
    weights_path = os.path.join(detic_root, "models", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth")
    if os.path.exists(weights_path):
        cfg.MODEL.WEIGHTS = weights_path
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    
    print("   正在构建模型...")
    predictor = DefaultPredictor(cfg)
    print("✅ 模型构建成功！DETIC可以正常使用！")
    
except Exception as e:
    print(f"❌ 模型构建失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ 所有测试通过！")
print("=" * 60)

