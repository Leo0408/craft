#!/usr/bin/env python
"""
DETIC完整安装和配置诊断脚本
在终端运行此脚本来排查DETIC安装问题

使用方法：
    source /home/fdse/anaconda3/etc/profile.d/conda.sh
    conda activate reflect_env
    cd /home/fdse/zzy/craft
    python diagnose_detic_installation.py
"""

import sys
import os
import importlib

print("=" * 70)
print("DETIC完整安装和配置诊断")
print("=" * 70)

# 1. 检查基本路径
print("\n" + "=" * 70)
print("1. 检查DETIC路径和文件")
print("=" * 70)

detic_root = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_root, "third_party", "CenterNet2")

paths_to_check = [
    ("DETIC根目录", detic_root),
    ("CenterNet2路径", centernet_path),
    ("detic/config.py", os.path.join(detic_root, "detic", "config.py")),
    ("detic/__init__.py", os.path.join(detic_root, "detic", "__init__.py")),
    ("configs目录", os.path.join(detic_root, "configs")),
    ("models目录", os.path.join(detic_root, "models")),
]

all_paths_ok = True
for name, path in paths_to_check:
    exists = os.path.exists(path)
    status = "✅" if exists else "❌"
    print(f"   {status} {name}: {path}")
    if not exists:
        all_paths_ok = False

if not all_paths_ok:
    print("\n⚠️  某些路径不存在，请检查DETIC是否正确克隆到指定位置")
    sys.exit(1)

# 2. 检查权重文件
print("\n" + "=" * 70)
print("2. 检查模型权重文件")
print("=" * 70)

weights_file = os.path.join(detic_root, "models", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth")
if os.path.exists(weights_file):
    size_mb = os.path.getsize(weights_file) / (1024 * 1024)
    print(f"   ✅ 权重文件存在: {weights_file}")
    print(f"      大小: {size_mb:.1f} MB")
else:
    print(f"   ❌ 权重文件不存在: {weights_file}")
    print("   💡 需要下载权重文件或使用URL加载")

# 3. 检查配置文件
print("\n" + "=" * 70)
print("3. 检查配置文件")
print("=" * 70)

config_file = os.path.join(detic_root, "configs", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
if os.path.exists(config_file):
    print(f"   ✅ 配置文件存在: {config_file}")
else:
    print(f"   ❌ 配置文件不存在: {config_file}")

# 4. 检查Python依赖
print("\n" + "=" * 70)
print("4. 检查Python依赖包")
print("=" * 70)

required_packages = {
    "torch": "PyTorch",
    "detectron2": "Detectron2",
    "fvcore": "fvcore",
    "iopath": "iopath",
    "PIL": "Pillow",
    "numpy": "NumPy",
    "cv2": "OpenCV",
}

for module_name, package_name in required_packages.items():
    try:
        # 使用importlib.import_module更安全
        import importlib
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        print(f"   ✅ {package_name}: {version}")
    except ImportError:
        print(f"   ❌ {package_name}: 未安装")
        print(f"      💡 安装命令: pip install {package_name.lower()}")
    except Exception as e:
        print(f"   ⚠️  {package_name}: 导入时出错 ({type(e).__name__})")
        # 不打印完整traceback，避免段错误时的混乱输出

# 5. 检查sys.path设置
print("\n" + "=" * 70)
print("5. 检查sys.path配置")
print("=" * 70)

print(f"   当前工作目录: {os.getcwd()}")
if detic_root not in sys.path:
    print(f"   ⚠️  DETIC根目录未在sys.path中")
    print(f"   💡 需要添加: {detic_root}")
else:
    print(f"   ✅ DETIC根目录已在sys.path中")

if centernet_path not in sys.path:
    print(f"   ⚠️  CenterNet2路径未在sys.path中")
    print(f"   💡 需要添加: {centernet_path}")
else:
    print(f"   ✅ CenterNet2路径已在sys.path中")

# 添加路径（如果不存在）
if detic_root not in sys.path:
    sys.path.insert(0, detic_root)
    print(f"   ✅ 已添加DETIC根目录到sys.path")

if centernet_path not in sys.path:
    sys.path.insert(0, centernet_path)
    print(f"   ✅ 已添加CenterNet2路径到sys.path")

# 6. 清理模块缓存（模拟全新环境）
print("\n" + "=" * 70)
print("6. 清理模块缓存（模拟全新环境）")
print("=" * 70)

modules_to_clear = [k for k in list(sys.modules.keys()) if any(x in k for x in ['detic', 'centernet', 'adet'])]
if modules_to_clear:
    for mod in modules_to_clear:
        del sys.modules[mod]
    print(f"   ✅ 已清理 {len(modules_to_clear)} 个模块: {', '.join(modules_to_clear[:5])}...")
else:
    print("   ℹ️  没有需要清理的模块")

# 7. 检查注册表状态（在导入之前）
print("\n" + "=" * 70)
print("7. 检查Detectron2注册表状态（导入前）")
print("=" * 70)

try:
    from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
    from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
    from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
    
    backbone_count = len(BACKBONE_REGISTRY._obj_map)
    proposal_count = len(PROPOSAL_GENERATOR_REGISTRY._obj_map)
    meta_arch_count = len(META_ARCH_REGISTRY._obj_map)
    
    print(f"   BACKBONE注册表: {backbone_count} 个组件")
    print(f"   PROPOSAL_GENERATOR注册表: {proposal_count} 个组件")
    print(f"   META_ARCH注册表: {meta_arch_count} 个组件")
    
    # 检查是否有build_mnv2_backbone
    if 'build_mnv2_backbone' in BACKBONE_REGISTRY._obj_map:
        print(f"   ⚠️  build_mnv2_backbone已经在注册表中（这可能导致冲突）")
    else:
        print(f"   ✅ build_mnv2_backbone不在注册表中")
        
except Exception as e:
    print(f"   ⚠️  无法检查注册表: {e}")

# 8. 测试导入adet.modeling
print("\n" + "=" * 70)
print("8. 测试导入adet.modeling")
print("=" * 70)

try:
    import adet.modeling
    print("   ✅ adet.modeling导入成功")
    
    # 检查FCOS是否可导入
    try:
        from adet.modeling.fcos import FCOS
        print("   ✅ FCOS类可导入")
    except Exception as e:
        print(f"   ❌ 无法导入FCOS: {e}")
        
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ⚠️  注册冲突: {str(e)[:100]}...")
        print("   💡 这通常发生在模块已被导入的情况下")
    else:
        print(f"   ❌ AssertionError: {e}")
except Exception as e:
    print(f"   ❌ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 9. 测试导入centernet.config
print("\n" + "=" * 70)
print("9. 测试导入centernet.config")
print("=" * 70)

try:
    from centernet.config import add_centernet_config
    print("   ✅ centernet.config导入成功")
    
    # 测试add_centernet_config函数
    try:
        from detectron2.config import get_cfg
        cfg = get_cfg()
        add_centernet_config(cfg)
        if hasattr(cfg.MODEL, 'CENTERNET'):
            print("   ✅ add_centernet_config函数正常工作")
        else:
            print("   ❌ add_centernet_config未正确添加配置")
    except Exception as e:
        print(f"   ❌ 测试add_centernet_config失败: {e}")
        
except Exception as e:
    print(f"   ❌ centernet.config导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 10. 测试导入detic.config（关键步骤）
print("\n" + "=" * 70)
print("10. 测试导入detic.config（关键步骤）")
print("=" * 70)

add_detic_config = None
try:
    from detic.config import add_detic_config
    print("   ✅ detic.config导入成功（无冲突）")
    
    # 测试add_detic_config函数
    try:
        from detectron2.config import get_cfg
        cfg = get_cfg()
        add_detic_config(cfg)
        print("   ✅ add_detic_config函数正常工作")
    except Exception as e:
        print(f"   ❌ 测试add_detic_config失败: {e}")
        
except AssertionError as e:
    if 'already registered' in str(e):
        print(f"   ❌ 注册冲突: {str(e)[:150]}...")
        print("\n   🔍 详细分析:")
        print("   - 错误类型: AssertionError (注册冲突)")
        print("   - 冲突组件: build_mnv2_backbone")
        print("   - 原因: 该组件已在BACKBONE注册表中注册")
        
        # 检查注册表状态
        try:
            from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
            if 'build_mnv2_backbone' in BACKBONE_REGISTRY._obj_map:
                print(f"   - 当前状态: build_mnv2_backbone已在注册表中")
                
                # 检查detic.config是否在sys.modules中
                if 'detic.config' in sys.modules:
                    print(f"   - detic.config在sys.modules中")
                    try:
                        add_detic_config = sys.modules['detic.config'].add_detic_config
                        print("   ✅ 可以从缓存获取add_detic_config")
                    except AttributeError:
                        print("   ❌ 无法从缓存获取add_detic_config")
                else:
                    print(f"   - detic.config不在sys.modules中（导入失败）")
                    print("\n   💡 解决方案:")
                    print("   1. 重启Python环境（如果是Jupyter，重启kernel）")
                    print("   2. 确保在导入detic之前没有导入其他使用相同注册表的模块")
                    print("   3. 检查是否有多个Python进程在使用相同的模块")
        except Exception as reg_e:
            print(f"   - 无法检查注册表: {reg_e}")
            
        print("\n   ⚠️  无法继续，因为detic.config导入失败")
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

# 11. 检查CustomRCNN注册
print("\n" + "=" * 70)
print("11. 检查CustomRCNN注册")
print("=" * 70)

try:
    from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
    
    if "CustomRCNN" not in META_ARCH_REGISTRY._obj_map:
        print("   ⚠️  CustomRCNN未注册，尝试导入detic.modeling...")
        try:
            import detic.modeling.meta_arch.custom_rcnn
            if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
                print("   ✅ CustomRCNN注册成功")
            else:
                print("   ❌ CustomRCNN仍未注册")
        except Exception as e:
            print(f"   ❌ 导入detic.modeling失败: {type(e).__name__}: {e}")
    else:
        print("   ✅ CustomRCNN已注册")
        
except Exception as e:
    print(f"   ⚠️  无法检查CustomRCNN: {e}")

# 12. 测试完整的模型加载流程
print("\n" + "=" * 70)
print("12. 测试完整的模型加载流程")
print("=" * 70)

try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    
    # 加载配置文件
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
        print("   ✅ 配置文件加载成功")
    else:
        print("   ⚠️  配置文件不存在，使用默认配置")
    
    # 设置权重
    if os.path.exists(weights_file):
        cfg.MODEL.WEIGHTS = weights_file
        print("   ✅ 权重文件路径设置成功")
    else:
        print("   ⚠️  权重文件不存在，将使用URL下载")
        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    
    print("   正在构建模型（可能需要一些时间）...")
    predictor = DefaultPredictor(cfg)
    print("   ✅✅✅ 模型构建成功！DETIC可以正常工作！")
    
except KeyError as e:
    if "CustomRCNN" in str(e):
        print(f"   ❌ 模型构建失败: CustomRCNN未注册")
        print("   💡 需要确保detic.modeling已成功导入")
    elif "CenterNet" in str(e) and "PROPOSAL_GENERATOR" in str(e):
        print(f"   ❌ 模型构建失败: CenterNet未注册")
        print("   💡 需要确保CenterNet proposal generator已注册")
    else:
        print(f"   ❌ 模型构建失败: {e}")
        import traceback
        traceback.print_exc()
except Exception as e:
    print(f"   ❌ 模型构建失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 总结
print("\n" + "=" * 70)
print("诊断总结")
print("=" * 70)
print("\n如果所有步骤都显示✅，说明DETIC配置正确。")
print("如果有❌，请根据上面的建议进行修复。")
print("\n如果仍然遇到注册冲突问题，建议：")
print("1. 完全重启Python环境（如果是Jupyter，重启kernel并重新运行所有cells）")
print("2. 确保没有其他Python进程在使用相同的模块")
print("3. 检查是否有多个版本的detectron2或其他依赖被安装")

