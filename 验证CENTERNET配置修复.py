# ============================================================================
# 验证 CENTERNET 配置修复
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import sys
import os

print("=" * 60)
print("验证 CENTERNET 配置修复")
print("=" * 60)

# 清除缓存
if 'centernet.config' in sys.modules:
    del sys.modules['centernet.config']
if 'centernet' in sys.modules:
    del sys.modules['centernet']

# 按照官方方式
detic_path = "/home/fdse/zzy/craft/Detic"
os.chdir(detic_path)
sys.path.insert(0, 'third_party/CenterNet2/')

print(f"\n1. 路径设置:")
print(f"   当前目录: {os.getcwd()}")
print(f"   添加路径: third_party/CenterNet2/")

# 验证导入和配置
print(f"\n2. 验证配置添加:")
try:
    from detectron2.config import get_cfg
    from centernet.config import add_centernet_config
    from detic.config import add_detic_config
    
    cfg = get_cfg()
    
    # 检查调用前
    print(f"   调用 add_centernet_config 前: hasattr(cfg.MODEL, 'CENTERNET') = {hasattr(cfg.MODEL, 'CENTERNET')}")
    
    # 添加配置
    add_centernet_config(cfg)
    
    # 检查调用后
    print(f"   调用 add_centernet_config 后: hasattr(cfg.MODEL, 'CENTERNET') = {hasattr(cfg.MODEL, 'CENTERNET')}")
    
    if hasattr(cfg.MODEL, 'CENTERNET'):
        print("   ✅ MODEL.CENTERNET 已添加")
        print(f"      NUM_CLASSES: {cfg.MODEL.CENTERNET.NUM_CLASSES}")
        print(f"      REG_WEIGHT: {cfg.MODEL.CENTERNET.REG_WEIGHT}")
    else:
        print("   ❌ MODEL.CENTERNET 未添加！")
        print("   💡 检查 add_centernet_config 函数是否正确")
    
    add_detic_config(cfg)
    
    # 测试合并配置文件
    config_file = os.path.join(detic_path, "configs", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
    if os.path.exists(config_file):
        print(f"\n3. 测试合并配置文件:")
        print(f"   配置文件: {config_file}")
        try:
            cfg.merge_from_file(config_file)
            print("   ✅ 配置文件合并成功！")
        except Exception as e:
            print(f"   ❌ 合并失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n3. 配置文件不存在: {config_file}")
    
    print("\n" + "=" * 60)
    print("✅ 配置验证完成！")
    print("=" * 60)
    print("\n💡 如果仍然失败:")
    print("   1. 重启 kernel (Kernel → Restart Kernel)")
    print("   2. 重新运行 Step 4")
    
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

