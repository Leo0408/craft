# ============================================================================
# 最终验证 DETIC 完整导入（在 Jupyter Notebook 的 conda reflect_env 中运行）
# ============================================================================

import sys
import os

print("=" * 60)
print("最终验证 DETIC 完整导入")
print("=" * 60)

# 清除所有相关缓存
modules_to_clear = [
    'centernet',
    'centernet.config',
    'centernet.modeling',
    'centernet.modeling.backbone',
    'centernet.modeling.backbone.fpn_p5',
    'centernet.modeling.backbone.bifpn',
    'detic',
    'detic.modeling',
    'detic.modeling.backbone',
    'detic.modeling.backbone.swintransformer',
    'detic.modeling.backbone.timm',
]

for mod in modules_to_clear:
    if mod in sys.modules:
        del sys.modules[mod]
        print(f"   ✅ 清除缓存: {mod}")

# 按照官方方式
detic_path = "/home/fdse/zzy/craft/Detic"
os.chdir(detic_path)
sys.path.insert(0, 'third_party/CenterNet2/')

print(f"\n1. 路径设置:")
print(f"   当前目录: {os.getcwd()}")
print(f"   添加路径: third_party/CenterNet2/")

# 验证文件存在
print(f"\n2. 检查关键文件:")
files_to_check = [
    "centernet/modeling/backbone/fpn_p5.py",
    "centernet/modeling/backbone/__init__.py",
    "centernet/modeling/backbone/bifpn.py",
    "centernet/config/__init__.py",
]

for f in files_to_check:
    full_path = os.path.join(detic_path, "third_party/CenterNet2", f)
    if os.path.exists(full_path):
        print(f"   ✅ {f}")
    else:
        print(f"   ❌ {f} 不存在")

# 验证导入
print(f"\n3. 验证导入:")
try:
    # 1. CenterNet2 config
    from centernet.config import add_centernet_config
    print("   ✅ from centernet.config import add_centernet_config 成功")
    
    # 2. Backbone modules
    from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
    print("   ✅ from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5 成功")
    
    from centernet.modeling.backbone.bifpn import BiFPN
    print("   ✅ from centernet.modeling.backbone.bifpn import BiFPN 成功")
    
    # 3. DETIC
    from detic.config import add_detic_config
    print("   ✅ from detic.config import add_detic_config 成功")
    
    # 4. 配置测试
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    print("   ✅ 配置添加成功")
    
    if hasattr(cfg.MODEL, 'CENTERNET'):
        print("   ✅ MODEL.CENTERNET 已添加")
    
    print("\n" + "=" * 60)
    print("✅ 所有导入验证成功！")
    print("=" * 60)
    print("\n💡 下一步:")
    print("   重新运行 Step 4 (初始化 DETIC + CLIP 检测器)")
    print("   应该看到: ✅ DETIC model loaded")
    
except Exception as e:
    print(f"   ❌ 失败: {type(e).__name__}: {e}")
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n   ⚠️  注册冲突！必须重启 kernel")
    elif "No module named" in str(e):
        print(f"\n   ⚠️  模块未找到: {e}")
        print("   💡 检查文件是否存在")
    import traceback
    traceback.print_exc()

