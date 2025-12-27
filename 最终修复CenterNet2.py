# ============================================================================
# 最终修复 CenterNet2 导入问题（在 Jupyter Notebook 中运行）
# ============================================================================

import sys
import os

print("=" * 60)
print("最终修复 CenterNet2 导入")
print("=" * 60)

centernet_path = "/home/fdse/zzy/craft/Detic/third_party/CenterNet2"

# 1. 确保 centernet 目录结构正确
print(f"\n1. 检查 centernet 目录结构:")
centernet_dir = os.path.join(centernet_path, "centernet")
if not os.path.exists(centernet_dir):
    os.makedirs(centernet_dir, exist_ok=True)
    print(f"   ✅ 创建 centernet 目录")

# 确保子目录存在
for subdir in ["config", "modeling", "modeling/backbone"]:
    subdir_path = os.path.join(centernet_dir, subdir)
    if not os.path.exists(subdir_path):
        os.makedirs(subdir_path, exist_ok=True)
        print(f"   ✅ 创建 {subdir} 目录")

# 2. 检查关键文件
print(f"\n2. 检查关键文件:")
files_to_check = [
    "centernet/config/__init__.py",
    "centernet/modeling/backbone/fpn_p5.py",
    "centernet/modeling/backbone/bifpn.py",
]

for file_path in files_to_check:
    full_path = os.path.join(centernet_path, file_path)
    if os.path.exists(full_path):
        print(f"   ✅ {file_path}")
    else:
        print(f"   ❌ {file_path} 不存在")

# 3. 添加路径并验证
print(f"\n3. 验证导入:")
sys.path.insert(0, centernet_path)

try:
    # 只导入 config，不导入 modeling（避免注册冲突）
    from centernet.config import add_centernet_config, get_cfg
    print("   ✅ from centernet.config import add_centernet_config, get_cfg 成功")
    
    # 测试配置
    from detectron2.config import get_cfg as d2_get_cfg
    cfg = d2_get_cfg()
    add_centernet_config(cfg)
    print("   ✅ add_centernet_config 可以调用")
    
    # 测试 backbone 导入（延迟导入，避免注册冲突）
    print("\n   测试 backbone 导入（延迟）...")
    from centernet.modeling.backbone.fpn_p5 import LastLevelP6P7_P5
    print("   ✅ LastLevelP6P7_P5 可以导入")
    
    from centernet.modeling.backbone.bifpn import BiFPN
    print("   ✅ BiFPN 可以导入")
    
    print("\n" + "=" * 60)
    print("✅ CenterNet2 修复成功！")
    print("=" * 60)
    
except Exception as e:
    print(f"   ❌ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# 4. 验证 DETIC 导入（不导入 modeling，避免注册冲突）
print(f"\n4. 验证 DETIC 导入:")
detic_path = "/home/fdse/zzy/craft/Detic"
sys.path.insert(0, detic_path)

try:
    # 先导入 config
    from detic import add_detic_config
    print("   ✅ from detic import add_detic_config 成功")
    
    # 测试配置
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
    import traceback
    traceback.print_exc()
    
    if "already registered" in str(e) or "FCOS" in str(e):
        print("\n💡 如果看到注册冲突错误:")
        print("   1. 重启 kernel (Kernel → Restart Kernel)")
        print("   2. 确保没有重复导入 adet 或 centernet")
        print("   3. 重新运行此脚本")

