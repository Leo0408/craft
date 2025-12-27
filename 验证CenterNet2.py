# ============================================================================
# 验证 CenterNet2 导入（在 Jupyter Notebook 中运行）
# ============================================================================

import sys
import os

print("=" * 60)
print("验证 CenterNet2 导入")
print("=" * 60)

# 1. 添加路径
detic_path = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_path, "third_party", "CenterNet2")

sys.path.insert(0, centernet_path)
sys.path.insert(0, detic_path)

print(f"\n1. 路径设置:")
print(f"   CenterNet2: {centernet_path}")
print(f"   Detic: {detic_path}")

# 2. 检查符号链接
print(f"\n2. 检查符号链接:")
centernet_link = os.path.join(centernet_path, "centernet")
if os.path.exists(centernet_link):
    if os.path.islink(centernet_link):
        print(f"   ✅ 符号链接存在: centernet -> {os.readlink(centernet_link)}")
    else:
        print(f"   ✅ centernet 目录存在")
else:
    print(f"   ❌ centernet 不存在，请运行修复脚本")

# 3. 验证 CenterNet2 导入
print(f"\n3. 验证 CenterNet2 导入:")
try:
    from centernet.config import add_centernet_config
    print("   ✅ from centernet.config import add_centernet_config 成功")
    
    # 测试配置
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    print("   ✅ add_centernet_config 可以调用")
    
except Exception as e:
    print(f"   ❌ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("\n💡 如果失败，请:")
    print("   1. 运行: exec(open('修复CenterNet2导入.py').read())")
    print("   2. 检查 NumPy 版本（应该是 1.x）")
    print("   3. 重启 kernel 后重试")

# 4. 验证 DETIC 导入
print(f"\n4. 验证 DETIC 导入:")
try:
    from detic import add_detic_config
    print("   ✅ from detic import add_detic_config 成功")
    
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

