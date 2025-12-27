# ============================================================================
# 最终验证 CenterNet2 导入（在 Jupyter Notebook 的 conda reflect_env 中运行）
# ============================================================================

import sys
import os

print("=" * 60)
print("最终验证 CenterNet2 导入")
print("=" * 60)

# 添加路径
detic_path = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_path, "third_party", "CenterNet2")

sys.path.insert(0, centernet_path)
sys.path.insert(0, detic_path)

print(f"\n1. 路径已添加:")
print(f"   {centernet_path}")
print(f"   {detic_path}")

# 验证 CenterNet2 导入（不导入整个 centernet，只导入 config）
print(f"\n2. 验证 CenterNet2 config 导入:")
try:
    # 直接从 config 模块导入，避免导入整个 centernet
    from centernet.config import add_centernet_config, get_cfg
    print("   ✅ from centernet.config import add_centernet_config, get_cfg 成功")
    
    # 测试配置
    from detectron2.config import get_cfg as d2_get_cfg
    cfg = d2_get_cfg()
    add_centernet_config(cfg)
    print("   ✅ add_centernet_config 可以调用")
    
except Exception as e:
    print(f"   ❌ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
    # 尝试直接导入 adet.config.config
    print("\n   尝试直接导入 adet.config.config...")
    try:
        from adet.config.config import get_cfg as adet_get_cfg
        print("   ✅ 可以直接导入 adet.config.config.get_cfg")
        print("   💡 可能需要修改 centernet/config/__init__.py")
    except Exception as e2:
        print(f"   ❌ 也失败: {e2}")

# 验证 DETIC 导入
print(f"\n3. 验证 DETIC 导入:")
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

