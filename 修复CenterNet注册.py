# ============================================================================
# 修复 CenterNet Proposal Generator 注册问题
# ============================================================================

import sys
import os

print("=" * 60)
print("修复 CenterNet Proposal Generator 注册")
print("=" * 60)

# 添加路径
detic_root = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_root, "third_party", "CenterNet2")

if centernet_path not in sys.path:
    sys.path.insert(0, centernet_path)

print(f"\n1. 路径设置:")
print(f"   DETIC root: {detic_root}")
print(f"   CenterNet2 path: {centernet_path}")
print(f"   ✅ 已添加到 sys.path")

# 检查 CenterNet proposal generator 是否存在
print(f"\n2. 检查 CenterNet proposal generator:")
try:
    from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
    
    # 列出所有已注册的 proposal generators
    registered = list(PROPOSAL_GENERATOR_REGISTRY._obj_map.keys())
    print(f"   已注册的 proposal generators: {registered}")
    
    if "CenterNet" in registered:
        print(f"   ✅ CenterNet 已注册")
    else:
        print(f"   ❌ CenterNet 未注册")
        print(f"   💡 需要导入 CenterNet proposal generator 模块")
        
        # 尝试导入 adet.modeling 来注册
        try:
            import adet.modeling
            print(f"   ✅ 已导入 adet.modeling")
            # 再次检查
            registered_after = list(PROPOSAL_GENERATOR_REGISTRY._obj_map.keys())
            print(f"   导入后的 proposal generators: {registered_after}")
            if "CenterNet" in registered_after:
                print(f"   ✅ CenterNet 现在已注册")
            else:
                print(f"   ⚠️  CenterNet 仍然未注册")
                print(f"   💡 可能需要手动注册 CenterNet")
        except Exception as e:
            print(f"   ❌ 导入 adet.modeling 失败: {e}")
            
except Exception as e:
    print(f"   ❌ 检查失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("检查完成")
print("=" * 60)

