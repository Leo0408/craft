# ============================================================================
# 修复 mobilenet 导入问题
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import sys
import os

print("=" * 60)
print("修复 mobilenet 导入问题")
print("=" * 60)

# 按照官方方式
detic_path = "/home/fdse/zzy/craft/Detic"
os.chdir(detic_path)
sys.path.insert(0, 'third_party/CenterNet2/')

print(f"\n1. 路径设置:")
print(f"   当前目录: {os.getcwd()}")
print(f"   添加路径: third_party/CenterNet2/")

# 验证导入
print(f"\n2. 验证导入:")
try:
    from centernet.config import add_centernet_config
    print("   ✅ from centernet.config import add_centernet_config 成功")
    
    # 测试 mobilenet（关键）
    from centernet.modeling.backbone.mobilenet import build_mnv2_backbone
    print("   ✅ mobilenet 可以导入")
    
    # 测试 bifpn
    from centernet.modeling.backbone.bifpn import BiFPN
    print("   ✅ BiFPN 可以导入")
    
    # 测试 DETIC
    from detic import add_detic_config
    print("   ✅ from detic import add_detic_config 成功")
    
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    print("   ✅ 配置添加成功")
    
    print("\n" + "=" * 60)
    print("✅ 所有测试成功！")
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
        print("   💡 检查符号链接是否正确")
    import traceback
    traceback.print_exc()

