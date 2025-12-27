# ============================================================================
# 最终测试 DETIC（按照官方方式）
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import sys
import os

print("=" * 60)
print("最终测试 DETIC（按照官方方式）")
print("=" * 60)

# 按照官方 demo.py 的方式
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
        print("\n   ⚠️  模块未找到，检查路径设置")
    import traceback
    traceback.print_exc()
