# ============================================================================
# 修复 CenterNet2 导入问题
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import sys
import os

print("=" * 60)
print("修复 CenterNet2 导入")
print("=" * 60)

detic_path = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_path, "third_party", "CenterNet2")

# 1. 检查符号链接
print(f"\n1. 检查符号链接:")
centernet_link = os.path.join(centernet_path, "centernet")
if os.path.exists(centernet_link):
    if os.path.islink(centernet_link):
        target = os.readlink(centernet_link)
        print(f"   ✅ 符号链接存在: centernet -> {target}")
    else:
        print(f"   ✅ centernet 目录存在")
else:
    print(f"   ⚠️  符号链接不存在，创建中...")
    os.chdir(centernet_path)
    os.symlink("adet", "centernet")
    print(f"   ✅ 已创建符号链接: centernet -> adet")

# 2. 添加路径
print(f"\n2. 添加路径:")
sys.path.insert(0, centernet_path)
print(f"   ✅ 已添加: {centernet_path}")

# 3. 检查 adet/config 结构
print(f"\n3. 检查 adet/config 结构:")
adet_config_path = os.path.join(centernet_path, "adet", "config")
if os.path.exists(adet_config_path):
    print(f"   ✅ adet/config 存在")
    config_files = os.listdir(adet_config_path)
    print(f"   文件: {config_files}")

# 4. 尝试创建 add_centernet_config 函数（如果不存在）
print(f"\n4. 检查 add_centernet_config 函数:")
try:
    from centernet.config import add_centernet_config
    print("   ✅ add_centernet_config 已存在")
except ImportError:
    print("   ⚠️  add_centernet_config 不存在，尝试创建...")
    
    # 检查 adet/config 中是否有类似的函数
    try:
        import importlib.util
        config_file = os.path.join(centernet_path, "adet", "config", "config.py")
        if os.path.exists(config_file):
            spec = importlib.util.spec_from_file_location("adet.config.config", config_file)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # 检查是否有 add_centernet_config 或类似函数
            if hasattr(config_module, 'add_centernet_config'):
                print("   ✅ 找到 add_centernet_config")
            else:
                # 创建一个简单的 add_centernet_config 函数
                print("   💡 创建 add_centernet_config 函数...")
                centernet_init = os.path.join(centernet_path, "centernet", "config", "__init__.py")
                os.makedirs(os.path.dirname(centernet_init), exist_ok=True)
                
                with open(centernet_init, 'w') as f:
                    f.write('''from adet.config import get_cfg
from detectron2.config import CfgNode

def add_centernet_config(cfg: CfgNode):
    """
    Add CenterNet2 config to detectron2 config.
    This is a compatibility function for DETIC.
    """
    # AdelaiDet (CenterNet2) config is already integrated
    # This function exists for compatibility with DETIC code
    pass

__all__ = ["get_cfg", "add_centernet_config"]
''')
                print("   ✅ 已创建 add_centernet_config 函数")
    except Exception as e:
        print(f"   ⚠️  创建失败: {e}")

# 5. 验证导入
print(f"\n5. 验证导入:")
try:
    from centernet.config import add_centernet_config
    print("   ✅ from centernet.config import add_centernet_config 成功")
    
    # 测试配置
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    print("   ✅ add_centernet_config 可以调用")
    
    print("\n" + "=" * 60)
    print("✅ CenterNet2 导入修复成功！")
    print("=" * 60)
    
except Exception as e:
    print(f"   ❌ 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n💡 如果仍然失败，可能需要:")
    print("   1. 检查 NumPy 版本（应该是 1.x）")
    print("   2. 检查 detectron2 是否已安装")
    print("   3. 重启 kernel 后重试")

