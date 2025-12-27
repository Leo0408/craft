
# ============================================================================
# DETIC 完整安装脚本
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import sys
import os
import subprocess

print("=" * 60)
print("DETIC 完整安装脚本")
print("=" * 60)

# 1. 检查当前环境
print(f"
1. 当前环境:")
print(f"   Python: {sys.executable}")
env_name = sys.executable.split('/')[-3] if 'envs' in sys.executable else 'base'
print(f"   环境: {env_name}")

# 2. 检查基础依赖
print(f"
2. 检查基础依赖:")
try:
    import numpy as np
    print(f"   ✅ NumPy: {np.__version__}")
    if np.__version__.startswith('2.'):
        print("      ⚠️  需要降级到 1.x")
except:
    print("   ❌ NumPy 未安装")

try:
    import detectron2
    print(f"   ✅ detectron2: {detectron2.__version__}")
except:
    print("   ❌ detectron2 未安装")

try:
    import torch
    print(f"   ✅ PyTorch: {torch.__version__}")
except:
    print("   ❌ PyTorch 未安装")

# 3. 安装 DETIC 包
print(f"
3. 安装 DETIC 包:")
detic_path = "/home/fdse/zzy/craft/Detic"
if os.path.exists(detic_path):
    print(f"   Detic 目录: {detic_path}")
    
    # 检查是否已安装
    try:
        import detic
        print("   ✅ DETIC 包已安装")
    except ImportError:
        print("   ⚠️  DETIC 包未安装，开始安装...")
        os.chdir(detic_path)
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ DETIC 包安装成功")
        else:
            print(f"   ⚠️  安装输出: {result.stdout}")
            print(f"   ⚠️  安装错误: {result.stderr}")
            print("   💡 尝试使用 --no-deps 选项...")
            result2 = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps"],
                capture_output=True,
                text=True
            )
            if result2.returncode == 0:
                print("   ✅ DETIC 包安装成功（跳过依赖）")
            else:
                print(f"   ❌ 安装失败: {result2.stderr}")
else:
    print(f"   ❌ Detic 目录不存在: {detic_path}")

# 4. 安装 DETIC 依赖
print(f"
4. 安装 DETIC 依赖:")
req_file = os.path.join(detic_path, "requirements.txt")
if os.path.exists(req_file):
    print(f"   安装 requirements.txt 中的依赖...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", req_file],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print("   ✅ 依赖安装成功")
    else:
        print(f"   ⚠️  部分依赖可能安装失败，继续...")

# 5. 验证安装
print(f"
5. 验证 DETIC 安装:")
sys.path.insert(0, detic_path)
try:
    from detic import add_detic_config
    print("   ✅ from detic import add_detic_config")
    
    from detic.modeling.utils import reset_cls_test
    print("   ✅ from detic.modeling.utils import reset_cls_test")
    
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_detic_config(cfg)
    print("   ✅ DETIC 配置可以添加")
    
    print("
" + "=" * 60)
    print("✅ DETIC 安装成功！")
    print("=" * 60)
    print("
💡 下一步:")
    print("   1. 重启 kernel (Kernel → Restart Kernel)")
    print("   2. 重新运行 Step 4 (初始化 DETIC + CLIP 检测器)")
    print("   3. 应该看到: ✅ DETIC model loaded")
    
except Exception as e:
    print(f"   ❌ 验证失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    print("
💡 如果仍然失败，请检查:")
    print("   1. NumPy 版本是否为 1.x")
    print("   2. detectron2 是否已安装")
    print("   3. 所有依赖是否已安装")
