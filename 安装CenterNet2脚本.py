
# ============================================================================
# CenterNet2 安装脚本（DETIC 依赖）
# 在 Jupyter Notebook 的 conda reflect_env 环境中运行
# ============================================================================

import os
import sys
import subprocess

print("=" * 60)
print("CenterNet2 安装脚本")
print("=" * 60)

detic_path = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_path, "third_party", "CenterNet2")

# 1. 检查目录
print(f"
1. 检查目录:")
print(f"   Detic 路径: {detic_path}")
print(f"   CenterNet2 路径: {centernet_path}")

if not os.path.exists(detic_path):
    print(f"   ❌ Detic 目录不存在")
    exit(1)

# 2. 创建 third_party 目录
third_party_dir = os.path.join(detic_path, "third_party")
os.makedirs(third_party_dir, exist_ok=True)
print(f"   ✅ third_party 目录: {third_party_dir}")

# 3. 检查 CenterNet2 是否已存在
if os.path.exists(centernet_path):
    items = os.listdir(centernet_path)
    if len(items) > 0:
        print(f"   ✅ CenterNet2 目录已存在且非空")
        print(f"      内容: {items[:5]}...")
    else:
        print(f"   ⚠️  CenterNet2 目录存在但为空，需要克隆")
        # 删除空目录
        os.rmdir(centernet_path)
else:
    print(f"   ⚠️  CenterNet2 目录不存在，需要克隆")

# 4. 克隆 CenterNet2
if not os.path.exists(centernet_path) or len(os.listdir(centernet_path)) == 0:
    print(f"
2. 克隆 CenterNet2 仓库...")
    os.chdir(third_party_dir)
    
    # 尝试从 AdelaiDet 克隆（DETIC 使用的版本）
    print("   尝试从 AdelaiDet 克隆...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/aim-uofa/AdelaiDet.git", "CenterNet2"],
        capture_output=True,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print("   ✅ CenterNet2 克隆成功（从 AdelaiDet）")
    else:
        print(f"   ⚠️  从 AdelaiDet 克隆失败: {result.stderr[:200]}")
        print("   尝试从原始 CenterNet2 仓库克隆...")
        
        result2 = subprocess.run(
            ["git", "clone", "https://github.com/xingyizhou/CenterNet2.git"],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result2.returncode == 0:
            print("   ✅ CenterNet2 克隆成功（从原始仓库）")
        else:
            print(f"   ❌ 克隆失败: {result2.stderr[:200]}")
            print("   💡 请手动克隆:")
            print("      cd /home/fdse/zzy/craft/Detic/third_party")
            print("      git clone https://github.com/aim-uofa/AdelaiDet.git CenterNet2")
            exit(1)

# 5. 安装 CenterNet2
if os.path.exists(centernet_path):
    print(f"
3. 安装 CenterNet2...")
    os.chdir(centernet_path)
    
    # 检查是否有 setup.py
    if os.path.exists("setup.py"):
        print("   找到 setup.py，开始安装...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("   ✅ CenterNet2 安装成功")
        else:
            print(f"   ⚠️  安装输出: {result.stdout[-500:]}")
            print(f"   ⚠️  安装错误: {result.stderr[-500:]}")
            print("   💡 尝试仅添加路径...")
    else:
        print("   ⚠️  未找到 setup.py，仅添加路径")
    
    # 添加路径到 sys.path
    if centernet_path not in sys.path:
        sys.path.insert(0, centernet_path)
        print(f"   ✅ 已添加路径: {centernet_path}")

# 6. 验证安装
print(f"
4. 验证 CenterNet2 安装:")
sys.path.insert(0, centernet_path)
try:
    from centernet.config import add_centernet_config
    print("   ✅ CenterNet2 可以导入")
    
    # 测试配置
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_centernet_config(cfg)
    print("   ✅ CenterNet2 配置可以添加")
    
except Exception as e:
    print(f"   ❌ 验证失败: {type(e).__name__}: {e}")
    print("   💡 可能需要:")
    print("      1. 检查 CenterNet2 是否正确克隆")
    print("      2. 检查 detectron2 是否已安装")
    print("      3. 手动添加路径到 sys.path")

# 7. 验证 DETIC 导入
print(f"
5. 验证 DETIC 导入:")
sys.path.insert(0, detic_path)
try:
    from detic import add_detic_config
    print("   ✅ DETIC 可以导入")
    
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_detic_config(cfg)
    print("   ✅ DETIC 配置可以添加")
    
    print("
" + "=" * 60)
    print("✅ CenterNet2 和 DETIC 安装成功！")
    print("=" * 60)
    print("
💡 下一步:")
    print("   1. 重启 kernel (Kernel → Restart Kernel)")
    print("   2. 重新运行 Step 4 (初始化 DETIC + CLIP 检测器)")
    print("   3. 应该看到: ✅ DETIC model loaded")
    
except Exception as e:
    print(f"   ❌ DETIC 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
