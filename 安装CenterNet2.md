# 安装 CenterNet2（DETIC 依赖）

## 🔍 问题

DETIC 需要 CenterNet2 模块，但该模块未安装，导致错误：
```
ModuleNotFoundError: No module named 'centernet'
```

## ✅ 解决方案

### 方法 1: 克隆 CenterNet2 到 DETIC 的 third_party 目录（推荐）

在 Jupyter Notebook 中运行：

```python
import os
import subprocess
import sys

detic_path = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_path, "third_party", "CenterNet2")

# 创建 third_party 目录（如果不存在）
third_party_dir = os.path.join(detic_path, "third_party")
os.makedirs(third_party_dir, exist_ok=True)

# 检查是否已存在
if os.path.exists(centernet_path):
    print(f"✅ CenterNet2 已存在: {centernet_path}")
else:
    print("克隆 CenterNet2 仓库...")
    os.chdir(third_party_dir)
    
    # 克隆仓库
    result = subprocess.run(
        ["git", "clone", "https://github.com/aim-uofa/AdelaiDet.git", "CenterNet2"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("✅ CenterNet2 克隆成功")
    else:
        print(f"⚠️  克隆失败: {result.stderr}")
        print("   尝试备用方法...")
        
        # 备用：直接克隆到 CenterNet2
        result2 = subprocess.run(
            ["git", "clone", "https://github.com/xingyizhou/CenterNet2.git"],
            capture_output=True,
            text=True
        )
        if result2.returncode == 0:
            print("✅ CenterNet2 克隆成功（备用方法）")
        else:
            print(f"❌ 克隆失败: {result2.stderr}")

# 安装 CenterNet2
if os.path.exists(centernet_path):
    print("\n安装 CenterNet2...")
    os.chdir(centernet_path)
    
    # 检查是否有 setup.py
    if os.path.exists("setup.py"):
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("✅ CenterNet2 安装成功")
        else:
            print(f"⚠️  安装输出: {result.stdout}")
            print(f"⚠️  安装错误: {result.stderr}")
    else:
        print("⚠️  未找到 setup.py，可能需要手动安装")
        print("   尝试添加路径到 sys.path...")
        
        # 添加路径
        import sys
        if centernet_path not in sys.path:
            sys.path.insert(0, centernet_path)
            print(f"✅ 已添加路径: {centernet_path}")
```

### 方法 2: 使用 pip 安装（如果可用）

```python
import sys
!{sys.executable} -m pip install centernet-ilvo
```

### 方法 3: 手动添加路径（临时方案）

如果 CenterNet2 已存在但无法导入，可以手动添加路径：

```python
import sys
import os

detic_path = "/home/fdse/zzy/craft/Detic"
centernet_path = os.path.join(detic_path, "third_party", "CenterNet2")

if os.path.exists(centernet_path):
    sys.path.insert(0, centernet_path)
    print(f"✅ 已添加 CenterNet2 路径: {centernet_path}")
    
    # 验证
    try:
        from centernet.config import add_centernet_config
        print("✅ CenterNet2 可以导入")
    except Exception as e:
        print(f"❌ 导入失败: {e}")
else:
    print(f"❌ CenterNet2 目录不存在: {centernet_path}")
```

## 🔧 修改 DETIC 导入代码

如果 CenterNet2 在 `third_party/CenterNet2` 目录中，需要在导入 DETIC 之前添加路径：

```python
import sys
import os

# 添加 DETIC 路径
detic_path = "/home/fdse/zzy/craft/Detic"
sys.path.insert(0, detic_path)

# 添加 CenterNet2 路径
centernet_path = os.path.join(detic_path, "third_party", "CenterNet2")
if os.path.exists(centernet_path):
    sys.path.insert(0, centernet_path)

# 现在可以导入 DETIC
from detic import add_detic_config
```

## 📋 完整安装步骤

1. **克隆 CenterNet2**：
   ```bash
   cd /home/fdse/zzy/craft/Detic/third_party
   git clone https://github.com/aim-uofa/AdelaiDet.git CenterNet2
   ```

2. **安装 CenterNet2**：
   ```bash
   cd CenterNet2
   pip install -e .
   ```

3. **验证安装**：
   ```python
   import sys
   sys.path.insert(0, '/home/fdse/zzy/craft/Detic/third_party/CenterNet2')
   from centernet.config import add_centernet_config
   print("✅ CenterNet2 可以导入")
   ```

4. **导入 DETIC**：
   ```python
   import sys
   sys.path.insert(0, '/home/fdse/zzy/craft/Detic')
   sys.path.insert(0, '/home/fdse/zzy/craft/Detic/third_party/CenterNet2')
   from detic import add_detic_config
   print("✅ DETIC 可以导入")
   ```

## 💡 注意事项

- CenterNet2 是 DETIC 的依赖，必须安装
- 如果使用 `pip install -e .` 安装 DETIC，可能需要手动处理 CenterNet2
- 确保 CenterNet2 路径在 DETIC 之前添加到 sys.path

