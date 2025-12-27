# DETIC 完整安装指南（解决加载失败问题）

## 🔍 问题诊断

从错误信息看，DETIC 无法加载的原因可能是：

1. **DETIC 模块未正确安装** - 仅仅克隆仓库不够，需要安装
2. **依赖缺失** - DETIC 的某些依赖未安装
3. **NumPy/PyTorch 兼容性** - 虽然已安装，但可能存在版本不匹配

## ✅ 完整安装步骤

### 步骤 1: 确认 conda 环境

在 Jupyter Notebook 中运行：

```python
import sys
print(f"当前 Python: {sys.executable}")
print(f"当前环境: {sys.executable.split('/')[-3] if 'envs' in sys.executable else 'base'}")
```

确认使用的是 `reflect_env` 环境。

### 步骤 2: 安装 DETIC 包

**重要**：仅仅克隆 DETIC 仓库不够，需要安装它！

在 Jupyter Notebook 中运行：

```python
import sys
import os

# 切换到 Detic 目录
detic_path = "/home/fdse/zzy/craft/Detic"
os.chdir(detic_path)

# 安装 DETIC 包（开发模式）
!{sys.executable} -m pip install -e .

# 或者如果上面失败，尝试：
# !{sys.executable} -m pip install -e . --no-deps
```

**说明**：
- `-e` 表示可编辑模式（editable mode）
- 这样安装后，DETIC 模块才能被正确导入
- `--no-deps` 可以跳过依赖检查（如果依赖已安装）

### 步骤 3: 安装 DETIC 依赖

在 Jupyter Notebook 中运行：

```python
import sys
detic_path = "/home/fdse/zzy/craft/Detic"
req_file = os.path.join(detic_path, "requirements.txt")

# 读取 requirements.txt
with open(req_file, 'r') as f:
    reqs = f.read().strip().split('\n')

# 安装依赖（跳过已安装的）
for req in reqs:
    if req.strip() and not req.strip().startswith('#'):
        try:
            !{sys.executable} -m pip install {req.strip()}
        except:
            print(f"跳过或失败: {req}")
```

或者直接运行：

```python
import sys
!{sys.executable} -m pip install -r /home/fdse/zzy/craft/Detic/requirements.txt
```

### 步骤 4: 验证安装

在 Jupyter Notebook 中运行：

```python
import sys
import os

# 添加 Detic 到路径
detic_path = "/home/fdse/zzy/craft/Detic"
sys.path.insert(0, detic_path)

# 检查基础环境
import numpy as np
print(f"NumPy: {np.__version__}")  # 应该是 1.x

import detectron2
print(f"detectron2: {detectron2.__version__}")

import torch
print(f"PyTorch: {torch.__version__}")

# 尝试导入 DETIC
try:
    from detic import add_detic_config
    print("✅ DETIC 模块可以导入")
    
    from detic.modeling.utils import reset_cls_test
    print("✅ DETIC 工具函数可以导入")
    
    # 测试配置
    from detectron2.config import get_cfg
    cfg = get_cfg()
    add_detic_config(cfg)
    print("✅ DETIC 配置可以添加")
    
except Exception as e:
    print(f"❌ DETIC 导入失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

### 步骤 5: 重新运行 Step 4

如果验证通过，重新运行 Step 4（初始化 DETIC + CLIP 检测器）。

**期望输出**：
```
✅ DETIC model loaded  ← 关键！
✅ CLIP model loaded (ViT-B/32)
✅ DETIC + CLIP detector initialized successfully!
```

## 🔧 如果仍然失败

### 方案 A: 检查 DETIC 安装状态

在 Jupyter Notebook 中运行：

```python
import sys
!{sys.executable} -m pip list | grep -i detic
```

如果看到 `detic` 包，说明已安装。

### 方案 B: 手动安装 DETIC 依赖

DETIC 的关键依赖：
- `opencv-python`
- `timm`
- `ftfy`
- `regex`
- `fasttext`
- `scikit-learn`
- `lvis`
- `nltk`
- `CLIP` (git+https://github.com/openai/CLIP.git)

在 Jupyter Notebook 中逐个安装：

```python
import sys
deps = [
    "opencv-python",
    "timm",
    "ftfy",
    "regex",
    "fasttext",
    "scikit-learn",
    "lvis",
    "nltk",
    "git+https://github.com/openai/CLIP.git"
]

for dep in deps:
    print(f"安装 {dep}...")
    !{sys.executable} -m pip install {dep}
```

### 方案 C: 使用 conda 安装部分依赖

```python
import sys
# 某些包用 conda 安装可能更稳定
!conda install -c conda-forge opencv scikit-learn -y
```

### 方案 D: 检查 DETIC 安装路径

在 Jupyter Notebook 中运行：

```python
import sys
import site

# 检查 DETIC 是否在 site-packages 中
for path in site.getsitepackages():
    detic_path = os.path.join(path, "detic")
    if os.path.exists(detic_path):
        print(f"✅ 找到 DETIC: {detic_path}")
        break
else:
    print("❌ DETIC 未在 site-packages 中找到")
    print("   需要运行: cd Detic && pip install -e .")
```

## 📋 完整检查清单

- [ ] 在 conda reflect_env 环境中
- [ ] NumPy 版本是 1.x
- [ ] detectron2 已安装
- [ ] PyTorch 已安装
- [ ] DETIC 包已安装（`pip install -e .`）
- [ ] DETIC 依赖已安装
- [ ] DETIC 模块可以导入
- [ ] Step 4 显示 `✅ DETIC model loaded`

## 💡 关键提示

1. **必须安装 DETIC 包**：仅仅克隆仓库不够，需要运行 `pip install -e .`
2. **使用正确的环境**：确保在 conda reflect_env 环境中安装
3. **使用 `-e` 模式**：可编辑模式，修改代码后无需重新安装
4. **检查错误信息**：现在代码会显示详细的错误信息，帮助定位问题

## 🎯 预期结果

安装成功后，运行 Step 4 应该看到：

```
📁 Using local weights: /home/fdse/zzy/craft/Detic/models/...
📁 Using default config: /home/fdse/zzy/craft/Detic/configs/...
✅ DETIC model loaded
✅ CLIP model loaded (ViT-B/32)
✅ DETIC + CLIP detector initialized successfully!
```

