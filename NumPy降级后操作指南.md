# NumPy 降级后操作指南

## ✅ 当前状态

- NumPy 已降级到 1.24.4 ✅
- 需要重启 Jupyter kernel 才能生效

## 📋 操作步骤

### 1. 重启 Jupyter Kernel

在 Jupyter Notebook 中：
- 点击菜单栏：`Kernel` → `Restart Kernel`
- 或者使用快捷键（如果有设置）

### 2. 重新运行必要的 Cell

重启后，需要按顺序重新运行：

#### Step 1: 导入库
- 运行第一个代码 cell（导入所有必要的库）

#### Step 4: 初始化 DETIC + CLIP 检测器
- 找到 Step 4 的代码 cell（DETIC + CLIP Detector Initialization）
- 重新运行该 cell
- **期望输出**：
  ```
  ✅ DETIC model loaded  ← 应该看到这个，而不是 "Full DETIC not available"
  ✅ CLIP model loaded (ViT-B/32)
  ✅ DETIC + CLIP detector initialized successfully!
  ```

#### Step 6: 生成 Scene Graph
- 重新运行 Step 6 的代码 cell
- **期望输出**：
  ```
  Frame 0 (stage 0): ✅ 4 objects, 6 relations
     Objects: coffee machine, purple cup, ...
     Relations: ...
  ```

### 3. 验证 DETIC 是否正常工作

如果 Step 4 的输出中看到：
- ✅ `✅ DETIC model loaded` → **成功！** DETIC 已正常加载
- ⚠️ `⚠️  Full DETIC not available` → **失败**，继续排查

### 4. 如果仍然失败

如果重启后仍然看到 "Full DETIC not available"，可以运行检查脚本：

```python
# 在 notebook 中运行
exec(open('检查DETIC导入.py').read())
```

或者手动检查：

```python
import numpy as np
print(f"NumPy 版本: {np.__version__}")  # 应该是 1.x

import sys
sys.path.insert(0, '/home/fdse/zzy/craft/Detic')
from detic import add_detic_config
print("✅ DETIC 导入成功")
```

## 🔍 常见问题

### Q: 重启 kernel 后需要重新运行所有 cell 吗？

A: 不需要。只需要重新运行：
- Step 1（导入库）
- Step 4（初始化检测器）
- Step 6（生成 scene graph）

其他步骤（如加载数据）可以跳过，除非它们依赖检测器。

### Q: 如何确认 NumPy 已降级？

A: 在 notebook 中运行：
```python
import numpy as np
print(np.__version__)  # 应该显示 1.24.4 或类似的 1.x 版本
```

### Q: 如果 DETIC 仍然无法导入怎么办？

A: 检查以下几点：
1. 确认 NumPy 版本是 1.x（重启 kernel 后）
2. 确认 Detic 目录存在：`/home/fdse/zzy/craft/Detic`
3. 检查是否有其他错误信息
4. 如果都不行，可以使用 CLIP-only 后备检测（已自动实现）

## 💡 提示

- 如果 DETIC 成功加载，检测精度会更高
- 如果 DETIC 仍然无法加载，会自动使用 CLIP-only 后备检测
- CLIP-only 检测虽然精度稍低，但通常也能工作

## ✅ 成功标志

重启并重新运行后，如果看到：
```
✅ DETIC model loaded
Frame 0 (stage 0): ✅ 4 objects, 6 relations
```

说明一切正常！🎉

