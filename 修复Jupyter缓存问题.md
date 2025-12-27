# 修复 Jupyter Notebook 缓存问题

## 问题

如果遇到 `UnboundLocalError: local variable 'DETIC_AVAILABLE' referenced before assignment` 错误，这通常是因为 Jupyter notebook 缓存了旧版本的模块。

## 解决方案

### 方案 1: 重启 Kernel（最简单）⭐

1. 在 Jupyter notebook 中：**Kernel → Restart Kernel**
2. 重新运行所有 cells（从 Step 1 开始）

### 方案 2: 强制重新加载模块（已添加到代码中）

代码已经添加了强制重新加载模块的功能。如果仍然有问题，可以手动执行：

```python
# 在 Step 4 Alternative cell 中，在导入之前添加：
import importlib
import sys

# 清除模块缓存
module_name = 'craft.perception.detic_clip_detector'
if module_name in sys.modules:
    del sys.modules[module_name]

# 重新导入
from craft.perception.detic_clip_detector import DeticClipDetector
```

### 方案 3: 检查代码版本

确认 `perception/detic_clip_detector.py` 中的 `_load_models` 方法使用的是实例变量：

```python
def _load_models(self):
    # 应该使用实例变量，而不是全局变量
    detic_available = self.detic_available  # ✅ 正确
    # 而不是
    # if DETIC_AVAILABLE:  # ❌ 错误（在方法中直接使用全局变量）
```

## 验证修复

运行 Step 4 Alternative 后，应该看到：

```
✅ Reloaded DeticClipDetector module
✅ DETIC + CLIP detector initialized successfully!
```

而不是：

```
UnboundLocalError: local variable 'DETIC_AVAILABLE' referenced before assignment
```

## 如果问题仍然存在

1. **完全重启 Jupyter**：关闭并重新打开 notebook
2. **检查文件**：确认 `perception/detic_clip_detector.py` 已保存最新版本
3. **手动验证**：在 Python 终端中测试导入：
   ```python
   from craft.perception.detic_clip_detector import DeticClipDetector
   detector = DeticClipDetector(device='cpu')
   ```

