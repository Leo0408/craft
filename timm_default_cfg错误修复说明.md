# timm 参数传递错误修复说明

## 🔍 问题根源

错误信息（可能出现的两种）：
```
TypeError: __init__() got an unexpected keyword argument 'default_cfg'
TypeError: __init__() got an unexpected keyword argument 'pretrained_custom_load'
```

### 为什么一直修改也没办法解决？

1. **Jupyter Notebook 模块缓存**：
   - Jupyter 会缓存已导入的模块
   - 即使修改了源代码，Jupyter 仍可能使用缓存的旧版本
   - 需要**重启 kernel** 才能加载新代码

2. **`build_model_with_cfg` 的行为**：
   - `build_model_with_cfg` 会将 `default_cfg` 参数传递给模型类的 `__init__`
   - `CustomResNet` 继承自 `ResNet`，而 `ResNet.__init__` **不接受** `default_cfg` 参数
   - 之前的修复只处理了 `DefaultCfg` 对象的赋值问题，但没有处理参数传递问题

3. **参数传递链**：
   ```
   build_model_with_cfg(CustomResNet, ..., default_cfg=cfg_to_use, ...)
   ↓
   CustomResNet.__init__(**kwargs)  # kwargs 包含 default_cfg
   ↓
   ResNet.__init__(**kwargs)  # ❌ ResNet 不接受 default_cfg
   ```

## ✅ 解决方案

### 修复内容

修改了 `/home/fdse/zzy/craft/Detic/detic/modeling/backbone/timm.py` 中的 `CustomResNet.__init__` 方法：

```python
class CustomResNet(ResNet):
    def __init__(self, **kwargs):
        self.out_indices = kwargs.pop('out_indices', None)
        # Remove parameters that build_model_with_cfg handles separately
        # but ResNet.__init__ doesn't accept
        kwargs.pop('default_cfg', None)  # ⬅️ 移除 default_cfg
        kwargs.pop('pretrained_custom_load', None)  # ⬅️ 移除 pretrained_custom_load
        super().__init__(**kwargs)
```

### 为什么这样修复？

1. **`build_model_with_cfg` 会处理这些参数**：
   - `build_model_with_cfg` 在调用模型类之前会处理 `default_cfg` 和 `pretrained_custom_load`
   - `default_cfg` 主要用于设置预训练模型的 URL 和配置
   - `pretrained_custom_load` 用于指示使用自定义的预训练模型加载方法
   - 这些参数不需要传递给 `ResNet.__init__`

2. **`ResNet.__init__` 不接受这些参数**：
   - `timm.models.resnet.ResNet` 的 `__init__` 方法不接受 `default_cfg` 或 `pretrained_custom_load` 参数
   - 如果传递了不支持的参数，会抛出 `TypeError`

3. **移除参数是安全的**：
   - 这些参数已经在 `build_model_with_cfg` 中处理过了
   - 移除它们不会影响模型的功能
   - `CustomResNet` 有自己的 `load_pretrained` 方法来处理自定义加载

## 📋 修复步骤

### 1. 确认修复已应用

```bash
# 检查修复后的代码
grep -A 3 "class CustomResNet" /home/fdse/zzy/craft/Detic/detic/modeling/backbone/timm.py
```

应该看到：
```python
class CustomResNet(ResNet):
    def __init__(self, **kwargs):
        self.out_indices = kwargs.pop('out_indices', None)
        kwargs.pop('default_cfg', None)  # ⬅️ 移除 default_cfg
        kwargs.pop('pretrained_custom_load', None)  # ⬅️ 移除 pretrained_custom_load
```

### 2. 清除 Jupyter 模块缓存

**重要：必须重启 kernel！**

在 Jupyter Notebook 中：
1. 点击 **Kernel → Restart Kernel**（或按 `Ctrl+M` 然后输入 `restart`）
2. 这会清除所有已导入的模块缓存

### 3. 重新运行 Step 4

重新运行初始化 DETIC + CLIP 检测器的 cell。

### 4. 如果仍然失败

如果重启 kernel 后仍然失败，可能是其他原因：

1. **检查文件是否真的被修改**：
   ```python
   import inspect
   from detic.modeling.backbone.timm import CustomResNet
   print(inspect.getsource(CustomResNet.__init__))
   ```
   应该看到 `kwargs.pop('default_cfg', None)`

2. **手动清除模块缓存**：
   ```python
   import sys
   modules_to_clear = [k for k in sys.modules.keys() if 'detic' in k or 'timm' in k]
   for m in modules_to_clear:
       del sys.modules[m]
   ```

3. **检查 timm 版本兼容性**：
   ```python
   import timm
   print(f"timm version: {timm.__version__}")
   ```

## 🎯 预期结果

修复后，应该看到：

```
============================================================
DETECTION METHOD: DETIC_CLIP
============================================================
Initializing DETIC + CLIP detector...
✅ Loaded DeticClipDetector module (fresh import)
   Using device: cpu
📁 Using local config: ...
📁 Using local weights: ...
✅ DETIC model loaded  ← 关键！不再出现 TypeError
✅ CLIP model loaded (ViT-B/32)
✅ DETIC + CLIP detector initialized successfully!
```

## 🔧 相关文件

- `/home/fdse/zzy/craft/Detic/detic/modeling/backbone/timm.py` - 修复的文件
- `/home/fdse/zzy/craft/perception/detic_clip_detector.py` - DETIC 检测器初始化代码

## 💡 总结

这个问题的根本原因是：
1. **参数传递问题**：`build_model_with_cfg` 传递了 `default_cfg` 和 `pretrained_custom_load`，但 `ResNet` 不接受这些参数
2. **模块缓存问题**：Jupyter 缓存了旧代码，需要重启才能加载新代码

修复方法：在 `CustomResNet.__init__` 中移除这些参数，因为 `build_model_with_cfg` 已经处理过它们了。

**注意**：如果将来还出现类似的 `unexpected keyword argument` 错误，可能是 `build_model_with_cfg` 传递了其他 `ResNet` 不接受的参数。解决方法相同：在 `CustomResNet.__init__` 中移除这些参数。

