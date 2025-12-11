# Step 5 模块重新加载修复

## 问题描述

在 Step 5 执行时出现错误：
```
AttributeError: 'ConstraintGenerator' object has no attribute 'compile_constraint'
```

## 问题原因

虽然 `compile_constraint` 方法已经添加到 `ConstraintGenerator` 类中，但 Jupyter notebook 的 Python 内核可能还没有重新加载更新后的模块。在 Jupyter 中，当你修改了 Python 文件后，需要重新导入模块才能使用新的方法。

## 修复方案

在 Cell 15（Step 5）的开头添加了模块重新加载代码：

```python
# Reload ConstraintGenerator to get the latest compile_constraint method
import importlib
from craft.reasoning import constraint_generator as cg_module
importlib.reload(cg_module)
from craft.reasoning.constraint_generator import ConstraintGenerator

# Recreate constraint_generator instance with updated class
if 'llm_prompter' in globals() and llm_prompter is not None:
    constraint_generator = ConstraintGenerator(llm_prompter)
    print("✅ Reloaded ConstraintGenerator with compile_constraint method")
else:
    print("⚠️  llm_prompter not found, using existing constraint_generator")
```

## 修复内容

1. ✅ **使用 `importlib.reload()`**：重新加载 `constraint_generator` 模块
2. ✅ **重新导入类**：从重新加载的模块中导入 `ConstraintGenerator`
3. ✅ **重新创建实例**：使用更新后的类重新创建 `constraint_generator` 实例

## 工作原理

1. **`importlib.reload(module)`**：强制 Python 重新读取并执行模块文件，加载最新的代码
2. **重新导入类**：确保使用最新版本的 `ConstraintGenerator` 类
3. **重新创建实例**：使用更新后的类创建新的 `constraint_generator` 实例，该实例现在包含 `compile_constraint` 方法

## 验证结果

所有修复已通过验证：
- ✅ 重新导入模块
- ✅ 重新创建实例
- ✅ compile_constraint调用

## 使用说明

现在可以重新运行 Cell 15（Step 5），应该能够正常调用 `compile_constraint` 方法。

## 替代方案

如果上述方法仍然不起作用，可以尝试：

1. **重启 Jupyter 内核**：在 Jupyter 菜单中选择 "Kernel" → "Restart Kernel"
2. **重新运行所有 Cell**：从 Cell 1 开始重新运行所有 cell

## 注意事项

- 模块重新加载只影响当前 notebook 会话
- 如果修改了其他模块文件，也需要类似地重新加载
- 在生产环境中，建议使用模块热重载机制或重启服务

