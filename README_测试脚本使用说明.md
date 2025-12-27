# DETIC测试脚本使用说明

## 在终端测试DETIC模块加载

如果在Jupyter notebook中遇到注册冲突问题，可以在终端运行测试脚本来诊断问题。

### 使用方法

1. **激活conda环境**：
   ```bash
   source /home/fdse/anaconda3/etc/profile.d/conda.sh
   conda activate reflect_env
   cd /home/fdse/zzy/craft
   ```

2. **运行测试脚本**：
   ```bash
   python test_detic_notebook_fix.py
   ```

   或者使用简化的测试脚本：
   ```bash
   python test_detic_simple.py
   ```

### 测试脚本说明

#### `test_detic_notebook_fix.py`
- **功能**：模拟Jupyter notebook环境，测试DETIC加载过程
- **特点**：
  - 清理模块缓存（模拟重启kernel）
  - 逐步测试每个导入步骤
  - 详细显示每个步骤的成功/失败状态
  - 提供清晰的错误信息和修复建议

#### `test_detic_simple.py`
- **功能**：简化的DETIC测试脚本
- **特点**：更快速，适合快速验证

### 常见问题和解决方案

#### 1. 注册冲突错误（`build_mnv2_backbone already registered`）

**原因**：在Jupyter notebook中，模块可能被多次加载，导致注册冲突。

**解决方案**：
1. **重启Jupyter kernel**（推荐）：
   - 在Jupyter notebook中：`Kernel → Restart Kernel`
   - 然后重新运行所有cells（按顺序）

2. **检查模块缓存**：
   - 如果问题仍然存在，可能是notebook环境的模块缓存问题
   - 建议：完全关闭Jupyter，重新打开notebook

#### 2. `detic.config`导入失败

**原因**：注册冲突导致模块导入中断。

**当前处理**：
- 代码已经改进了错误处理
- 如果检测到注册冲突，会尝试从缓存中获取函数
- 如果缓存中也没有，会给出明确的错误提示

#### 3. CustomRCNN未注册

**原因**：`detic.modeling`未能成功导入。

**解决方案**：
- 确保`detic.config`成功导入后，再导入`detic.modeling`
- 如果仍有问题，重启kernel

### 输出说明

测试脚本的输出包含以下标记：
- ✅ 成功
- ⚠️  警告（可能的问题，但不影响继续）
- ❌ 错误（需要修复）
- ℹ️  信息（状态提示）

### 在Notebook中使用

如果终端测试通过，但在notebook中仍然失败：

1. **确保重启kernel**：
   - 运行`test_detic_notebook_fix.py`会清理模块缓存
   - 但在notebook中，你需要手动重启kernel

2. **按顺序运行cells**：
   - 确保所有导入cells都在正确的顺序
   - 特别是Cell 5（Step 4）应该在所有导入完成后运行

3. **检查DETECTION_METHOD设置**：
   - 在Cell 5中，确保`DETECTION_METHOD = 'detic_clip'`

### 成功标志

如果看到以下输出，说明DETIC加载成功：
```
✅✅✅ DETIC模型加载成功！
✅ DETIC可以正常使用！
```

如果看到以下输出，说明使用的是CLIP-only模式（也可以工作，但精度略低）：
```
⚠️  DETIC未加载，但CLIP-only模式可用
⚠️  精度会略有下降，但仍可检测对象
```

