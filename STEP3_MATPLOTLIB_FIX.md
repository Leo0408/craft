# Step 3 matplotlib API 修复

## 问题描述

Step 3 生成完整视频时出现错误：
```
AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'
```

## 问题原因

在 matplotlib 3.10.1 中：
- ✅ `buffer_rgba()` 可用
- ✅ `print_to_buffer()` 可用
- ❌ `renderer.tostring_rgb()` **不可用**（已废弃）

原来的代码尝试使用 `renderer.tostring_rgb()`，但在新版本的 matplotlib 中这个方法不存在。

## 修复方案

### 修复后的代码逻辑

```python
# Convert figure to numpy array
fig.canvas.draw()
# Handle different matplotlib versions
try:
    # Method 1: buffer_rgba (matplotlib 3.5+, recommended)
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    h, w = fig.canvas.get_width_height()
    buf = buf.reshape((h, w, 4))
    buf = buf[:, :, :3]  # Remove alpha channel, keep RGB
except (AttributeError, TypeError) as e:
    # Method 2: print_to_buffer (alternative)
    try:
        buf_data = fig.canvas.print_to_buffer()
        buf = np.frombuffer(buf_data[0], dtype=np.uint8)
        h, w = fig.canvas.get_width_height()
        buf = buf.reshape((h, w, 4))
        buf = buf[:, :, :3]  # Remove alpha channel
    except (AttributeError, TypeError) as e2:
        # Method 3: Fallback - Save to buffer and read back with PIL
        try:
            import io
            buf_io = io.BytesIO()
            fig.savefig(buf_io, format='png', bbox_inches='tight', dpi=100)
            buf_io.seek(0)
            from PIL import Image
            img = Image.open(buf_io)
            buf = np.array(img)
            # Handle RGBA or RGB
            if len(buf.shape) == 3 and buf.shape[2] == 4:
                buf = buf[:, :, :3]  # Remove alpha channel
            elif len(buf.shape) == 3 and buf.shape[2] == 3:
                pass  # Already RGB
            buf_io.close()
        except Exception as e3:
            # Final fallback: Create a blank image
            h, w = fig.canvas.get_width_height()
            buf = np.zeros((h, w, 3), dtype=np.uint8)
            print(f"⚠️  Warning: All matplotlib conversion methods failed, using blank image. Last error: {e3}")
plt.close(fig)

return buf
```

## 修复要点

### 1. 移除了不可用的方法

- ❌ 移除了 `renderer.tostring_rgb()`（在 matplotlib 3.10.1 中不可用）

### 2. 使用可用的方法（按优先级）

1. **Method 1**: `buffer_rgba()` - matplotlib 3.5+ 推荐方法
2. **Method 2**: `print_to_buffer()` - 替代方法
3. **Method 3**: PIL fallback - 保存为 PNG 然后读取（最可靠）

### 3. 改进的错误处理

- ✅ 每个方法都有独立的异常处理
- ✅ 最终 fallback 创建空白图像（避免崩溃）
- ✅ 详细的错误信息

## 测试结果

在 matplotlib 3.10.1 中测试：
- ✅ `buffer_rgba()`: 成功
- ✅ `print_to_buffer()`: 成功
- ❌ `renderer.tostring_rgb()`: 失败（已移除）

## 使用说明

修复后，Step 3 应该能够：
1. ✅ 成功生成完整视频（包含 scene graphs）
2. ✅ 在 notebook 中显示视频
3. ✅ 处理不同版本的 matplotlib

## 注意事项

1. **Jupyter Kernel 重启**：
   - 如果修复后仍然出错，可能需要重启 Jupyter kernel
   - 或者使用 `importlib.reload()` 重新加载模块

2. **PIL 依赖**：
   - 确保安装了 `Pillow` 包（`pip install Pillow`）
   - PIL fallback 需要这个依赖

3. **性能**：
   - `buffer_rgba()` 最快
   - `print_to_buffer()` 次之
   - PIL fallback 最慢（但最可靠）

## 故障排除

如果修复后仍然出错：

1. **重启 Jupyter Kernel**：
   ```python
   # 在 notebook 中
   import importlib
   import craft.utils.video_generator
   importlib.reload(craft.utils.video_generator)
   ```

2. **检查 matplotlib 版本**：
   ```python
   import matplotlib
   print(f"matplotlib version: {matplotlib.__version__}")
   ```

3. **检查 PIL 是否安装**：
   ```python
   try:
       from PIL import Image
       print("✅ PIL installed")
   except ImportError:
       print("❌ PIL not installed, run: pip install Pillow")
   ```

4. **手动测试**：
   ```python
   import matplotlib.pyplot as plt
   import numpy as np
   
   fig = plt.figure()
   ax = fig.add_subplot(111)
   ax.plot([1, 2, 3], [1, 2, 3])
   fig.canvas.draw()
   
   # 测试 buffer_rgba
   try:
       buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
       print(f"✅ buffer_rgba works: {len(buf)} bytes")
   except Exception as e:
       print(f"❌ buffer_rgba failed: {e}")
   
   plt.close(fig)
   ```

