# STEP 1 (ALTERNATIVE) 和 STEP 2 调试指南

## 问题现象

执行 STEP 2 后没有出现：
```
✅ Using X frames from STEP 1 (ALTERNATIVE)
   These frames were loaded from REFLECT's pre-generated data
```

## 可能的原因

### 1. STEP 1 (ALTERNATIVE) 没有成功加载 frames

**检查方法**：
在 STEP 1 (ALTERNATIVE) 执行后，检查以下变量：

```python
# 在 notebook 中执行
print(f"frames 变量是否存在: {'frames' in globals()}")
if 'frames' in globals():
    print(f"frames 长度: {len(frames)}")
    print(f"frames 类型: {type(frames)}")
else:
    print("❌ frames 变量不存在！")
```

**可能的原因**：
- `ego_img_dir` 路径不存在
- `REFLECT_DATA_ROOT` 或 `TASK_NAME` 设置错误
- 图像文件不存在或格式不正确

### 2. frames 变量为空列表

即使 `frames` 变量存在，如果它是空列表 `[]`，检查条件 `len(frames) > 0` 会失败。

**检查方法**：
```python
# 在 notebook 中执行
if 'frames' in globals():
    if len(frames) == 0:
        print("⚠️  frames 存在但是为空！")
        print("   检查 STEP 1 (ALTERNATIVE) 的输出，看是否有错误消息")
    else:
        print(f"✅ frames 有 {len(frames)} 个元素")
```

### 3. Kernel 重启导致变量丢失

如果在执行 STEP 1 (ALTERNATIVE) 后重启了 kernel，所有变量都会丢失。

**解决方法**：
- 确保在同一个 kernel session 中执行 STEP 1 (ALTERNATIVE) 和 STEP 2
- 不要重启 kernel

### 4. 路径配置错误

**检查方法**：
在 STEP 1 (ALTERNATIVE) 中，检查路径是否正确：

```python
# 在 STEP 1 (ALTERNATIVE) 中执行
import os
REFLECT_DATA_ROOT = "/home/fdse/zzy/reflect"
TASK_NAME = "makeCoffee-1"
TASK_FOLDER = f"thor_tasks/makeCoffee/{TASK_NAME}"
reflect_task_path = os.path.join(REFLECT_DATA_ROOT, TASK_FOLDER)

print(f"检查路径: {reflect_task_path}")
print(f"路径存在: {os.path.exists(reflect_task_path)}")

ego_img_dir = os.path.join(reflect_task_path, "ego_img")
print(f"ego_img 目录: {ego_img_dir}")
print(f"ego_img 目录存在: {os.path.exists(ego_img_dir)}")

if os.path.exists(ego_img_dir):
    import glob
    img_files = glob.glob(os.path.join(ego_img_dir, "img_step_*.png"))
    print(f"找到 {len(img_files)} 个图像文件")
    if len(img_files) > 0:
        print(f"前 5 个文件: {img_files[:5]}")
```

## 调试步骤

### 步骤 1: 验证 STEP 1 (ALTERNATIVE) 执行成功

1. 执行 STEP 1 (ALTERNATIVE) 的 cell
2. 检查输出，应该看到：
   ```
   ✅ Loaded X frames
   ```
3. 如果没有看到这个输出，说明 frames 没有被加载

### 步骤 2: 检查变量

在 STEP 1 (ALTERNATIVE) 执行后，立即执行：

```python
# 检查变量
print("="*80)
print("变量检查")
print("="*80)
print(f"frames in globals(): {'frames' in globals()}")
if 'frames' in globals():
    print(f"len(frames): {len(frames)}")
    print(f"type(frames): {type(frames)}")
    if len(frames) > 0:
        print(f"frames[0] shape: {frames[0].shape if hasattr(frames[0], 'shape') else 'N/A'}")
print(f"frame_annotations in globals(): {'frame_annotations' in globals()}")
if 'frame_annotations' in globals():
    print(f"len(frame_annotations): {len(frame_annotations)}")
```

### 步骤 3: 执行 STEP 2

1. 确保 STEP 1 (ALTERNATIVE) 已执行且 frames 不为空
2. 执行 STEP 2
3. 检查输出

### 步骤 4: 如果仍然没有看到消息

检查 STEP 2 的代码是否正确：

```python
# 在 STEP 2 开始处添加调试代码
print("="*80)
print("STEP 2 调试信息")
print("="*80)
print(f"'frames' in globals(): {'frames' in globals()}")
if 'frames' in globals():
    print(f"len(frames): {len(frames)}")
    print(f"frames > 0: {len(frames) > 0}")
    print(f"检查条件结果: {'frames' in globals() and len(frames) > 0}")
```

## 常见问题解决

### 问题 1: ego_img 目录不存在

**症状**：STEP 1 (ALTERNATIVE) 输出：
```
❌ Ego image directory not found: /path/to/ego_img
```

**解决**：
1. 检查 `REFLECT_DATA_ROOT` 和 `TASK_NAME` 是否正确
2. 确认 REFLECT 数据确实存在于该路径
3. 如果数据在其他位置，修改路径配置

### 问题 2: 图像文件不存在

**症状**：STEP 1 (ALTERNATIVE) 输出：
```
📹 Loading 0 frames from ego_img directory...
✅ Loaded 0 frames
```

**解决**：
1. 检查 `ego_img` 目录中是否有 `img_step_*.png` 文件
2. 确认文件命名格式正确
3. 如果文件在其他位置，需要调整代码

### 问题 3: Kernel 重启

**症状**：执行 STEP 2 时提示变量不存在

**解决**：
1. 重新执行 STEP 1 (ALTERNATIVE)
2. 确保在同一个 kernel session 中执行两个步骤

## 验证修复

修复后，执行以下步骤验证：

1. **执行 STEP 1 (ALTERNATIVE)**
   - 应该看到 "✅ Loaded X frames"
   - 确认 frames 变量已创建

2. **检查变量**（可选）
   ```python
   print(f"frames: {len(frames) if 'frames' in globals() else 'NOT FOUND'}")
   ```

3. **执行 STEP 2**
   - 应该看到 "✅ Using X frames from STEP 1 (ALTERNATIVE)"
   - 可视化应该显示正确的 frames

## 联系支持

如果问题仍然存在，请提供：
1. STEP 1 (ALTERNATIVE) 的完整输出
2. STEP 2 的完整输出
3. 变量检查的结果
4. 路径配置信息

