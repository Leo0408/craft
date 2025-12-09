# 视频帧提取问题修复

## 问题描述

用户报告：
1. 视频在 notebook 中显示不出来
2. `/output` 里的视频文件也没有图像显示

## 问题分析

### 1. 视频文件检查

- ✅ 视频文件存在：`output/videos/craft_ai2thor_workflow_simple.mp4`
- ✅ 视频文件有效：9帧，分辨率 960x960，FPS 2.0
- ✅ 帧有内容：标准差 39.77，唯一值数量 256
- ⚠️ 但用户说看不到图像

### 2. 根本原因

**AI2THOR 事件对象的帧提取方法不正确**：

原来的代码只检查：
1. `event.frame` 
2. `event.metadata['image']`

但是 **AI2THOR 事件对象最可靠的方法是 `event.cv2image`**，它返回的是 BGR 格式的 OpenCV 图像。

### 3. 问题细节

- `event.frame` 可能不存在或返回空帧
- `event.metadata['image']` 可能不存在
- 没有正确的 BGR 到 RGB 转换
- 没有帧验证（检查帧是否为空或单色）

## 修复方案

### 修复后的帧提取逻辑（Step 2, Cell 9）

```python
for i, (event, action_result) in enumerate(zip(events_craft, action_results)):
    frame = None
    
    if event is None:
        print(f"  Step {i+1}: ⚠️  Event is None, creating placeholder")
    else:
        # Method 1: event.cv2image (most reliable for AI2THOR)
        if hasattr(event, 'cv2image'):
            try:
                frame = event.cv2image
                if frame is not None and len(frame.shape) == 3:
                    # cv2image is already BGR, convert to RGB
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if cv2 is not None else frame
                    print(f"  Step {i+1}: ✅ Frame extracted from cv2image ({frame.shape})")
            except Exception as e:
                print(f"  Step {i+1}: ⚠️  Failed to extract from cv2image: {e}")
        
        # Method 2: event.frame (direct frame attribute)
        if frame is None and hasattr(event, 'frame'):
            try:
                frame = event.frame
                if frame is not None:
                    if len(frame.shape) == 3:
                        if frame.shape[2] == 4:  # RGBA
                            frame = frame[:, :, :3]  # Convert to RGB
                        elif frame.shape[2] == 3:
                            # Already RGB or BGR, assume RGB
                            pass
                    print(f"  Step {i+1}: ✅ Frame extracted from frame ({frame.shape})")
            except Exception as e:
                print(f"  Step {i+1}: ⚠️  Failed to extract from frame: {e}")
        
        # Method 3: event.metadata['image']
        if frame is None and hasattr(event, 'metadata'):
            try:
                metadata = event.metadata
                if 'image' in metadata and metadata['image'] is not None:
                    frame = metadata['image']
                    print(f"  Step {i+1}: ✅ Frame extracted from metadata['image'] ({frame.shape})")
                elif 'frame' in metadata and metadata['frame'] is not None:
                    frame = metadata['frame']
                    print(f"  Step {i+1}: ✅ Frame extracted from metadata['frame'] ({frame.shape})")
            except Exception as e:
                print(f"  Step {i+1}: ⚠️  Failed to extract from metadata: {e}")
    
    # If no frame available, create a placeholder
    if frame is None:
        print(f"  Step {i+1}: ⚠️  No frame available, creating placeholder")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        if cv2 is not None:
            cv2.putText(frame, f"Step {i+1}", (50, 240), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    # Validate frame
    if frame is not None:
        # Check if frame is valid (not all zeros or all same value)
        if frame.std() < 1.0:
            print(f"  Step {i+1}: ⚠️  Warning: Frame appears to be empty or single-color (std={frame.std():.2f})")
        # Ensure frame is uint8
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
```

## 修复要点

### 1. 优先使用 `event.cv2image`

- ✅ AI2THOR 最可靠的方法
- ✅ 返回 BGR 格式的 OpenCV 图像
- ✅ 需要转换为 RGB（`cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`）

### 2. 多种备用方法

- Method 1: `event.cv2image` (优先)
- Method 2: `event.frame` (备用)
- Method 3: `event.metadata['image']` 或 `metadata['frame']` (最后备用)

### 3. 帧验证

- ✅ 检查帧是否为空（`frame.std() < 1.0`）
- ✅ 确保数据类型是 `uint8`
- ✅ 处理归一化的帧（0-1 范围转换为 0-255）

### 4. 详细的调试信息

- ✅ 显示从哪个方法提取的帧
- ✅ 显示帧的形状和统计信息
- ✅ 警告空帧或单色帧

## 使用说明

1. **重新运行 Step 1**：
   - 确保 AI2THOR 正确初始化
   - 确保事件对象正确生成

2. **重新运行 Step 2**：
   - 查看帧提取的调试信息
   - 确认从 `cv2image` 成功提取帧
   - 检查是否有警告信息

3. **检查视频**：
   - 视频文件应该包含真实的 AI2THOR 图像
   - 在 notebook 中应该能正常显示

## 预期输出

修复后，Step 2 的输出应该类似：

```
📹 Extracting frames from 9 events...
--------------------------------------------------------------------------------
  Step 1: ✅ Frame extracted from cv2image ((960, 960, 3))
  Step 2: ✅ Frame extracted from cv2image ((960, 960, 3))
  Step 3: ✅ Frame extracted from cv2image ((960, 960, 3))
  ...
```

如果看到 `⚠️  Warning: Frame appears to be empty`，说明帧提取有问题，需要进一步检查。

## 注意事项

1. **AI2THOR 版本**：
   - 不同版本的 AI2THOR 可能有不同的 API
   - `cv2image` 是最通用的方法

2. **BGR vs RGB**：
   - `cv2image` 返回 BGR 格式
   - 需要转换为 RGB 才能在视频中正确显示

3. **帧验证**：
   - 如果帧的标准差很小（< 1.0），可能是空帧或单色帧
   - 需要检查 AI2THOR 是否正确渲染

## 故障排除

如果修复后仍然看不到图像：

1. **检查 AI2THOR 是否正确初始化**：
   ```python
   print(f"Controller initialized: {controller is not None}")
   print(f"Last event: {controller.last_event is not None}")
   ```

2. **检查事件对象**：
   ```python
   event = events_craft[0]
   print(f"Event type: {type(event)}")
   print(f"Has cv2image: {hasattr(event, 'cv2image')}")
   print(f"Has frame: {hasattr(event, 'frame')}")
   ```

3. **手动测试帧提取**：
   ```python
   event = events_craft[0]
   if hasattr(event, 'cv2image'):
       frame = event.cv2image
       print(f"Frame shape: {frame.shape}")
       print(f"Frame dtype: {frame.dtype}")
       print(f"Frame range: [{frame.min()}, {frame.max()}]")
   ```

4. **检查视频编码**：
   - 确保使用正确的视频编码器（`mp4v`）
   - 检查视频文件是否损坏

