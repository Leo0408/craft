# Demo2.ipynb 流程分析与 Step6 问题诊断

## 📋 整体流程概述（参考 Method.md）

根据 `demo2.ipynb` 和 `Method.md`，CRAFT++ 框架在真实环境中的完整流程如下：

### 1. **Step 1-2: 环境设置与配置**
- 配置 Hugging Face 镜像（用于模型下载）
- 导入必要的模块（CRAFT 核心模块、感知模块、推理模块等）
- 设置任务配置（任务文件夹、LLM API、相机内参、检测阈值等）

### 2. **Step 3: 加载 REFLECT 数据集**
- 使用 `ReflectDataLoader` 加载任务信息（从 `tasks_real_world.json`）
- 打开 zarr 文件（`replay_buffer.zarr`）
- 获取总帧数和动作阶段信息

**关键输出**：
- `task_info`: 任务信息（动作序列、对象列表、成功条件等）
- `zarr_group`: zarr 文件对象
- `total_frames`: 总帧数（例如：6568 帧）

### 3. **Step 4: 初始化模型和检测器**
- **MDETR 检测器**：用于开放词表目标检测
- **SceneAnalyzer**：用于计算空间关系
- **ReflectSceneGraphBuilder**：用于从 RGB-D 图像构建场景图
- **EnvironmentMemory**：用于处理遮挡、噪声和传感器误差（Method.md Section 3）
- **LLM 组件**：用于约束生成和失败分析

### 4. **Step 5: 选择关键帧并加载 RGB-D 数据**
- 根据动作阶段变化选择关键帧（例如：动作转换前后的帧）
- 从 zarr 文件加载 RGB 和深度图像
- 存储到 `frame_data` 字典中

**关键输出**：
- `key_frames`: 关键帧索引列表（例如：[0, 921, 941, ...]）
- `frame_data`: 每帧的 RGB 和深度数据

### 5. **Step 6: 生成场景图（带 Environment Memory）** ⚠️ **问题发生在这里**

这是核心步骤，流程如下：

```
对于每个关键帧：
  1. 使用 ReflectSceneGraphBuilder.process_frame() 处理帧
     ↓
  2. MDETR 检测器检测物体 (detect_with_depth)
     ↓
  3. 如果检测到物体：
     - 从深度图生成点云
     - 计算 3D 边界框
     - 创建场景图节点
     - 计算空间关系（inside, on_top_of, near）
     - 创建场景图边
     ↓
  4. 更新 Environment Memory
     - 使用 Kalman-like 平滑位置
     - 处理遮挡（物体未检测到 ≠ 消失）
     - 更新置信度
     ↓
  5. 从 Memory 生成平滑后的场景图
```

**关键代码位置**：
- `perception/reflect_scene_graph_builder.py:102-316` - `process_frame()` 方法
- `perception/mdetr_detector.py:195-286` - `detect_objects()` 方法

### 6. **Step 7-12: 后续处理**
- Step 7: 可视化场景图和检测结果
- Step 8: 生成动作感知约束（LLM 或模板化方法）
- Step 9: 编译约束为可执行代码
- Step 10: 使用约束验证进行失败检测
- Step 11: 生成渐进式解释
- Step 12: 总结结果

---

## 🔍 Step6 出现空场景图的原因分析

### 问题现象

```
--- Processing frame 921 (stage 0) ---
Nothing detected in frame 921

  📊 Raw Scene Graph:
    Nodes: 0
    Edges: 0

  🧠 Memory-Smoothed Scene Graph:
    Nodes: 0
    Edges: 0
```

### 根本原因

根据代码分析，空场景图的产生流程如下：

1. **检测阶段**（`ReflectSceneGraphBuilder.process_frame()` 第 138-143 行）：
   ```python
   detections = self.detector.detect_with_depth(
       rgb_pil,
       depth,
       object_list,
       self.camera_intrinsics
   )
   ```

2. **检测结果检查**（第 145-147 行）：
   ```python
   if len(detections) == 0:
       print(f"Nothing detected in frame {step_idx}")
       return local_sg  # 返回空场景图
   ```

3. **MDETR 检测逻辑**（`MDETRDetector.detect_objects()` 第 195-286 行）：
   - 对 `object_list` 中的每个对象运行 MDETR 模型
   - 使用置信度阈值过滤：`keep = (probas > self.threshold).cpu()`
   - 如果所有对象的检测置信度都低于阈值，返回空列表

### 可能的原因

#### 1. **检测阈值过高** ⭐ **最可能的原因**
- **当前阈值**：`DETECTION_THRESHOLD = 0.7`（在 Step 2 中设置）
- **问题**：如果 frame 921 中物体的检测置信度都低于 0.7，则会被过滤掉
- **解决方案**：
  ```python
  # 在 Step 2 中降低阈值
  DETECTION_THRESHOLD = 0.5  # 或更低，如 0.3
  ```

#### 2. **物体不在视野中**
- **原因**：frame 921 时，任务相关物体（coffee machine, cup 等）可能不在相机视野内
- **验证方法**：检查 frame 921 的 RGB 图像，看是否有相关物体
- **解决方案**：这是正常情况，Environment Memory 应该处理这种情况（但需要之前帧有检测到）

#### 3. **物体被遮挡**
- **原因**：物体可能被机械臂、其他物体或相机视角遮挡
- **验证方法**：查看 RGB 图像，检查是否有遮挡
- **解决方案**：Environment Memory 应该从之前帧的记忆中恢复被遮挡的物体（但需要之前帧有检测到）

#### 4. **检测器模型问题**
- **原因**：MDETR 模型可能未正确加载或性能不佳
- **验证方法**：检查 Step 4 的输出，确认模型是否成功加载
- **解决方案**：确保模型正确加载，检查设备（CPU/GPU）

#### 5. **图像质量问题**
- **原因**：frame 921 的图像可能模糊、过暗或过亮
- **验证方法**：可视化 frame 921 的 RGB 图像
- **解决方案**：图像预处理或调整检测参数

#### 6. **对象名称不匹配**
- **原因**：`object_list` 中的对象名称可能与 MDETR 期望的格式不匹配
- **当前 object_list**：`['coffee machine', 'purple cup', 'blue cup with handle', 'table on the left of sink']`
- **验证方法**：检查 MDETR 是否能识别这些对象名称
- **解决方案**：调整对象名称格式，或使用更通用的名称

### 诊断步骤

#### 步骤 1: 检查检测器输出
在 Step 6 中添加调试代码：

```python
# 在 process_frame 调用前添加
print(f"Processing frame {step_idx}")
print(f"Object list: {object_list}")
print(f"Detection threshold: {DETECTION_THRESHOLD}")

# 在 detect_with_depth 调用后添加
print(f"Raw detections count: {len(detections)}")
if len(detections) > 0:
    for det in detections:
        print(f"  - {det['label']}: confidence={det['confidence']:.3f}")
```

#### 步骤 2: 可视化 frame 921
在 Step 5 后添加：

```python
# 可视化 frame 921
if 921 in frame_data:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(frame_data[921]['rgb'])
    axes[0].set_title('Frame 921 - RGB')
    axes[0].axis('off')
    axes[1].imshow(frame_data[921]['depth'], cmap='jet')
    axes[1].set_title('Frame 921 - Depth')
    axes[1].axis('off')
    plt.show()
```

#### 步骤 3: 检查 Environment Memory 状态
在 Step 6 中添加：

```python
# 在处理 frame 921 前，检查 Memory 状态
if step_idx == 921:
    world_state = environment_memory.get_world_state()
    print(f"Memory state before frame 921:")
    print(f"  Objects in memory: {list(world_state['objects'].keys())}")
    for obj_name, obj_state in world_state['objects'].items():
        print(f"    - {obj_name}: occluded={obj_state['occluded']}, confidence={obj_state['confidence']:.2f}")
```

### 解决方案建议

#### 方案 1: 降低检测阈值（推荐）
```python
# 在 Step 2 中修改
DETECTION_THRESHOLD = 0.5  # 从 0.7 降低到 0.5
```

#### 方案 2: 使用 Environment Memory 恢复被遮挡物体
如果之前帧检测到了物体，Environment Memory 应该能够恢复它们。检查 Memory 是否正确更新：

```python
# 在 Step 6 中，确保使用 Memory 恢复的物体
# 当前代码已经实现了这个逻辑（第 238-271 行），但需要确保之前帧有检测到物体
```

#### 方案 3: 调整对象名称
```python
# 尝试更通用的名称
object_list = ['coffee machine', 'cup', 'table']  # 简化名称
```

#### 方案 4: 检查 zarr 文件
确认 frame 921 的数据是否正确加载：

```python
# 在 Step 5 后添加
if 921 in frame_data:
    rgb = frame_data[921]['rgb']
    depth = frame_data[921]['depth']
    print(f"Frame 921 RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"Frame 921 Depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"Depth range: [{depth.min()}, {depth.max()}]")
```

### 与 Method.md 的对应关系

根据 `Method.md` Section 1.5 和 Section 3：

1. **多模态感知层**（Section 1.5.1）：
   - 使用 MDETR 进行开放词表检测
   - 检测结果应该包含置信度分数
   - **问题**：如果检测置信度低于阈值，物体不会被检测到

2. **环境记忆模块**（Section 3）：
   - 应该处理遮挡情况（物体未检测到 ≠ 消失）
   - 使用时间连续性约束世界状态
   - **问题**：如果之前帧也没有检测到物体，Memory 无法恢复

3. **场景图构建**（Section 1.5.4）：
   - 从检测结果构建场景图
   - 如果检测结果为空，场景图也为空

### 总结

**Step6 出现空场景图的主要原因是：MDETR 检测器在 frame 921 时没有检测到任何物体（置信度低于阈值 0.7）。**

**最可能的解决方案**：
1. 降低检测阈值（从 0.7 降到 0.5 或更低）
2. 检查 frame 921 的图像质量
3. 确认对象名称格式是否正确
4. 如果之前帧检测到了物体，Environment Memory 应该能够恢复（但需要检查 Memory 是否正确更新）

**建议的调试流程**：
1. 首先可视化 frame 921 的图像，确认是否有物体
2. 添加调试输出，查看检测器的原始输出
3. 降低检测阈值，重新运行
4. 检查 Environment Memory 的状态，确认是否有之前帧的记忆

