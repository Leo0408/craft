# Step 6 和 Step 7 问题诊断报告

## 🔍 问题现象

Step 7 后 scene graph 都是空的。

## 🔎 根本原因

从代码分析发现：

1. **`scene_graph_builder` 未定义**
   - `scene_graph_builder` 应该在 cell 12 中初始化
   - Step 6 (cell 15) 中使用了 `scene_graph_builder.process_frame()`
   - 但执行时出现 `NameError: name 'scene_graph_builder' is not defined`

2. **错误处理逻辑**
   - Step 6 中有 try-except 块捕获异常
   - 当 `scene_graph_builder` 未定义时，异常被捕获
   - 然后创建了空的 `SceneGraph` 对象
   - 导致所有 scene graph 都是空的

3. **错误输出示例**
   ```
   ⚠️  Error processing frame 0: name 'scene_graph_builder' is not defined
   ⚠️  Error processing frame 921: name 'scene_graph_builder' is not defined
   ...
   ✅ Generated 9 raw scene graphs
   ✅ Generated 9 memory-smoothed scene graphs
   ```
   虽然显示"生成了 9 个 scene graph"，但实际上都是空的。

## 🔧 解决方案

### 方案 1：确保 cell 12 已运行（推荐）

在运行 Step 6 之前，确保：
1. **运行 cell 12**（初始化 `scene_graph_builder`）
2. **检查输出**：应该看到 `✅ Scene graph builder initialized`

### 方案 2：在 Step 6 中添加检查

在 Step 6 开始时添加检查：

```python
# Check if scene_graph_builder is defined
if 'scene_graph_builder' not in globals():
    print("⚠️  scene_graph_builder not found, initializing...")
    from craft.perception import ReflectSceneGraphBuilder
    scene_graph_builder = ReflectSceneGraphBuilder(
        detector=detector,
        scene_analyzer=scene_analyzer,
        camera_intrinsics=CAMERA_INTRINSICS,
        voxel_size=0.01
    )
    print("✅ Scene graph builder initialized")
```

### 方案 3：修复异常处理逻辑

在 Step 6 的异常处理中，不要创建空的 SceneGraph，而是：
1. 打印清晰的错误信息
2. 跳过该帧或重新初始化 `scene_graph_builder`

## 📋 检查清单

在运行 Step 6 之前，请确认：

- [ ] Cell 12 已运行
- [ ] 看到 `✅ Scene graph builder initialized` 输出
- [ ] `detector` 已初始化（Step 4）
- [ ] `scene_analyzer` 已初始化（Step 4）
- [ ] `CAMERA_INTRINSICS` 已定义（Step 2）

## 🎯 验证

修复后，运行 Step 6 应该看到：

```
--- Processing frame 0 (stage 0) ---
✅ Detected objects:
  - coffee machine (conf=0.85)
  - purple cup (conf=0.92)
...

📊 Raw Scene Graph:
  Nodes: 4
  Edges: 6
```

而不是：

```
⚠️  Error processing frame 0: name 'scene_graph_builder' is not defined
📊 Raw Scene Graph:
  Nodes: 0
  Edges: 0
```

