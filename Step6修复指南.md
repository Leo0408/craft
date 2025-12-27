# Step 6 和 Step 7 问题修复指南

## 🔍 问题诊断

**问题现象**：Step 7 后 scene graph 都是空的

**根本原因**：
1. `scene_graph_builder` 在 cell 12 初始化
2. Step 6 (cell 15) 使用了 `scene_graph_builder.process_frame()`
3. 但执行时 `scene_graph_builder` 未定义，导致 `NameError`
4. 异常被捕获后创建了空的 `SceneGraph`，所以所有 scene graph 都是空的

## 🔧 修复方案

### 方案 1：确保 cell 12 已运行（最简单）

**步骤**：
1. 找到 cell 12（初始化 `scene_graph_builder` 的 cell）
2. **重新运行 cell 12**
3. 确认看到输出：`✅ Scene graph builder initialized`
4. 然后运行 Step 6

### 方案 2：在 Step 6 开头添加检查代码（推荐）

在 Step 6 的代码开头（在 `scene_graphs = {}` 之前）添加以下代码：

```python
# ============================================================================
# CRITICAL: Check if scene_graph_builder is initialized
# ============================================================================
if 'scene_graph_builder' not in globals() or scene_graph_builder is None:
    print("⚠️  scene_graph_builder not found, initializing...")
    from craft.perception import ReflectSceneGraphBuilder
    
    # Check if required components are available
    if 'detector' not in globals() or detector is None:
        raise ValueError("detector is not initialized. Please run Step 4 first.")
    if 'scene_analyzer' not in globals() or scene_analyzer is None:
        raise ValueError("scene_analyzer is not initialized. Please run Step 4 first.")
    if 'CAMERA_INTRINSICS' not in globals():
        raise ValueError("CAMERA_INTRINSICS is not defined. Please run Step 2 first.")
    
    scene_graph_builder = ReflectSceneGraphBuilder(
        detector=detector,
        scene_analyzer=scene_analyzer,
        camera_intrinsics=CAMERA_INTRINSICS,
        voxel_size=0.01  # 1cm voxel size for point cloud downsampling
    )
    print("✅ Scene graph builder initialized")
```

**添加位置**：在 Step 6 的第一行代码之前，即：

```python
# Generate scene graphs for each key frame with Environment Memory integration

# [在这里添加上面的检查代码]

scene_graphs = {}
scene_graphs_with_memory = {}  # Scene graphs after memory smoothing
...
```

## ✅ 验证修复

修复后，运行 Step 6 应该看到：

**正常输出**：
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

**而不是**：
```
⚠️  Error processing frame 0: name 'scene_graph_builder' is not defined
📊 Raw Scene Graph:
  Nodes: 0
  Edges: 0
```

## 📋 检查清单

在运行 Step 6 之前，请确认：

- [ ] **Step 2** 已运行（定义 `CAMERA_INTRINSICS`）
- [ ] **Step 4** 已运行（初始化 `detector` 和 `scene_analyzer`）
- [ ] **Cell 12** 已运行（初始化 `scene_graph_builder`）
- [ ] 看到 `✅ Scene graph builder initialized` 输出

## 🎯 快速修复步骤

1. **找到 Step 6 的 cell**（应该是 cell 15）
2. **在代码开头添加检查代码**（见方案 2）
3. **重新运行 Step 6**
4. **检查输出**：应该看到检测到的对象和 scene graph 节点/边

## 💡 为什么会出现这个问题？

在 Jupyter notebook 中，如果某个 cell 没有运行，或者运行失败，变量就不会被定义。如果直接运行 Step 6 而跳过了 cell 12，就会出现 `scene_graph_builder` 未定义的问题。

添加检查代码可以：
1. 自动检测 `scene_graph_builder` 是否存在
2. 如果不存在，自动初始化
3. 提供清晰的错误信息，帮助定位问题

