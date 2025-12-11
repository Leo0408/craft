# Demo2 真实环境方法说明文档

本文档说明 `demo2.ipynb` 中使用的真实环境方法和与 CRAFT 框架的融合流程。

## 目录

1. [概述](#概述)
2. [真实环境数据加载](#真实环境数据加载)
3. [目标检测方法 (MDETR)](#目标检测方法-mdetr)
4. [场景图生成流程](#场景图生成流程)
5. [与 CRAFT 框架的融合](#与-craft-框架的融合)
6. [关键差异与注意事项](#关键差异与注意事项)

---

## 概述

`demo2.ipynb` 展示了如何在真实机器人执行环境中应用 CRAFT++ 框架。与 `demo1.ipynb`（模拟环境）的主要区别在于：

- **数据源**: 使用 REFLECT 真实世界数据集（zarr 格式），而非 AI2THOR 模拟数据
- **目标检测**: 使用 MDETR 真实检测器，而非模拟检测器
- **场景图构建**: 从真实 RGB-D 图像生成，而非从模拟器元数据提取
- **点云处理**: 真实的深度图到点云转换和 3D 边界框计算

---

## 真实环境数据加载

### 1. REFLECT 数据集结构

REFLECT 真实世界数据集采用 zarr 格式存储，结构如下：

```
reflect_dataset/real_data/{task_folder}/
├── replay_buffer.zarr/          # Zarr 格式的机器人执行数据
│   ├── data/
│   │   ├── stage/               # 动作阶段信息
│   │   ├── gripper_pos/         # 夹爪位置
│   │   └── videos/
│   │       ├── color/           # RGB 图像: {step_idx}.0.0.0
│   │       └── depth/           # 深度图像: {step_idx}.0.0
└── videos/                      # 文件系统格式（备选）
    ├── color/
    └── depth/
```

### 2. 数据加载方法

使用 `ReflectDataLoader` 类加载数据：

```python
from craft.utils import ReflectDataLoader

# 初始化数据加载器
data_loader = ReflectDataLoader(data_root="/path/to/reflect")

# 加载任务信息
task_info = data_loader.load_task_info("makeCoffee2")

# 加载 zarr 文件
zarr_group = data_loader.load_zarr_file("makeCoffee2")

# 加载特定帧的 RGB 和深度图像
rgb = data_loader.load_frame_rgb(zarr_group, step_idx=100, folder_name="makeCoffee2")
depth = data_loader.load_frame_depth(zarr_group, step_idx=100, folder_name="makeCoffee2")
```

### 3. 关键帧选择

真实环境中，不是所有帧都需要处理。关键帧选择策略：

- **动作转换帧**: 在动作阶段变化时选择前后几帧
- **固定间隔采样**: 每隔 N 帧采样一次
- **基于动作阶段**: 每个动作阶段选择开始、中间、结束帧

```python
# 从 zarr 获取动作阶段
stages = np.array(zarr_group['data/stage'])

# 选择动作转换帧
key_frames = [0]  # 第一帧
prev_stage = stages[0]
for i in range(1, len(stages)):
    if stages[i] != prev_stage:
        key_frames.append(i - 10)  # 转换前
        key_frames.append(i + 10)  # 转换后
        prev_stage = stages[i]
```

---

## 目标检测方法 (MDETR)

### 1. MDETR 简介

MDETR (Modulated Detection for End-to-End Multi-Modal Understanding) 是 REFLECT 框架中使用的目标检测器，具有以下特点：

- **文本引导检测**: 使用自然语言描述进行目标检测（如 "purple cup", "coffee machine"）
- **实例分割**: 不仅提供边界框，还提供像素级掩码
- **端到端训练**: 检测和分割联合训练

### 2. MDETR 检测器包装器

为了在 CRAFT 框架中使用 MDETR，我们创建了 `MDETRDetector` 包装器（`craft/perception/mdetr_detector.py`）：

```python
from craft.perception.mdetr_detector import MDETRDetector

# 初始化 MDETR 检测器
detector = MDETRDetector(
    device="cuda:0",
    threshold=0.7,  # 检测置信度阈值
    pretrained=True
)

# 检测物体
detections = detector.detect_objects(rgb_image, object_list=["coffee machine", "purple cup"])

# 带深度信息的检测（用于 3D 定位）
detections_3d = detector.detect_with_depth(
    rgb_image, 
    depth_image, 
    object_list, 
    camera_intrinsics
)
```

### 3. 检测输出格式

MDETR 检测器返回的检测结果格式：

```python
detection = {
    'label': 'coffee machine',      # 物体名称
    'bbox': [x1, y1, x2, y2],       # 2D 边界框
    'mask': np.ndarray,              # 像素级掩码（布尔数组）
    'confidence': 0.95,              # 检测置信度
    'position_3d': (x, y, z)         # 3D 位置（如果使用深度）
}
```

### 4. 检测阈值调整

真实环境中，检测阈值可能需要调整：

- **默认阈值**: 0.96（REFLECT 原始设置）
- **降低阈值**: 0.7（提高召回率，但可能增加误检）
- **任务相关**: 不同任务可能需要不同阈值

```python
# 降低阈值以提高检测率
detector = MDETRDetector(threshold=0.7)
```

---

## 场景图生成流程

### 1. 整体流程

真实环境中的场景图生成流程：

```
RGB-D 图像
    ↓
MDETR 目标检测 → 2D 边界框 + 掩码
    ↓
深度图到点云转换 → 3D 点云
    ↓
点云处理（降采样、去噪） → 清理后的点云
    ↓
3D 边界框计算 → 3D 位置和尺寸
    ↓
空间关系计算 → 空间关系（inside, on_top_of, near 等）
    ↓
场景图构建 → SceneGraph 对象
```

### 2. 深度图到点云转换

使用相机内参将深度图转换为 3D 点云：

```python
def depth_to_point_cloud(depth, mask, camera_intrinsics):
    """
    将深度图转换为点云
    
    Args:
        depth: 深度图 (H x W)
        mask: 物体掩码（可选）
        camera_intrinsics: 相机内参 {fx, fy, cx, cy}
    
    Returns:
        点云数组 (N x 3)
    """
    h, w = depth.shape
    fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
    cx, cy = camera_intrinsics['cx'], camera_intrinsics['cy']
    
    # 创建坐标网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # 应用掩码（如果提供）
    if mask is not None:
        valid_mask = (depth > 0) & mask
    else:
        valid_mask = depth > 0
    
    # 提取有效像素
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    z_valid = depth[valid_mask]
    
    # 转换为 3D 坐标
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    
    # 堆叠为点云
    points = np.stack([x, y, z_valid], axis=1)
    return points
```

### 3. 点云处理

使用 Open3D 进行点云处理：

```python
import open3d as o3d

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 体素下采样（减少点数）
voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.01)  # 1cm 体素

# 统计离群点去除（去噪）
_, ind = voxel_down_pcd.remove_statistical_outlier(
    nb_neighbors=20, 
    std_ratio=0.1
)
inlier = voxel_down_pcd.select_by_index(ind)
```

### 4. 3D 边界框计算

从点云计算 3D 轴对齐边界框：

```python
# 计算 3D 边界框
boxes3d_pts = o3d.utility.Vector3dVector(points)
bbox3d = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)

# 获取边界框中心
center = bbox3d.get_center()

# 获取边界框尺寸
extent = bbox3d.get_extent()
```

### 5. 空间关系计算

基于 3D 位置和边界框计算空间关系：

```python
from craft.perception import SceneAnalyzer

scene_analyzer = SceneAnalyzer()

# 计算空间关系
relations = scene_analyzer.compute_spatial_relations(detections)

# 关系类型包括：
# - "inside": 物体 A 在物体 B 内部
# - "on_top_of": 物体 A 在物体 B 上方
# - "near": 物体 A 靠近物体 B
# - "none": 无明显关系
```

### 6. 场景图构建

使用 `ReflectSceneGraphBuilder` 构建场景图：

```python
from craft.perception import ReflectSceneGraphBuilder

# 初始化场景图构建器
builder = ReflectSceneGraphBuilder(
    detector=detector,
    scene_analyzer=scene_analyzer,
    camera_intrinsics=camera_intrinsics,
    voxel_size=0.01
)

# 处理单帧并生成场景图
scene_graph = builder.process_frame(
    rgb=rgb_image,
    depth=depth_image,
    step_idx=frame_idx,
    object_list=["coffee machine", "purple cup"],
    distractor_list=[],
    task_info=task_info
)
```

---

## 与 CRAFT 框架的融合

### 1. 数据流融合

真实环境数据流与 CRAFT 框架的融合：

```
REFLECT 数据集 (zarr)
    ↓
ReflectDataLoader → 加载 RGB-D 数据
    ↓
MDETRDetector → 目标检测
    ↓
ReflectSceneGraphBuilder → 场景图生成
    ↓
CRAFT SceneGraph → 统一场景图格式
    ↓
ConstraintGenerator → 约束生成（LLM）
    ↓
ConstraintCompiler → 约束编译
    ↓
Failure Detection → 失败检测
    ↓
FailureAnalyzer → 失败分析
```

### 2. 场景图格式统一

REFLECT 的场景图格式与 CRAFT 的场景图格式统一：

```python
# CRAFT SceneGraph 格式
scene_graph = SceneGraph(task=task_info)

# 节点 (Node)
node = Node(
    name="coffee machine",
    object_type="coffee machine",
    position=(x, y, z),  # 3D 位置
    attributes={
        'bbox3d': bbox3d,
        'bbox2d': bbox2d,
        'pcd': point_cloud
    }
)

# 边 (Edge)
edge = Edge(
    start=node1,
    end=node2,
    edge_type="inside",  # 空间关系
    confidence=0.95
)
```

### 3. 约束生成与验证

场景图生成后，后续流程与 demo1 相同：

1. **约束生成**: 使用 LLM 从场景图和任务信息生成约束
2. **约束编译**: 将约束编译为可执行代码
3. **失败检测**: 验证约束是否满足
4. **失败分析**: 生成详细的失败解释

---

## 关键差异与注意事项

### 1. 与模拟环境的差异

| 方面 | 模拟环境 (Demo1) | 真实环境 (Demo2) |
|------|-----------------|------------------|
| **数据源** | AI2THOR events | REFLECT zarr 文件 |
| **目标检测** | 从 metadata 提取 | MDETR 检测器 |
| **3D 位置** | 直接从 metadata | 深度图到点云转换 |
| **遮挡处理** | 无（完全可见） | 需要处理遮挡 |
| **噪声处理** | 无噪声 | 需要去噪 |
| **检测失败** | 不会发生 | 可能检测不到物体 |

### 2. 性能考虑

- **检测速度**: MDETR 检测较慢，建议只处理关键帧
- **内存使用**: 点云数据占用大量内存，需要及时释放
- **GPU 使用**: MDETR 需要 GPU，确保 CUDA 可用

### 3. 常见问题

#### 问题 1: MDETR 检测不到物体

**解决方案**:
- 降低检测阈值（从 0.96 到 0.7）
- 检查物体名称是否与训练数据匹配
- 尝试不同的物体描述

#### 问题 2: 点云为空或点数过少

**解决方案**:
- 检查深度图是否有效（非零值）
- 检查掩码是否正确
- 调整体素下采样参数

#### 问题 3: 3D 位置不准确

**解决方案**:
- 检查相机内参是否正确
- 使用掩码内的中位数深度而非中心点深度
- 考虑使用点云中心而非边界框中心

### 4. 最佳实践

1. **关键帧选择**: 不要处理所有帧，选择有代表性的关键帧
2. **检测阈值**: 根据任务调整阈值，平衡召回率和精确率
3. **点云处理**: 使用体素下采样和去噪提高质量
4. **错误处理**: 添加异常处理，避免单个帧失败影响整个流程
5. **可视化**: 可视化检测结果和场景图，便于调试

---

## 后续融合流程

### 1. 完整工作流

```
Step 1: 数据加载
    ↓
Step 2: 模型初始化（MDETR, SceneAnalyzer, LLM）
    ↓
Step 3: 关键帧选择
    ↓
Step 4: 场景图生成（MDETR + 点云 + 空间关系）
    ↓
Step 5: 约束生成（LLM）
    ↓
Step 6: 约束编译（AST/DSL）
    ↓
Step 7: 失败检测（约束验证）
    ↓
Step 8: 失败分析（因果链、根因）
```

### 2. 与 CRAFT 其他模块的集成

- **TaskExecutor**: 可以基于失败分析结果生成修正计划
- **CorrectionPlanner**: 生成具体的修正动作序列
- **Environment Memory**: 跟踪物体状态变化，处理遮挡

### 3. 扩展方向

- **多模态融合**: 结合音频事件检测（AudioCLIP）
- **时序建模**: 跟踪物体状态随时间的变化
- **不确定性处理**: 处理检测和定位的不确定性
- **在线学习**: 从失败中学习，改进检测和约束

---

## 总结

`demo2.ipynb` 展示了如何在真实机器人执行环境中应用 CRAFT++ 框架。关键点包括：

1. **使用 REFLECT 真实数据集**: 从 zarr 文件加载 RGB-D 数据
2. **MDETR 目标检测**: 使用真实检测器进行文本引导的目标检测
3. **点云处理**: 从深度图生成 3D 点云并计算空间关系
4. **场景图构建**: 构建包含 3D 信息的场景图
5. **CRAFT 框架融合**: 与约束生成、失败检测等模块无缝集成

通过这种方式，CRAFT++ 框架可以同时支持模拟环境和真实环境，为机器人失败检测和修正提供统一的解决方案。

