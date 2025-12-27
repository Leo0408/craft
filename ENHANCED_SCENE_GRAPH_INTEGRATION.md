# Enhanced Scene Graph Integration - REFLECT Features in CRAFT

## 概述

已成功将 REFLECT 的优势整合到 CRAFT 框架中，同时保持了 CRAFT 的动态判断优势。

## 整合的功能

### 1. ✅ 更详细的状态信息（组合状态）

**实现位置**: `core/enhanced_scene_graph_utils.py` - `extract_composite_state()`

**功能**:
- 支持组合状态，例如：
  - `"open and filled with coffee"`
  - `"turned on and dirty"`
  - `"filled with water and clean"`
- 动态提取多个状态属性：
  - `openable` → "open" / "closed"
  - `toggleable` → "turned on" / "turned off"
  - `canFillWithLiquid` → "filled with {liquid}" / "empty"
  - `sliceable` → "not sliced"
  - `isBroken` → "not cracked"
  - `dirtyable` → "dirty" / "clean"

**使用方式**:
```python
from craft.core.enhanced_scene_graph_utils import extract_composite_state

state = extract_composite_state(obj)
# 返回: "filled with coffee and dirty" 或 None
```

**在 demo3.ipynb 中的集成**:
- 已替换原有的简单状态提取逻辑
- 现在使用 `extract_composite_state()` 提取组合状态

---

### 2. ✅ 更丰富的关系类型

**实现位置**: `core/enhanced_scene_graph_utils.py` - `add_rich_spatial_relations()`

**新增关系类型**:
1. **above** - 对象在上方
2. **below** - 对象在下方
3. **left_of** - 对象在左侧
4. **right_of** - 对象在右侧
5. **blocking** - 对象遮挡关系

**原有关系类型**（保留）:
- `holding` - 机器人持有
- `inside` - 在容器内
- `on_top_of` - 在表面上

**阈值参数**（来自 REFLECT）:
- `IN_CONTACT_DISTANCE = 0.1` (10cm) - 接触距离
- `CLOSE_DISTANCE = 0.4` (40cm) - 接近距离
- `NORM_THRESH_UP_DOWN = 0.9` - 上下方向阈值
- `NORM_THRESH_LEFT_RIGHT = 0.8` - 左右方向阈值
- `NORM_THRESH_FRONT_BACK = 0.9` - 前后方向阈值
- `OCCLUDE_RATIO_THRESH = 0.5` - 遮挡比例阈值
- `DEPTH_THRESH = 0.9` - 深度阈值

**使用方式**:
```python
from craft.core.enhanced_scene_graph_utils import add_rich_spatial_relations

add_rich_spatial_relations(
    sg, objects,
    use_point_cloud=False,  # 如果有点云数据，设置为 True
    camera_world_xyz=(x, y, z),
    rotation=rotation,
    horizon=horizon
)
```

**在 demo3.ipynb 中的集成**:
- 已在 `generate_scene_graph_from_event()` 函数末尾添加
- 自动计算相机空间向量用于方向关系判断

---

### 3. ✅ 基于点云的精确计算

**实现位置**: `core/enhanced_scene_graph_utils.py`

**功能**:
1. **点云距离计算** - `get_point_cloud_distance()`
   - 计算两个点云之间的最小距离
   - 支持 numpy array 和 torch tensor

2. **点云包含判断** - `is_inside_point_cloud()`
   - 判断源点云是否在目标点云内部
   - 使用阈值判断（默认 0.5）

3. **相机空间向量计算** - `calculate_camera_space_vector()`
   - 将世界坐标转换为相机空间坐标
   - 如果 REFLECT 工具可用，使用其转换函数
   - 否则使用简单的世界空间向量

**Node 类增强**:
- 添加了点云相关属性：
  - `pcd` - 点云数据 (N x 3)
  - `corner_pts` - 3D 边界框角点 (8 x 3)
  - `bbox2d` - 2D 边界框 (4 x 1)
  - `depth` - 深度图像

**使用方式**:
```python
# 创建带点云数据的节点
node = Node(
    name="Mug",
    object_type="Mug",
    pcd=point_cloud_data,  # N x 3 array
    corner_pts=corner_points,  # 8 x 3 array
    bbox2d=bbox_2d,  # 4 x 1 array
    depth=depth_image
)

# 在生成场景图时使用点云
add_rich_spatial_relations(
    sg, objects,
    use_point_cloud=True,  # 启用点云计算
    ...
)
```

**注意**:
- 点云数据需要从深度帧中提取（AI2THOR 的 `event.depth_frame`）
- 当前实现支持点云，但需要手动提供点云数据
- 如果 `use_point_cloud=False`，将使用位置信息进行计算

---

## 文件结构

```
craft/
├── core/
│   ├── scene_graph.py                    # 增强的 Node 类（支持点云）
│   ├── enhanced_scene_graph_utils.py     # 新增：工具函数
│   └── enhanced_generate_scene_graph.py  # 新增：增强版生成函数（参考）
└── demo3.ipynb                           # 已更新：使用增强功能
```

---

## 使用示例

### 基本使用（组合状态 + 丰富关系）

```python
from craft.core.scene_graph import SceneGraph, Node, Edge
from craft.core.enhanced_scene_graph_utils import (
    extract_composite_state,
    add_rich_spatial_relations
)

# 在 generate_scene_graph_from_event 中：
# 1. 提取组合状态
state = extract_composite_state(obj)  # "filled with coffee and dirty"

# 2. 创建节点
node = Node(name="Mug", object_type="Mug", state=state)

# 3. 添加丰富关系
add_rich_spatial_relations(sg, objects, use_point_cloud=False, ...)
```

### 使用点云数据

```python
# 如果有点云数据
node = Node(
    name="Mug",
    object_type="Mug",
    pcd=point_cloud,  # N x 3
    corner_pts=corner_points,
    bbox2d=bbox_2d,
    depth=depth_image
)

# 启用点云计算
add_rich_spatial_relations(
    sg, objects,
    use_point_cloud=True,  # 使用点云
    ...
)
```

---

## 优势总结

### ✅ 保持了 CRAFT 的优势
- **动态判断**：不依赖写死的对象类型列表
- **Action-aware**：支持 timestep 和 action 参数
- **简洁代码**：逻辑清晰，易于维护

### ✅ 整合了 REFLECT 的优势
- **组合状态**：支持多个状态的组合描述
- **丰富关系**：7 种空间关系类型（vs 原来的 3 种）
- **点云支持**：如果有点云数据，可以进行更精确的计算

### ✅ 向后兼容
- 原有代码仍然可以工作
- 新功能通过可选参数启用
- 如果没有点云数据，自动回退到位置信息计算

---

## 下一步

1. **点云提取**：如果需要，可以添加从 AI2THOR depth_frame 提取点云的函数
2. **性能优化**：如果点云数据很大，可以添加下采样
3. **测试**：添加单元测试验证新功能

---

## 更新日期

2024-12-25

