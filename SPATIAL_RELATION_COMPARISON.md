# 空间关系判断逻辑对比：CRAFT vs REFLECT

## 一、总体对比

| 特性 | CRAFT (当前实现) | REFLECT |
|------|------------------|---------|
| **关系类型数量** | 3种：inside, on_top_of, holding | 7种：inside, on_top_of, above, below, left_of, right_of, blocking |
| **数据源** | AI2THOR metadata (position, parentReceptacles) | 点云 (point cloud) + 相机空间坐标 |
| **inside/on_top_of 判断** | 基于 parentReceptacles 元数据 + 动态类型判断 | 基于点云距离和边界框检查 |
| **位置关系判断** | 基于 3D position (z_diff, horizontal_dist) | 基于点云距离 + 相机空间归一化向量 |
| **表面类型判断** | 动态关键词匹配 | 写死的关键词检查 (`"countertop" in node.name`) |

---

## 二、on_top_of 关系判断逻辑对比

### 2.1 CRAFT 的实现

**位置**：`core/enhanced_generate_scene_graph.py` (Step 3, 行 148-183)

```python
# Step 3: Add on_top_of relations based on position (for objects not in containers)
for obj in objects:
    node = sg.get_node(obj.get('name', 'unknown'))
    if not node or not node.position:
        continue
    
    # Skip if already in container
    if obj.get('parentReceptacles'):
        continue
    
    obj_pos = node.position
    
    # Check if on top of other objects
    for other_obj in objects:
        if obj.get('objectId') == other_obj.get('objectId'):
            continue
        
        other_node = sg.get_node(other_obj.get('name', 'unknown'))
        if not other_node or not other_node.position:
            continue
        
        other_pos = other_node.position
        other_type = other_obj.get('objectType', '').lower()
        
        # Calculate spatial relationship
        z_diff = obj_pos[2] - other_pos[2]  # 垂直高度差
        horizontal_dist = ((obj_pos[0] - other_pos[0])**2 + (obj_pos[1] - other_pos[1])**2)**0.5  # 水平距离
        
        # Dynamic surface type detection
        is_surface = any(kw in other_type for kw in ['countertop', 'table', 'stoveburner', 'burner', 'sink'])
        
        # 判断条件：0.05 < z_diff < 0.5 且 horizontal_dist < 0.2 且 is_surface
        if (0.05 < z_diff < 0.5 and horizontal_dist < 0.2 and is_surface):
            edge_key = (node.name, other_node.name)
            existing_edge = sg.edges.get(edge_key)
            if not existing_edge or existing_edge.edge_type != 'inside':
                sg.add_edge(Edge(node, other_node, "on_top_of"))
```

**特点**：
- ✅ **基于 3D position**：使用世界坐标的 z 轴高度差和水平距离
- ✅ **动态表面类型判断**：通过关键词匹配判断是否为表面类型
- ✅ **阈值条件**：
  - 垂直高度差：`0.05 < z_diff < 0.5` (5cm - 50cm)
  - 水平距离：`horizontal_dist < 0.2` (20cm)
  - 目标必须是表面类型（countertop, table, stoveburner, burner, sink）

**优先级**：
1. 先处理 `parentReceptacles`（Step 2）：如果是容器 → inside，如果是表面 → on_top_of
2. 再处理位置关系（Step 3）：只对**不在容器内**的对象判断 on_top_of

---

### 2.2 REFLECT 的实现

**位置**：`reflect/main/scene_graph.py` (`add_edge` 方法, 行 227-268)

```python
def add_edge(self, node, new_node):
    # 1. 坐标转换到相机空间
    pos_A = world_space_xyz_to_camera_space_xyz(node.pos3d, self.camera_world_xyz, self.rotation, self.horizon)
    pos_B = world_space_xyz_to_camera_space_xyz(new_node.pos3d, self.camera_world_xyz, self.rotation, self.horizon)
    cam_arr = pos_B - pos_A
    norm_vector = cam_arr / np.linalg.norm(cam_arr)
    
    # 2. 计算点云距离
    dist = get_pcd_dist(node.pcd, new_node.pcd)
    
    box_A, box_B = np.array(node.corner_pts), np.array(new_node.corner_pts)
    box_A_pts, box_B_pts = np.array(node.pcd), np.array(new_node.pcd)
    
    # IN CONTACT relations (distance < 0.1m)
    if dist < IN_CONTACT_DISTANCE:  # 0.1m = 10cm
        # 情况 1：检查 inside 关系（基于点云）
        if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):  # thresh=0.5
            # 特殊情况：countertop 或 stove burner → on_top_of（而不是 inside）
            if "countertop" in node.name or "stove burner" in node.name:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
            else:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
        
        # 情况 2：检查 on_top_of 关系（基于边界框和点云）
        elif len(np.where((box_B_pts[:, 0] < box_A[4, 0]) & (box_B_pts[:, 0] > box_A[0, 0]) & 
                (box_B_pts[:, 2] < box_A[4, 2]) & (box_B_pts[:, 2] > box_A[0, 2]))[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:  # 70%
            # 检查垂直方向：box_B 的点是否在 box_A 上方
            if len(np.where(box_B_pts[:, 1] > box_A[4, 1])[0]) > len(box_B_pts) * ON_TOP_OF_THRESH:  # 70%
                # 特殊情况：切片的水果在 bowl 内 → inside（而不是 on_top_of）
                if 'slice' in new_node.name and node.name == 'bowl':
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
                else:
                    self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
            # 反向检查：box_A 的点是否在 box_B 上方
            elif len(np.where(box_A_pts[:, 1] > box_B[4, 1])[0]) > len(box_A_pts) * ON_TOP_OF_THRESH:  # 70%
                if node.name not in BULKY_OBJECTS:
                    if 'slice' in node.name and new_node.name == 'bowl':
                        self.edges[(node.name, new_node.name)] = Edge(node, new_node, "inside")
                    else:
                        self.edges[(node.name, new_node.name)] = Edge(node, new_node, "on top of")
```

**特点**：
- ✅ **基于点云距离**：使用 `get_pcd_dist()` 计算两个点云之间的最小距离
- ✅ **基于边界框检查**：
  - 检查 box_B 的点是否在 box_A 的 x-z 平面范围内（70% 阈值）
  - 检查 box_B 的点是否在 box_A 上方（y 轴，70% 阈值）
- ❌ **写死的类型判断**：
  - `"countertop" in node.name` 或 `"stove burner" in node.name` → on_top_of
  - `'slice' in new_node.name and node.name == 'bowl'` → inside（特殊情况）
- ✅ **相机空间坐标**：先将世界坐标转换到相机空间，然后计算归一化向量

**阈值**：
- `IN_CONTACT_DISTANCE = 0.1m` (10cm)：只有当距离 < 10cm 时才判断 inside/on_top_of
- `ON_TOP_OF_THRESH = 0.7` (70%)：需要 70% 的点满足条件
- `INSIDE_THRESH = 0.5` (50%)：inside 关系需要 50% 的点在目标内部

---

## 三、inside 关系判断逻辑对比

### 3.1 CRAFT 的实现

**位置**：`core/enhanced_generate_scene_graph.py` (Step 2, 行 125-146)

```python
# Container relations (parentReceptacles)
if obj.get('parentReceptacles'):
    for parent_id in obj.get('parentReceptacles', []):
        for other_obj in objects:
            if other_obj.get('objectId') == parent_id:
                parent_node = sg.get_node(other_obj.get('name', 'unknown'))
                if parent_node:
                    # Dynamic relation type judgment
                    has_receptacle = bool(other_obj.get('receptacleObjectIds', []))
                    is_openable_container = 'isOpen' in other_obj or other_obj.get('openable', False)
                    receptacle_count = len(other_obj.get('receptacleObjectIds', [])) if isinstance(other_obj.get('receptacleObjectIds'), list) else 0
                    
                    # 判断逻辑：
                    # - 如果是可打开的容器 OR 有 receptacleObjectIds → inside
                    # - 否则 → on_top_of（表面类型）
                    if is_openable_container or (has_receptacle and receptacle_count > 0):
                        edge_key = (node.name, parent_node.name)
                        if edge_key not in sg.edges:
                            sg.add_edge(Edge(node, parent_node, "inside"))
                    else:
                        edge_key = (node.name, parent_node.name)
                        if edge_key not in sg.edges:
                            sg.add_edge(Edge(node, parent_node, "on_top_of"))
```

**特点**：
- ✅ **基于 AI2THOR metadata**：使用 `parentReceptacles` 和 `receptacleObjectIds`
- ✅ **动态判断**：
  - `is_openable_container`：检查对象是否有 `isOpen` 属性或 `openable=True`
  - `has_receptacle`：检查对象是否有 `receptacleObjectIds`（表示它可以容纳其他对象）
- ✅ **不依赖位置**：完全基于元数据，不需要位置信息

---

### 3.2 REFLECT 的实现

**位置**：`reflect/main/scene_graph.py` (`add_edge` 方法, 行 246-253)

```python
# IN CONTACT relations (distance < 0.1m)
if dist < IN_CONTACT_DISTANCE:  # 0.1m
    # 检查 inside 关系
    if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):  # 0.5
        # 特殊情况：countertop 或 stove burner → on_top_of（而不是 inside）
        if "countertop" in node.name or "stove burner" in node.name:
            self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
        else:
            self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
```

**辅助函数** `is_inside`：
```python
def is_inside(src_pts, target_pts, thresh=0.5):
    # 获取 target 的边界框
    target_min = np.min(target_pts, axis=0)
    target_max = np.max(target_pts, axis=0)
    
    # 计算 src 中有多少点在 target 的边界框内
    inside_mask = np.all((src_pts >= target_min) & (src_pts <= target_max), axis=1)
    inside_ratio = np.sum(inside_mask) / len(src_pts)
    
    # 如果 >= 50% 的点在内部 → inside
    return inside_ratio >= thresh
```

**特点**：
- ✅ **基于点云边界框**：检查源对象的点有多少在目标对象的边界框内
- ✅ **阈值判断**：需要 50% 的点在目标内部才判定为 inside
- ❌ **写死的特殊情况**：countertop 和 stove burner 被硬编码为 on_top_of
- ✅ **距离限制**：只有当点云距离 < 10cm 时才判断 inside

---

## 四、其他位置关系对比

### 4.1 CRAFT 的实现

**位置**：`core/enhanced_scene_graph_utils.py` (`add_rich_spatial_relations` 函数)

CRAFT 支持额外的关系类型（如果启用 `use_rich_relations=True`）：

1. **above / below**：
```python
# CLOSE TO relations (distance < 0.4m)
if dist < CLOSE_DISTANCE:  # 0.4m
    norm_vector = calculate_camera_space_vector(...)
    
    if abs(norm_vector[1]) > NORM_THRESH_UP_DOWN:  # 0.9
        if norm_vector[1] > 0:
            sg.add_edge(Edge(node1, node2, "above"))
        else:
            sg.add_edge(Edge(node1, node2, "below"))
```

2. **left_of / right_of**：
```python
elif abs(norm_vector[0]) > NORM_THRESH_LEFT_RIGHT:  # 0.8
    if norm_vector[0] > 0:
        sg.add_edge(Edge(node1, node2, "right_of"))
    else:
        sg.add_edge(Edge(node1, node2, "left_of"))
```

3. **blocking**：
```python
elif abs(norm_vector[2]) > NORM_THRESH_FRONT_BACK:  # 0.9
    # 计算 2D bbox 的 IoU 和遮挡比例
    occlude_ratio = inters / area2
    depth_occlude = np.sum(depth1 <= np.min(depth2)) / len(depth1)
    
    if occlude_ratio > OCCLUDE_RATIO_THRESH and depth_occlude > DEPTH_THRESH:
        sg.add_edge(Edge(node1, node2, "blocking"))
```

**阈值**：
- `CLOSE_DISTANCE = 0.4m` (40cm)
- `NORM_THRESH_UP_DOWN = 0.9`
- `NORM_THRESH_LEFT_RIGHT = 0.8`
- `NORM_THRESH_FRONT_BACK = 0.9`
- `OCCLUDE_RATIO_THRESH = 0.5` (50%)
- `DEPTH_THRESH = 0.9` (90%)

---

### 4.2 REFLECT 的实现

**位置**：`reflect/main/scene_graph.py` (`add_edge` 方法, 行 270-289)

REFLECT 也支持相同的关系类型，逻辑几乎相同：

```python
# CLOSE TO relations (distance < 0.4m)
if dist < CLOSE_DISTANCE and (new_node.name, node.name) not in self.edges:
    if node.name not in BULKY_OBJECTS and new_node.name not in BULKY_OBJECTS:
        # Above/Below
        if abs(norm_vector[1]) > NORM_THRESH_UP_DOWN:  # 0.9
            if norm_vector[1] > 0:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "above")
            else:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "below")
        
        # Left/Right
        elif abs(norm_vector[0]) > NORM_THRESH_LEFT_RIGHT:  # 0.8
            if norm_vector[0] > 0:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the right of")
            else:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on the left of")
        
        # Blocking
        elif abs(norm_vector[2]) > NORM_THRESH_FRONT_BACK:  # 0.9
            iou, inters = get_iou(new_node.bbox2d, node.bbox2d)
            occlude_ratio = inters / ((node.bbox2d[2]-node.bbox2d[0]) * (node.bbox2d[3]-node.bbox2d[1]))
            
            if occlude_ratio > OCCLUDE_RATIO_THRESH and len(np.where(new_node.depth <= np.min(node.depth))[0]) > len(new_node.depth) * DEPTH_THRESH:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "blocking")
```

**差异**：
- REFLECT 检查 `BULKY_OBJECTS`（大物体），避免为这些物体添加方向关系
- REFLECT 的关系名称略有不同：`"on the right of"` vs CRAFT 的 `"right_of"`

---

## 五、关键差异总结

### 5.1 on_top_of 判断

| 方面 | CRAFT | REFLECT |
|------|-------|---------|
| **数据源** | 3D position (world space) | Point cloud (相机空间) |
| **判断条件** | z_diff (0.05-0.5m) + horizontal_dist (<0.2m) + is_surface | 点云距离 (<0.1m) + 边界框检查 (70% 阈值) |
| **表面类型判断** | 动态关键词匹配 | 写死：`"countertop" in node.name` |
| **特殊情况** | 无 | `'slice' in name and node.name == 'bowl'` → inside |
| **优先级** | 先处理 parentReceptacles，再处理位置关系 | 先判断 inside，再判断 on_top_of |

### 5.2 inside 判断

| 方面 | CRAFT | REFLECT |
|------|-------|---------|
| **数据源** | AI2THOR metadata (`parentReceptacles`, `receptacleObjectIds`) | Point cloud 边界框 |
| **判断方法** | 元数据属性检查 | 点云点的边界框包含检查 (50% 阈值) |
| **动态性** | ✅ 完全动态（基于对象属性） | ❌ 部分写死（countertop/stove burner 例外） |
| **准确性** | 高（基于 ground truth metadata） | 依赖于点云质量和边界框准确性 |

### 5.3 优缺点分析

#### CRAFT 的优势：
1. ✅ **动态判断**：不依赖写死的对象类型列表
2. ✅ **基于元数据**：inside/on_top_of 判断使用 AI2THOR 的 ground truth metadata
3. ✅ **简洁的逻辑**：代码清晰，易于理解和维护
4. ✅ **可扩展性**：自动适应新对象类型

#### CRAFT 的劣势：
1. ❌ **需要位置信息**：on_top_of 判断依赖 3D position，如果 position 不准确会影响结果
2. ❌ **阈值固定**：z_diff 和 horizontal_dist 的阈值是固定的，可能不适合所有场景
3. ❌ **关系类型较少**：只有 3 种基本关系（holding, inside, on_top_of）

#### REFLECT 的优势：
1. ✅ **基于点云**：使用点云数据可以更精确地计算空间关系
2. ✅ **关系类型丰富**：支持 7 种关系（inside, on_top_of, above, below, left_of, right_of, blocking）
3. ✅ **边界框检查**：使用 3D 边界框可以更准确地判断 inside 关系

#### REFLECT 的劣势：
1. ❌ **写死的类型判断**：countertop、stove burner 等被硬编码
2. ❌ **需要点云数据**：必须有点云数据才能准确计算
3. ❌ **复杂度高**：边界框检查和相机空间转换增加复杂度
4. ❌ **特殊情况处理**：需要为特殊情况（如 slice in bowl）写死逻辑

---

## 六、建议

### 6.1 结合两者优势

1. **inside/on_top_of 判断**：
   - **优先使用 CRAFT 的方法**（基于 `parentReceptacles` metadata）
   - **备用 REFLECT 的方法**（基于点云边界框），当 metadata 不可用时

2. **on_top_of 位置判断**：
   - **保留 CRAFT 的动态表面类型判断**
   - **添加 REFLECT 的点云距离检查**，提高准确性

3. **其他位置关系**：
   - **保留 CRAFT 的 rich relations**（above, below, left_of, right_of, blocking）
   - **统一关系名称**（CRAFT 使用下划线，REFLECT 使用空格）

### 6.2 改进方向

1. **自适应阈值**：
   - 根据场景类型（厨房、客厅等）调整阈值
   - 根据对象类型（小物体、大物体）调整阈值

2. **混合判断**：
   - 如果 metadata 可用，优先使用 metadata
   - 如果 metadata 不可用或不确定，使用点云/位置信息

3. **置信度分数**：
   - 为每个关系添加置信度分数
   - 基于 metadata 的关系置信度 = 1.0
   - 基于位置/点云的关系置信度 < 1.0

---

## 七、代码示例：改进的混合方法

```python
def determine_spatial_relation(obj1, obj2, node1, node2, use_point_cloud=False):
    """
    混合方法：优先使用 metadata，备用位置/点云信息
    """
    # 优先级 1：基于 parentReceptacles metadata（最可靠）
    if obj1.get('parentReceptacles'):
        for parent_id in obj1.get('parentReceptacles', []):
            if obj2.get('objectId') == parent_id:
                # 使用 CRAFT 的动态判断逻辑
                has_receptacle = bool(obj2.get('receptacleObjectIds', []))
                is_openable = obj2.get('openable', False) or 'isOpen' in obj2
                
                if is_openable or has_receptacle:
                    return ("inside", 1.0)  # 置信度 1.0
                else:
                    return ("on_top_of", 1.0)  # 置信度 1.0
    
    # 优先级 2：基于位置信息（如果 metadata 不可用）
    if node1.position and node2.position:
        z_diff = node1.position[2] - node2.position[2]
        horizontal_dist = ((node1.position[0] - node2.position[0])**2 + 
                          (node1.position[1] - node2.position[1])**2)**0.5
        
        # 使用 CRAFT 的逻辑
        other_type = obj2.get('objectType', '').lower()
        is_surface = any(kw in other_type for kw in ['countertop', 'table', 'stoveburner', 'burner', 'sink'])
        
        if (0.05 < z_diff < 0.5 and horizontal_dist < 0.2 and is_surface):
            return ("on_top_of", 0.85)  # 置信度 0.85（比 metadata 低）
    
    # 优先级 3：基于点云（如果可用且前两者都不可用）
    if use_point_cloud and node1.pcd is not None and node2.pcd is not None:
        dist = get_point_cloud_distance(node1.pcd, node2.pcd)
        if dist < IN_CONTACT_DISTANCE:
            if is_inside_point_cloud(node1.pcd, node2.pcd, INSIDE_THRESH):
                # 使用 REFLECT 的逻辑，但保持动态判断
                other_type = obj2.get('objectType', '').lower()
                is_surface = any(kw in other_type for kw in ['countertop', 'stoveburner', 'burner'])
                if is_surface:
                    return ("on_top_of", 0.75)  # 置信度 0.75
                else:
                    return ("inside", 0.75)
    
    return None
```

