# Scene Graph 生成逻辑对比：demo3.ipynb vs reflect/main/scene_graph.py

## 一、整体架构对比

### demo3.ipynb (CRAFT 版本)
- **函数式设计**：使用独立的 `generate_scene_graph_from_event()` 函数
- **Action-aware**：支持 `timestep` 和 `action` 参数，用于记录场景图对应的时间步和动作
- **使用统一的 SceneGraph 类**：来自 `craft.core.scene_graph`

### reflect/main/scene_graph.py (REFLECT 版本)
- **类式设计**：SceneGraph 类在初始化时接收 event 和 task
- **无 Action-aware**：不记录时间步和动作信息
- **自定义 Node/Edge 类**：在同一个文件中定义

---

## 二、节点（Node）创建对比

### demo3.ipynb
```python
# 动态提取对象状态（不依赖写死的类型列表）
state = None
if obj.get('isOpen') is not None:
    state = 'open' if obj.get('isOpen', False) else 'closed'
elif obj.get('isToggledOn') is not None:
    state = 'on' if obj.get('isToggledOn', False) else 'off'
elif obj.get('isFilled') is not None:
    state = 'filled' if obj.get('isFilled', False) else 'empty'

obj_type = obj.get('objectType', '')

node = Node(
    name=obj.get('name', 'unknown'),
    object_type=obj_type,
    state=state,
    position=tuple(obj.get('position', {}).values()) if isinstance(obj.get('position'), dict) else None,
    attributes={
        'isFilled': obj.get('isFilled', False),
        'isOpen': obj.get('isOpen', False),
        'isToggled': obj.get('isToggledOn', False),
        'isPickedUp': obj.get('isPickedUp', False),
    }
)
```

**特点**：
- ✅ **动态状态提取**：根据对象的实际属性动态判断状态，不依赖写死的对象类型列表
- ✅ **简洁的状态判断**：只检查三个主要属性（isOpen, isToggledOn, isFilled）
- ✅ **统一的数据结构**：使用标准的 Node 类

### reflect/main/scene_graph.py
```python
# 使用写死的 state_dict
state_dict = {
    "Fridge": ["open", "closed"],
    "Faucet": ["turned on", "turned off"],
    "Pot": ["filled with coffee", "filled with water", "filled with wine", "empty", "dirty", "clean"],
    # ... 更多写死的类型
}

def get_gt_object_state(node_name, event):
    object_name = node_name.split("|")[0]
    if object_name in state_dict and len(node_name.split("|")) == 4:
        gt_states = []
        for obj in event.metadata["objects"]:
            if node_name == obj["objectId"]:
                if obj["openable"]:
                    if obj["isOpen"]:
                        gt_states.append("open")
                    else:
                        gt_states.append("closed")
                if obj["sliceable"]:
                    if not obj["isSliced"]:
                        gt_states.append("not sliced")
                # ... 更多条件判断
        return " and ".join(gt_states)
    return None

# Node 创建（在 add_node 方法中）
node = Node(name, object_id, pos3d, corner_pts, bbox2d, pcd, depth, global_node)
```

**特点**：
- ❌ **写死的类型列表**：需要维护 state_dict，无法自动适应新对象类型
- ❌ **复杂的状态判断**：需要检查多个属性（openable, sliceable, canFillWithLiquid, toggleable, dirtyable）
- ✅ **更丰富的状态信息**：支持组合状态（如 "filled with coffee and dirty"）
- ✅ **支持点云数据**：Node 包含 pcd（点云）、corner_pts（3D边界框角点）等

---

## 三、关系（Edge）生成对比

### demo3.ipynb

#### 1. holding 关系
```python
if obj.get('isPickedUp', False):
    robot_node = sg.get_node("Robot")
    if not robot_node:
        robot_node = Node(name="Robot", object_type="Robot")
        sg.add_node(robot_node)
    sg.add_edge(Edge(robot_node, node, "holding"))
```

#### 2. inside/on_top_of 关系（基于 parentReceptacles）
```python
if obj.get('parentReceptacles'):
    for parent_id in obj.get('parentReceptacles', []):
        for other_obj in objects:
            if other_obj.get('objectId') == parent_id:
                parent_node = sg.get_node(other_obj.get('name', 'unknown'))
                if parent_node:
                    # 动态判断关系类型
                    has_receptacle = bool(other_obj.get('receptacleObjectIds', []))
                    is_openable_container = 'isOpen' in other_obj
                    receptacle_count = len(other_obj.get('receptacleObjectIds', []))
                    
                    if is_openable_container or (has_receptacle and receptacle_count > 0):
                        sg.add_edge(Edge(node, parent_node, "inside"))
                    else:
                        sg.add_edge(Edge(node, parent_node, "on_top_of"))
```

**特点**：
- ✅ **基于元数据**：使用 AI2THOR 的 `parentReceptacles` 和 `receptacleObjectIds` 属性
- ✅ **动态判断**：不依赖写死的对象类型列表
- ✅ **简洁的逻辑**：只生成 inside 和 on_top_of 两种关系

#### 3. on_top_of 关系（基于位置信息）
```python
# 对于不在容器内的对象，基于位置信息判断 on_top_of
for obj in objects:
    if obj.get('parentReceptacles'):
        continue  # 跳过已在容器内的对象
    
    obj_pos = node.position
    for other_obj in objects:
        other_pos = other_node.position
        z_diff = obj_pos[2] - other_pos[2]
        horizontal_dist = ((obj_pos[0] - other_pos[0])**2 + (obj_pos[1] - other_pos[1])**2)**0.5
        
        # 动态判断表面类型
        def is_surface_type_dynamic(obj_type: str) -> bool:
            type_lower = obj_type.lower()
            surface_keywords = ['countertop', 'table', 'stoveburner', 'burner', 'sink']
            return any(kw in type_lower for kw in surface_keywords)
        
        is_surface = is_surface_type_dynamic(other_type)
        
        if (0.05 < z_diff < 0.5 and horizontal_dist < 0.2 and is_surface):
            sg.add_edge(Edge(node, other_node, "on_top_of"))
```

**特点**：
- ✅ **基于位置计算**：使用 3D 位置信息判断空间关系
- ✅ **动态表面判断**：通过关键词匹配判断表面类型
- ✅ **严格的条件**：高度差和水平距离都有阈值限制

### reflect/main/scene_graph.py

#### 关系生成（在 add_edge 方法中）
```python
def add_edge(self, node, new_node):
    # 1. 坐标转换到相机空间
    pos_A = world_space_xyz_to_camera_space_xyz(...)
    pos_B = world_space_xyz_to_camera_space_xyz(...)
    cam_arr = pos_B - pos_A
    norm_vector = cam_arr / np.linalg.norm(cam_arr)
    
    # 2. 计算点云距离
    dist = get_pcd_dist(node.pcd, new_node.pcd)
    
    # 3. IN CONTACT 关系（距离 < 0.1m）
    if dist < IN_CONTACT_DISTANCE:
        if is_inside(src_pts=box_B_pts, target_pts=box_A_pts, thresh=INSIDE_THRESH):
            if "countertop" in node.name or "stove burner" in node.name:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "on top of")
            else:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "inside")
        elif # 检查 on_top_of 条件（基于点云和边界框）
            # 使用 ON_TOP_OF_THRESH = 0.7 判断
            ...
    
    # 4. CLOSE TO 关系（距离 < 0.4m）
    if dist < CLOSE_DISTANCE:
        if abs(norm_vector[1]) > NORM_THRESH_UP_DOWN:
            if norm_vector[1] > 0:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "above")
            else:
                self.edges[(new_node.name, node.name)] = Edge(new_node, node, "below")
        elif abs(norm_vector[0]) > NORM_THRESH_LEFT_RIGHT:
            # "on the right of" / "on the left of"
        elif abs(norm_vector[2]) > NORM_THRESH_FRONT_BACK:
            # "blocking" 关系（基于遮挡）
```

**特点**：
- ✅ **基于点云**：使用点云数据计算距离和空间关系
- ✅ **更丰富的关系类型**：支持 inside, on_top_of, above, below, on_the_right_of, on_the_left_of, blocking
- ✅ **基于相机空间**：坐标转换到相机空间进行计算
- ❌ **写死的类型判断**：如 `"countertop" in node.name` 或 `"stove burner" in node.name`
- ❌ **硬编码阈值**：多个阈值参数（IN_CONTACT_DISTANCE, CLOSE_DISTANCE, INSIDE_THRESH 等）

---

## 四、关键差异总结

| 特性 | demo3.ipynb (CRAFT) | reflect/main/scene_graph.py |
|------|---------------------|------------------------------|
| **状态提取** | ✅ 动态（基于对象属性） | ❌ 写死的 state_dict |
| **关系判断** | ✅ 动态（基于元数据属性） | ❌ 部分写死（如 countertop 判断） |
| **关系类型** | 3种：holding, inside, on_top_of | 7种：inside, on_top_of, above, below, left/right, blocking |
| **数据源** | AI2THOR metadata | 点云 + metadata |
| **位置计算** | 3D 位置（position） | 点云距离 + 相机空间坐标 |
| **Action-aware** | ✅ 支持 | ❌ 不支持 |
| **代码复杂度** | 低（~170行） | 高（~337行） |
| **可扩展性** | ✅ 高（动态判断） | ❌ 低（需要维护类型列表） |

---

## 五、优势分析

### demo3.ipynb (CRAFT) 的优势
1. **更好的可扩展性**：不依赖写死的对象类型列表，可以自动适应新对象
2. **更简洁的代码**：逻辑清晰，易于维护
3. **Action-aware 支持**：可以记录场景图对应的时间步和动作
4. **统一的架构**：使用统一的 SceneGraph 类，便于集成

### reflect/main/scene_graph.py 的优势
1. **更丰富的关系类型**：支持更多空间关系（above, below, blocking 等）
2. **基于点云**：使用点云数据可以更准确地计算空间关系
3. **更详细的状态信息**：支持组合状态（如 "filled with coffee and dirty"）
4. **更精确的空间计算**：使用相机空间坐标和点云距离

---

## 六、建议

1. **结合两者优势**：
   - 采用 CRAFT 的动态判断逻辑（状态和关系）
   - 保留 REFLECT 的丰富关系类型（above, below, blocking）
   - 如果有点云数据，可以使用点云进行更精确的计算

2. **统一接口**：
   - 建议统一使用 CRAFT 的 SceneGraph 类
   - 在生成逻辑中支持可选的点云数据

3. **保持 Action-aware**：
   - 保留 CRAFT 的 timestep 和 action 参数，这对任务执行很重要

