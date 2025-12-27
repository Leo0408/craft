# Scene Graph 生成逻辑说明（动态版本）

## 一、整体流程

```
AI2THOR Event (结构化环境元数据)
    ↓
generate_scene_graph_from_event()
    ├─ 动态提取对象状态（从 metadata 属性自动判断）
    ├─ 动态判断关系类型（从 receptacleObjectIds 等属性判断）
    └─ 添加关系边（holding, inside, on_top_of）
    ↓
完整场景图 (所有对象 + 所有关系)
    ↓
extract_task_relevant_subgraph_with_closure()
    ├─ 动态提取任务相关对象（从 actions, success_condition 解析）
    ├─ BFS 闭包扩展（沿着因果关系边）
    └─ 动态判断功能性支撑（基于任务相关性和对象属性）
    ↓
裁剪后子图 (任务相关对象 + 因果闭包)
```

## 二、详细生成逻辑

### 2.1 完整场景图生成 (`generate_scene_graph_from_event`)

**核心原则**：**环境驱动，不依赖写死的对象类型列表**

**输入**：AI2THOR event, task_info, timestep, action

**步骤**：

1. **创建 Action-aware Scene Graph**
   ```python
   sg = SceneGraph(task=task_info, timestep=timestep, action=action)
   ```

2. **提取所有对象并创建节点**
   - 从 `event.metadata.objects` 获取所有对象
   - 为每个对象创建 `Node`，包含：
     - `name`: 对象名称（如 `CoffeeMachine_43b68f52`）
     - `object_type`: 对象类型（如 `CoffeeMachine`）
     - `state`: **动态提取的状态**（见下文）
     - `position`: 3D 位置
     - `attributes`: 属性字典（isFilled, isOpen, isToggled, isPickedUp）

3. **状态提取规则（动态，不依赖写死的类型列表）**
   ```python
   # 动态检查对象是否有可开关属性
   if obj.get('isOpen') is not None:
       state = 'open' if obj.get('isOpen', False) else 'closed'
   
   # 动态检查对象是否有可切换属性
   elif obj.get('isToggledOn') is not None:
       state = 'on' if obj.get('isToggledOn', False) else 'off'
   
   # 动态检查对象是否有可填充属性
   elif obj.get('isFilled') is not None:
       state = 'filled' if obj.get('isFilled', False) else 'empty'
   
   # 其他对象：state = None（不设置状态）
   ```
   
   **优势**：
   - ✅ 不依赖写死的对象类型列表
   - ✅ 自动适应新的对象类型
   - ✅ 基于环境提供的属性动态判断

4. **添加关系边（动态判断关系类型）**

   **a) holding 关系**：
   - 如果对象被拿起 (`isPickedUp=True`)，创建 `Robot --holding--> Object`
   
   **b) inside/on_top_of 关系**（从 `parentReceptacles`，**动态判断**）：
   ```python
   if obj.parentReceptacles:
       for parent_id in parentReceptacles:
           # 动态判断容器类型：通过 receptacleObjectIds 判断
           has_receptacle = bool(other_obj.get('receptacleObjectIds', []))
           is_openable_container = 'isOpen' in other_obj
           
           # 动态判断关系类型：
           # - 可打开的容器（Fridge, Cabinet等）→ inside
           # - 有 receptacleObjectIds 的容器 → inside
           # - 否则 → on_top_of（表面类型）
           if is_openable_container or (has_receptacle and receptacle_count > 0):
               relation = "inside"
           else:
               relation = "on_top_of"  # 表面类型（CounterTop, Table等）
   ```
   
   **优势**：
   - ✅ 不依赖写死的容器/表面类型列表
   - ✅ 通过 `receptacleObjectIds` 动态判断容器
   - ✅ 通过 `isOpen` 属性判断可打开容器
   
   **c) on_top_of 关系**（基于位置信息，**动态判断表面类型**）：
   ```python
   def is_surface_type_dynamic(obj_type: str) -> bool:
       """动态判断是否是表面类型（基于对象类型特征）"""
       type_lower = obj_type.lower()
       # 表面类型的特征：通常包含 top, table, burner, sink 等关键词
       surface_keywords = ['countertop', 'table', 'stoveburner', 'burner', 'sink']
       return any(kw in type_lower for kw in surface_keywords)
   ```
   - 条件：
     - 垂直高度差：0.05m < z_diff < 0.5m
     - 水平距离：< 0.2m
     - 下方对象是表面类型（**动态判断**）
   - **防止自环**：检查 `obj.objectId != other_obj.objectId`

### 2.2 任务相关对象提取 (`extract_task_relevant_objects`)

**核心原则**：**从任务描述和动作序列动态提取，不依赖固定对象集合**

**提取来源**：

1. **从 actions 中提取**：
   ```python
   # 解析动作字符串，例如: "(pick_up, Mug)" 或 "(put_in, Mug, CoffeeMachine)"
   matches = re.findall(r'\(([^)]+)\)', action_str)
   # 提取动作参数中的对象名（跳过第一个动作类型）
   ```

2. **从 success_condition 中提取**：
   ```python
   # 提取大写开头的单词（通常是对象名）
   obj_matches = re.findall(r'\b([A-Z][a-zA-Z]+)\b', success_condition)
   ```
   ⚠️ **注意**：这个规则可能提取过多对象，可以优化为关键词匹配

3. **从 preactions 中提取**：
   - 与 actions 相同的解析逻辑

**输出**：相关对象名称的集合（如 `{'Mug', 'CoffeeMachine', 'CounterTop'}`）

**优势**：
- ✅ 不依赖任务模板或固定对象集合
- ✅ 自动适应不同任务
- ✅ 跨任务和跨场景的泛化能力

### 2.3 闭包裁剪 (`extract_task_relevant_subgraph_with_closure`)

**算法**：BFS 闭包扩展（**动态判断，不依赖写死的列表**）

**步骤**：

1. **查找初始相关节点**
   ```python
   # 精确匹配或部分匹配（对象名/类型）
   for node in full_scene_graph.nodes:
       if node.name in relevant_object_names:
           initial_nodes.append(node)
       # 或通过对象类型匹配
   ```

2. **BFS 闭包扩展**
   ```python
   closure = set(initial_nodes)
   queue = deque(initial_nodes)
   
   ALLOWED_CAUSAL_RELATIONS = {"inside", "holding", "on_top_of"}
   
   while queue:
       obj = queue.popleft()
       for edge in get_edges_of(obj):
           if edge.edge_type in ALLOWED_CAUSAL_RELATIONS:
               # 动态判断是否应该添加到闭包
   ```

3. **允许的关系类型（语义定义，不依赖对象类型）**
   ```python
   ALLOWED_CAUSAL_RELATIONS = {"inside", "holding", "on_top_of"}
   ```
   这些关系类型是语义定义的，不依赖具体对象类型。

4. **防止 CounterTop 泛化（动态判断）**
   ```python
   def is_surface_type(node: Node) -> bool:
       """动态判断节点是否是表面类型（基于对象类型特征）"""
       type_lower = node.object_type.lower()
       surface_keywords = ['countertop', 'table', 'stoveburner', 'burner']
       return any(kw in type_lower for kw in surface_keywords)
   
   def is_container_type(node: Node) -> bool:
       """动态判断节点是否是容器类型（基于场景图中的关系）"""
       # 方法1：检查节点是否在 inside 关系的目标端
       for (start_name, end_name), edge in self.edges.items():
           if edge.end.name == node.name and edge.edge_type == "inside":
               return True
       # 方法2：通过对象类型特征（启发式）
       type_lower = node.object_type.lower()
       container_keywords = ['basin', 'machine', 'fridge', 'cabinet', 'drawer', 'microwave']
       return any(kw in type_lower for kw in container_keywords)
   
   # 对于 on_top_of 关系，动态判断
   if edge.edge_type == "on_top_of":
       if is_surface_type(container_node):
           # 如果容器是表面类型，且对象不是任务相关对象，跳过
           if not is_task_relevant_object(obj_name_base, obj_node.object_type):
               continue
   ```

## 三、动态判断规则（不依赖写死的列表）

### 3.1 状态提取（完全动态）

**不再使用写死的对象类型列表**，而是：
- 检查对象是否有 `isOpen` 属性 → 可开关对象
- 检查对象是否有 `isToggledOn` 属性 → 可切换对象
- 检查对象是否有 `isFilled` 属性 → 可填充对象

### 3.2 关系类型判断（动态）

**不再使用写死的容器/表面类型列表**，而是：
- 通过 `receptacleObjectIds` 判断是否是容器
- 通过 `isOpen` 属性判断可打开容器
- 通过对象类型关键词（countertop, table, burner）判断表面类型

### 3.3 闭包扩展规则（部分动态）

**允许的关系类型**（语义定义，不依赖对象类型）：
```python
ALLOWED_CAUSAL_RELATIONS = {"inside", "holding", "on_top_of"}
```

**容器/表面类型判断**（动态）：
- 通过场景图中的关系判断（如果节点在 inside 关系的目标端，则是容器）
- 通过对象类型关键词判断（启发式，但比写死列表更灵活）

**功能性支撑判断**（动态）：
- 基于任务相关对象动态判断
- 基于对象属性（isPickedUp）判断可操作性

## 四、优势总结

### 4.1 环境驱动
- ✅ 状态和关系直接来源于环境的结构化信息
- ✅ 不依赖视觉模型或语言模型推断
- ✅ 保证物理一致性和可验证性

### 4.2 动态适应
- ✅ 不依赖写死的对象类型列表
- ✅ 自动适应新的对象类型
- ✅ 跨任务和跨场景的泛化能力

### 4.3 任务相关
- ✅ 从任务描述和动作序列动态提取相关对象
- ✅ 不依赖任务模板或固定对象集合
- ✅ 支持不同任务和不同场景

### 4.4 动作对齐
- ✅ 与动作序列显式对齐（Action-aware）
- ✅ 支持动作级约束绑定与失败定位
- ✅ 明确每个场景图的验证目的和时间步

## 五、实现位置

- **状态提取**：`demo3.ipynb` - `generate_scene_graph_from_event()` 函数
- **关系类型判断**：`demo3.ipynb` - `generate_scene_graph_from_event()` 函数
- **闭包扩展**：`core/scene_graph.py` - `extract_task_relevant_subgraph_with_closure()` 方法
- **任务对象提取**：`demo3.ipynb` - `extract_task_relevant_objects()` 函数

## 六、与 REFLECT 的区别

| 特性 | REFLECT | CRAFT (本文) |
|------|---------|--------------|
| **场景抽象方式** | 基于关键帧和视觉描述 | 基于结构化环境元数据 |
| **状态推断** | 视觉模型推断 | 环境直接提供 |
| **关系推断** | 视觉模型推断 | 环境直接提供 |
| **对象类型** | 依赖固定对象集合 | 动态从环境提取 |
| **任务相关性** | 依赖任务模板 | 动态从任务描述提取 |
| **可执行性** | 不确定 | 确定（可验证） |
| **物理一致性** | 可能不一致 | 保证一致 |
