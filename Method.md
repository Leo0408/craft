下面为你生成 全新、升级版、结构清晰、可直接写入论文 / GitHub / 项目文档的 CRAFT++（增强版）框架 Markdown。
这一版本完全结合你前面所有优化要求，并严格对齐：
	•	可执行逻辑约束（constraint AST）
	•	时序与因果链（pre/post/invariants）
	•	环境记忆（Environment Memory）
	•	结构化 LLM 输出
	•	动态验证关键帧（而非每帧检查）
	•	可扩展到真实/模拟环境

这就是你论文最终会使用的版本。

⸻

🧠 CRAFT++：基于可执行逻辑约束与环境记忆的机器人失败检测框架（Markdown 版本）

⸻

#️⃣ 0. 框架概述（Overview）

CRAFT++ 旨在解决零样本 LLM 系统在真实场景失败检测中的三个核心问题：
	•	感知噪声导致的错误判断（遮挡、不稳定检测 → 假失败/假成功）
	•	缺乏物理可验证性（LLM“看图编故事” → 幻觉式成功判断）
	•	缺乏因果链/动作前后逻辑（例如：未加水却被判定能加热水壶）

CRAFT++ 的核心思想是：
让 LLM 生成可执行约束（Executable Constraints），并通过逻辑引擎与时序记忆进行验证，从而实现与视觉无关、与场景无关的确定性失败检测。

**关键改进（基于 improve1.md 和 improve2.md）**：
	•	动作级约束（Action-Level Constraints）：约束必须明确绑定到具体动作，而非任务级
	•	正确区分容器状态和对象状态：容器空/满 vs 对象填充
	•	动作执行前后分别检查：Precondition 在动作前，Postcondition 在动作后
	•	失败类型分类：Precondition Violation, Postcondition Violation, Goal Not Achieved
	•	场景图裁剪：只保留任务相关的最小子图
	•	**Precondition 失败后立即停止**：如果 Precondition 失败，不再检查后续动作和 Goal（CRAFT 核心逻辑）
	•	**Goal 只用于成功判定**：Goal 不参与失败溯因，只在没有 Precondition 失败时检查

框架包含三层：

(Perception + Memory) → Scene Graph (Task-Relevant Subgraph) → Action-Level Constraint Compiler → Constraint Executor (Action-by-Action)


⸻

#️⃣ 1. 场景图构建（Scene Graph Construction）

场景图用于描述：
	•	对象（节点）
	•	关系（边）
	•	几何/状态属性（state, bbox, pose, confidence）
	•	时间特征（last_seen_ts, velocity）

输入：
	•	检测结果 detections
	•	空间关系 spatial_relations
	•	任务信息 task_info

输出：
	•	SceneGraph（结构化场景表示）

✔ 伪代码

Algorithm BuildSceneGraph(detections, spatial_relations, task_info):

    scene_graph = SceneGraph()

    # 1. 创建对象节点
    for det in detections:
        node = SceneNode(
            name = det.name,
            type = det.obj_type,
            state = det.state,
            bbox = det.bbox,
            pose = det.pose,
            confidence = det.confidence,
            last_seen_ts = current_time()
        )
        scene_graph.add_node(node)

    # 2. 创建空间关系
    for rel in spatial_relations:
        scene_graph.add_edge(
            Edge(rel.obj1, rel.obj2, rel.type, rel.confidence)
        )

    # 3. 附加任务信息
    scene_graph.task_info = task_info

    return scene_graph


⸻

#️⃣ 1.5 真实环境场景图构建（Real-World Scene Graph Construction）

**核心思想**：CRAFT is **perception-agnostic** and can be directly integrated with open-vocabulary perception models (CLIP/Detic/DINO/SAM) in real-world environments.

## 1.5.1 多模态感知层（Multi-Modal Perception Layer）

真实环境中，CRAFT 使用开放词表检测器作为前端感知层：

**推荐组合**（真实环境常见做法）：

### 方案 A：DETIC + CLIP + ByteTrack（推荐）⭐

| 模块 | 推荐模型 | 作用 |
|------|---------|------|
| 物体检测 | **DETIC** | 开放词表物体识别（比 MDETR 更鲁棒） |
| 语义过滤 | **CLIP** | 语义匹配、prompt 扩展、误检过滤 |
| 跟踪 | **ByteTrack** | 多目标跟踪，处理遮挡和ID切换 |
| 记忆 | **Environment Memory** | 时序平滑、遮挡预测、置信度衰减 |

**核心优势**：
- ✅ **DETIC**：更强的开放词表检测能力，支持 21k 类别
- ✅ **CLIP 语义过滤**：通过语义相似度过滤误检，支持 prompt 扩展（如 "cup" → "a cup", "the cup", "coffee cup"）
- ✅ **ByteTrack 跟踪**：处理遮挡、ID 切换，提供稳定的对象轨迹
- ✅ **Memory 集成**：与 Environment Memory 无缝集成，提供时序一致性

**工作流程**：
```
RGB-D Stream
    ↓
DETIC Detection (open-vocab, 21k classes)
    ↓
CLIP Semantic Filtering (filter by object_list, expand prompts)
    ↓
ByteTrack Tracking (multi-object tracking)
    ↓
Environment Memory (temporal smoothing, occlusion handling)
    ↓
Scene Graph Construction (with confidence scores)
```

### 方案 B：MDETR（原始方案，用于对比）

| 模块 | 推荐模型 | 作用 |
|------|---------|------|
| 物体检测 | MDETR | 开放词表物体识别 |
| 语义对齐 | CLIP（可选） | 属性识别（mug / cup / coffee machine） |
| 分割 | SAM / SAM2 | 像素级掩码生成 |
| 跟踪 | SORT / ByteTrack | 多目标跟踪 |
| 深度 | RGB-D 或单目估计 | 3D 位置估计 |

**注意**：MDETR 在某些场景下可能检测不到对象，建议使用 DETIC + CLIP 方案。

**输入**：RGB-D Stream 或 RGB Stream

**输出**：带置信度的检测结果

```python
Detection = {
    "label": "Mug",
    "bbox": [x1, y1, x2, y2],
    "confidence": 0.83,
    "mask": ...,
    "depth": ...,
    "position_3d": (x, y, z)
}
```

## 1.5.2 真实环境场景图节点（Real-World Scene Graph Node）

真实环境中的节点包含感知相关和时间相关信息：

```python
class Node:
    def __init__(self, name, obj_type):
        self.name = name
        self.object_type = obj_type
        
        # 感知相关（带置信度）
        self.bbox = None
        self.mask = None
        self.confidence = 0.0  # 检测置信度
        self.position_3d = None  # 3D 位置
        
        # 状态属性（CRAFT关心的）
        self.attributes = {
            "isPickedUp": False,
            "isOpen": None,
            "isFilled": None,
            "isToggled": None
        }
        
        # 时间信息（关键）
        self.last_seen_ts = None
        self.velocity = None  # 用于预测遮挡位置
```

## 1.5.3 关系边的构建（带置信度）

真实环境里，关系一定是"软"的（带置信度）：

### 增强的空间关系检测算法

**改进版本**：使用 3D 边界框和几何分析，支持多种关系类型：

```python
def compute_spatial_relations(detections):
    """
    增强的空间关系计算（优先级顺序）
    
    输入：检测结果（包含 position_3d 和可选的 bbox3d）
    输出：关系列表 (obj1, obj2, relation_type, confidence)
    """
    relations = []
    
    for det1, det2 in pairs(detections):
        # 优先级 1: inside 关系（使用 3D 边界框）
        if has_bbox3d(det1) and has_bbox3d(det2):
            if bbox_inside(det1.bbox3d, det2.bbox3d, overlap_ratio=0.7):
                relations.append((det1, det2, "inside", 0.9))
                continue  # 跳过其他关系
        
        # 优先级 2: on_top_of 关系（改进版）
        distance = compute_distance(det1.position_3d, det2.position_3d)
        if distance < close_threshold:
            z_diff = det1.z - det2.z
            horizontal_dist = compute_horizontal_distance(det1, det2)
            
            # 要求：垂直高度差 > 50mm 且水平距离 < 200mm
            if z_diff > 50mm and horizontal_dist < 200mm:
                relations.append((det1, det2, "on_top_of", 0.85))
                continue
        
        # 优先级 3: in_contact 关系
        if distance < 100mm:
            relations.append((det1, det2, "in_contact", 1.0))
            continue
        
        # 优先级 4: near 关系（默认）
        if distance < 400mm:
            relations.append((det1, det2, "near", 0.7))
    
    return relations
```

### 关系类型和检测条件

| 关系类型 | 检测条件 | 置信度 | 说明 |
|---------|---------|--------|------|
| **inside** | 3D 边界框包含（重叠率 ≥ 70%） | 0.9 | 物体 A 在物体 B 内部（如杯子在咖啡机内） |
| **on_top_of** | 垂直高度差 > 50mm 且水平距离 < 200mm | 0.85 | 物体 A 在物体 B 上方（如杯子在桌子上） |
| **in_contact** | 3D 距离 < 100mm | 1.0 | 物体直接接触 |
| **near** | 3D 距离 < 400mm | 0.7 | 物体接近但没有特定空间关系 |

### 检测算法伪代码

```
Algorithm ComputeSpatialRelations(detections):
    relations = []
    
    FOR each pair (det1, det2) in detections:
        
        # Priority 1: Check "inside" using 3D bounding boxes
        IF det1.bbox3d AND det2.bbox3d:
            IF bbox_inside(det1.bbox3d, det2.bbox3d, overlap_ratio=0.7):
                relations.append((det1, det2, "inside", 0.9))
                CONTINUE  # Skip other relations
        
        # Priority 2: Check "on_top_of" with enhanced criteria
        distance = ||det1.position - det2.position||
        IF distance < 400mm:
            z_diff = det1.z - det2.z
            horizontal_dist = ||(det1.x, det1.y) - (det2.x, det2.y)||
            
            IF z_diff > 50mm AND horizontal_dist < 200mm:
                relations.append((det1, det2, "on_top_of", 0.85))
                CONTINUE
        
        # Priority 3: Check "in_contact"
        IF distance < 100mm:
            relations.append((det1, det2, "in_contact", 1.0))
            CONTINUE
        
        # Priority 4: Default to "near"
        IF distance < 400mm:
            relations.append((det1, det2, "near", 0.7))
    
    RETURN relations
```

### 关键改进点

1. **3D 边界框支持**：
   - 使用 3D 边界框检测 "inside" 关系
   - 支持 Open3D `AxisAlignedBoundingBox` 和字典格式
   - 计算边界框重叠率（默认 70%）

2. **改进的 on_top_of 检测**：
   - 不仅检查垂直高度差，还检查水平距离
   - 降低阈值：从 700mm 降到 50mm（更符合真实环境）
   - 避免误判：要求水平距离也要小

3. **优先级机制**：
   - 按优先级检测：inside → on_top_of → in_contact → near
   - 一旦检测到高优先级关系，跳过低优先级检测

4. **单位自动检测**：
   - 自动检测坐标单位（米或毫米）
   - 根据单位调整阈值

**关键点**：
- 真实环境的关系推断基于几何（bbox, 3D position），不是 ground truth
- 每个关系都有置信度分数
- 使用 3D 边界框可以更准确地检测 "inside" 和 "on_top_of" 关系
- 这正是为什么需要 Environment Memory 来稳定关系

## 1.5.4 真实环境完整流程

### DETIC + CLIP 方案（推荐）

```
RGB-D Stream
    ↓
DETIC Detection (open-vocab, 21k classes)
    ↓
CLIP Semantic Filtering
    - Prompt expansion ("cup" → ["a cup", "the cup", "coffee cup"])
    - Semantic similarity filtering (threshold: 0.25)
    ↓
ByteTrack Multi-object Tracking
    - Track IDs across frames
    - Handle occlusion and ID switches
    ↓
Scene Graph Construction (with confidence)
    - Nodes: objects with DETIC confidence + CLIP score
    - Edges: spatial relations with confidence
    ↓
Environment Memory (temporal smoothing, occlusion handling)
    - Kalman-like position smoothing
    - Occlusion prediction
    - Confidence decay for unseen objects
    ↓
Smoothed Scene Graph (for constraint validation)
```

### MDETR 方案（原始，用于对比）

```
RGB-D Stream
    ↓
MDETR Detection (open-vocab)
    ↓
Multi-object Tracking (optional)
    ↓
Scene Graph Construction (with confidence)
    ↓
Environment Memory (temporal smoothing, occlusion handling)
    ↓
Smoothed Scene Graph (for constraint validation)
```

**与仿真环境的区别**：
- 仿真：Scene Graph 直接来自 ground truth
- 真实：Scene Graph 经过感知 → 记忆平滑 → 置信度处理


⸻

#️⃣ 1.1 场景图裁剪（Task-Relevant Subgraph Extraction）

**核心思想**：从完整场景图中裁剪出"与当前子任务相关的最小子图"，减少复杂度，提高约束生成和验证的效率。

**问题**：
	•	完整场景图可能包含大量无关对象（例如：90个节点，36条边）
	•	约束生成和验证时只需要关注与任务相关的对象
	•	减少场景图大小可以：
		- 降低 LLM 输入长度
		- 提高约束生成质量
		- 加快约束验证速度

**解决方案（优化版）**：使用**闭包（Closure）方法**进行子图裁剪

## 1.1.1 闭包裁剪算法（Closure-based Subgraph Extraction）

**核心改进**：使用 BFS 从任务相关对象开始，沿着 `inside`/`on_top_of`/`supported_by` 边扩展，确保包含所有相关的容器和支撑结构。

**为什么需要闭包方法**：
	•	简单裁剪可能遗漏重要的容器对象（如 Pot 在 Sink 中，但 Sink 不在任务相关对象列表中）
	•	需要包含支撑结构（如 Mug 在 CounterTop 上，CounterTop 需要被包含）
	•	确保约束验证时能正确检查容器状态和支撑关系

✔ 伪代码

Algorithm ExtractTaskRelevantSubgraphWithClosure(full_scene_graph, task_info):

    # 1. 提取任务相关对象名称
    relevant_objects = Set()
    
    # 从 actions 中提取
    for action in task_info.actions:
        # 解析动作字符串，例如: "(pick_up, Mug)" 或 "(put_in, Mug, CoffeeMachine)"
        objects = ParseActionParameters(action)
        relevant_objects.add_all(objects)
    
    # 从 success_condition 中提取
    objects = ExtractObjectNames(task_info.success_condition)
    relevant_objects.add_all(objects)
    
    # 从 preactions 中提取（如果有）
    for preaction in task_info.preactions:
        objects = ParseActionParameters(preaction)
        relevant_objects.add_all(objects)
    
    # 2. 查找初始相关节点
    initial_nodes = []
    for node in full_scene_graph.nodes:
        if IsRelevant(node, relevant_objects):
            initial_nodes.append(node)
    
    # 3. 使用 BFS 闭包扩展：沿着 inside/on_top_of/supported_by 边扩展
    closure = Set(initial_nodes)
    queue = Queue(initial_nodes)
    expansion_edge_types = ["inside", "on_top_of", "supported_by"]
    
    while queue is not empty:
        obj = queue.pop()
        for edge in full_scene_graph.get_edges_of(obj):
            if edge.type in expansion_edge_types:
                # 确定目标节点（边的另一端）
                target_node = edge.dst if edge.src == obj else edge.src
                
                # 如果目标节点不在闭包中，添加到闭包和队列
                if target_node not in closure:
                    closure.add(target_node)
                    queue.append(target_node)
    
    # 4. 创建子图，包含闭包中的所有节点
    subgraph = SceneGraph()
    for node in closure:
            subgraph.add_node(node)
    
    # 5. 添加闭包内节点之间的所有边
    for edge in full_scene_graph.edges:
        if edge.start in closure AND edge.end in closure:
            subgraph.add_edge(edge)
    
    return subgraph

**示例**：
	•	任务：makeCoffee，相关对象：{Mug, CoffeeMachine}
	•	初始节点：Mug, CoffeeMachine
	•	闭包扩展：
		- Mug --[on_top_of]--> CounterTop → 添加 CounterTop
		- CoffeeMachine --[inside]--> CounterTop → CounterTop 已在闭包中
		- Mug --[inside]--> CoffeeMachine → 关系已存在
	•	最终子图：{Mug, CoffeeMachine, CounterTop} + 相关边

## 1.1.2 Action-aware Scene Graph

**核心思想**：Scene Graph 必须明确绑定到具体的动作和时间步，以便正确区分 Precondition、Postcondition 和 Invariant 的验证时机。

**问题**：
	•	传统 SG 只表示"当前帧的世界"，无法区分：
		- Precondition violation（动作前检查）
		- Postcondition violation（动作后检查）
		- Invariant violation（动作前后都要检查）
	•	无法明确 SG 是为了验证哪个 Action

**解决方案**：Action-conditioned Scene Graph

```python
class SceneGraph:
    def __init__(self, task=None, event=None, timestep=None, action=None):
        self.nodes = Set()
        self.edges = Dict()
        self.task = task
        self.event = event
        # Action-aware fields
        self.timestep: Optional[int] = timestep  # Which timestep this SG represents
        self.action: Optional[str] = action  # Which action this SG is for
```

**SG 用途与时间步对应关系**：

| SG 用途 | 使用哪一帧 | timestep | action |
|---------|-----------|----------|--------|
| **Precondition** | action 前一帧 | `action_idx - 1` | `action` |
| **Postcondition** | action 后一帧 | `action_idx + 1` | `action` |
| **Invariant** | action 前 & 后 | `action_idx - 1` 和 `action_idx + 1` | `action` |
| **Goal** | 最终帧 | `len(events) - 1` | `None` |

**使用示例**：

```python
# Precondition 验证：使用动作执行前的场景图
pre_sg = generate_scene_graph_from_event(
    events[action_idx - 1], 
    task_info, 
    timestep=action_idx - 1, 
    action=action
)
pre_sg = pre_sg.extract_task_relevant_subgraph_with_closure(task_info)

# Postcondition 验证：使用动作执行后的场景图
post_sg = generate_scene_graph_from_event(
    events[action_idx + 1], 
    task_info, 
    timestep=action_idx + 1, 
    action=action
)
post_sg = post_sg.extract_task_relevant_subgraph_with_closure(task_info)
```

**优势**：
	•	明确每个 SG 的验证目的（Precondition/Postcondition/Invariant）
	•	自动选择正确的时间步进行验证
	•	支持动作级失败溯因（知道是哪个动作的哪个约束失败）

**匹配策略**：
	•	精确匹配：对象名称完全匹配
	•	部分匹配：对象名称包含关系（例如 "Mug" 匹配 "Mug-1"）
	•	类型匹配：对象类型匹配（例如 "Mug" 匹配 objectType="Mug"）

**示例**：
	•	任务：makeCoffee
	•	Actions: ["(pick_up, Mug)", "(put_in, Mug, CoffeeMachine)", ...]
	•	Success condition: "a clean mug is filled with coffee"
	•	提取对象：{Mug, CoffeeMachine, Sink, Faucet, CounterTop}
	•	完整场景图：90个节点 → 裁剪后：~10个节点

**实现位置**：
	•	`core/scene_graph.py`：`SceneGraph.extract_task_relevant_subgraph_with_closure()` 方法
	•	使用方式：`task_relevant_sg = full_sg.extract_task_relevant_subgraph_with_closure(task_info)`

## 1.1.2 Action-aware Scene Graph

**核心思想**：Scene Graph 必须明确绑定到具体的动作和时间步，以便正确区分 Precondition、Postcondition 和 Invariant 的验证时机。

**问题**：
	•	传统 SG 只表示"当前帧的世界"，无法区分：
		- Precondition violation（动作前检查）
		- Postcondition violation（动作后检查）
		- Invariant violation（动作前后都要检查）
	•	无法明确 SG 是为了验证哪个 Action

**解决方案**：Action-conditioned Scene Graph

```python
class SceneGraph:
    def __init__(self, task=None, event=None, timestep=None, action=None):
        self.nodes = Set()
        self.edges = Dict()
        self.task = task
        self.event = event
        # Action-aware fields
        self.timestep: Optional[int] = timestep  # Which timestep this SG represents
        self.action: Optional[str] = action  # Which action this SG is for
```

**SG 用途与时间步对应关系**：

| SG 用途 | 使用哪一帧 | timestep | action |
|---------|-----------|----------|--------|
| **Precondition** | action 前一帧 | `action_idx - 1` | `action` |
| **Postcondition** | action 后一帧 | `action_idx + 1` | `action` |
| **Invariant** | action 前 & 后 | `action_idx - 1` 和 `action_idx + 1` | `action` |
| **Goal** | 最终帧 | `len(events) - 1` | `None` |

**使用示例**：

```python
# Precondition 验证：使用动作执行前的场景图
pre_sg = generate_scene_graph_from_event(
    events[action_idx - 1], 
    task_info, 
    timestep=action_idx - 1, 
    action=action
)
pre_sg = pre_sg.extract_task_relevant_subgraph_with_closure(task_info)

# Postcondition 验证：使用动作执行后的场景图
post_sg = generate_scene_graph_from_event(
    events[action_idx + 1], 
    task_info, 
    timestep=action_idx + 1, 
    action=action
)
post_sg = post_sg.extract_task_relevant_subgraph_with_closure(task_info)
```

**优势**：
	•	明确每个 SG 的验证目的（Precondition/Postcondition/Invariant）
	•	自动选择正确的时间步进行验证
	•	支持动作级失败溯因（知道是哪个动作的哪个约束失败）


⸻

#️⃣ 2. 动作感知约束生成（Action-aware Constraint Generation）

### 2.1 核心思想（Action-centric vs. Goal-centric）
传统的约束生成往往仅从任务的“最终目标”出发（Goal-centric），这导致验证过程集中在状态校验上，容易遗漏中间动作的因果要求。CRAFT++ 采用**动作感知约束生成（Action-aware Constraint Generation）**，将验证重心从“目标层”下移至“动作层”。

**核心断言**：因果失败发生在动作层，而不是目标层。只要约束生成不以动作序列为中心，就必然遗漏关键因果条件。

### 2.2 动作语义模板库（Action Semantic Templates）
系统预定义了一组物理常识模板，规定了每个动作的必要前置条件（Preconditions）和预期后置条件（Postconditions）。

| 动作 (Action) | 前置条件 (Preconditions) | 后置条件 (Postconditions) |
| :--- | :--- | :--- |
| `pick_up(X)` | `reachable(X)`, `gripper_empty` | `holding(X)` |
| `put_on(X, Y)` | `holding(X)` | `is_on(X, Y)` |
| `put_in(X, Y)` | `holding(X)`, `container_open(Y)`, `container_empty(Y)` | `inside(X, Y)` |
| `toggle_on(Y)` | `reachable(Y)` | `toggled(Y) == True` |
| `toggle_off(Y)`| `toggled(Y) == True` | `toggled(Y) == False` |

### 2.3 主算法：GenerateActionAwareConstraints（改进版）

#### 2.3.1 当前实现的问题（基于 improve4.md）

虽然约束语义模板设计正确，但当前实现存在系统性错误：

**❌ 问题 1：使用 final_scene_graph 生成所有动作约束**
- `final_sg` 是任务完成后的"未来世界"
- LLM 在生成第 1～N 步动作的约束时不可避免"偷看未来"
- 导致前后置条件严重错位，违反因果顺序

**❌ 问题 2：约束生成是"任务级"，但目标是"动作级"**
- 一次 Prompt 覆盖整个任务 + 全动作序列
- LLM 会自动补全跨动作的因果链，无法保证每条约束只对应一个原子动作

**❌ 问题 3：Action Binding 是"事后修补"，而非"生成即绑定"**
- LLM 生成后再用字符串/对象/语义匹配回绑动作
- Binding 逻辑复杂且不可靠

#### 2.3.2 改进后的算法（Action-centric Constraint Instantiation）

**核心改动思想**：从 **"Task-level constraint generation"** → 转为 **"Action-centric constraint instantiation"**

```python
Algorithm GenerateActionAwareConstraints_Improved:
Input: action_sequence
Output: action_bound_constraints

BEGIN
    constraints = []
    FOR i, action IN enumerate(action_sequence):
        # 1. 解析当前动作（不依赖 scene graph）
        action_type, action_args = parse_action(action)
        
        # 2. 查找动作模板
        template = ACTION_TEMPLATE_LIBRARY.get(action_type)
        IF template is None:
            CONTINUE
        
        # 3. 为当前动作生成前置条件 (Preconditions)
        FOR each pre_template IN template["pre"]:
            constraint = Instantiate(
                pre_template,
                action_args,
                type='precondition',
                action_index=i,  # 生成时就绑定
                action=action
            )
            constraints.append(constraint)
        
        # 4. 为当前动作生成后置条件 (Postconditions)
        FOR each post_template IN template["post"]:
            constraint = Instantiate(
                post_template,
                action_args,
                type='postcondition',
                action_index=i,  # 生成时就绑定
                action=action
            )
            constraints.append(constraint)
    
    RETURN constraints
END
```

**关键改进点**：

1. **按动作生成**：每个动作独立生成约束，避免因果混乱
2. **生成即绑定**：约束生成时就绑定 `action_index = i`，无需后续匹配
3. **不依赖 final_sg**：约束生成阶段不使用最终场景图，只用于验证阶段
4. **Action-local Prompt**：如果使用 LLM，Prompt 只包含当前动作信息，不包含全任务上下文

#### 2.3.3 LLM-based 改进版 Prompt 设计

如果使用 LLM 方法，应采用 Action-local Prompt：

**System Prompt**:
```
You are a robot task analyzer. Generate constraints for a SINGLE action by 
instantiating the provided action semantic template.
```

**User Prompt** (只包含当前动作):
```
Current Action: {action}
Action Index: {action_index}
Action Type: {action_type}
Action Arguments: {action_args}

Action Semantic Template:
    Preconditions: {pre_templates}
    Postconditions: {post_templates}

Rules:
- Generate constraints ONLY for this action
- Do NOT reference future actions
- Do NOT reference final goal state
- Instantiate templates with actual object names from action arguments

Output JSON format:
{
  "constraints": [
    {
      "id": "C1",
      "type": "precondition",
      "template": "holding(Mug)",
      "description": "Robot must be holding the Mug",
      "action_index": {action_index},
      "action": "{action}"
    }
  ]
}
```

**明确禁止**：
- ❌ 引入后续动作
- ❌ 引入最终 goal 状态
- ❌ 使用 final_scene_graph 作为输入

#### 2.3.4 Scene Graph 使用策略

**原则**：
- ❌ **不使用** `final_sg` 做动作约束生成
- ✅ **只用于**：
  - 约束校验（constraint checking）
  - 失败分析（why failed）
  - 物体属性补充（isOpen / isFilled）

**推荐方案**：
- **约束生成阶段**：不依赖 scene graph
- **约束验证阶段**：使用 scene graph

### 2.4 约束实例化与编译（模板 → 可执行代码）

CRAFT++ 支持两种约束生成方式：

#### 方式 1：LLM-based 约束生成（可选）
使用 LLM 生成约束描述，然后解析和编译为可执行代码。

#### 方式 2：模板化约束生成（推荐，improve3.md 方案）✅

**核心特点**：
- ✅ **不依赖 LLM**：直接从动作序列和预定义模板生成约束
- ✅ **不依赖 eval**：使用可执行函数，避免字符串拼接和 eval 的安全风险
- ✅ **确定性**：相同输入总是产生相同输出，完全可复现
- ✅ **直接编译**：约束生成时直接编译为可执行函数（`executable: Callable`）

**实现原理**：

生成的结构化约束被直接映射为可执行的判定函数（Executable Predicates），避免了自然语言解析的不确定性。

**Predicate 实现映射**（PREDICATE_IMPL）：

*   **`holding(X)`**：检查场景图中是否有 holding 边，或节点的 `isPickedUp` 属性为 True
*   **`container_empty(Y)`**：检查容器 Y 内是否有对象（通过检查 `inside` 类型的边）
*   **`container_open(Y)`**：检查容器的 `isOpen` 属性
*   **`gripper_empty`**：检查是否没有任何节点被标记为 `isPickedUp`
*   **`on_top_of(X, Y)`**：检查场景图中是否存在 `on_top_of` 类型的边
*   **`inside(X, Y)`**：检查场景图中是否存在 `inside` 类型的边
*   **`reachable(X)`**：检查对象 X 是否存在于场景图中
*   **`toggled_on(Y)`** / **`toggled_off(Y)`**：检查设备的 `isToggled` 属性
*   **`filled(Y)`**：检查对象的 `isFilled` 属性

**约束生成流程**：

```
Action Sequence → Parse Actions → Instantiate Templates → Compile to Executable Functions
```

每个约束包含：
- `predicate`: 谓词名称（如 "holding", "inside"）
- `args`: 参数列表（如 ["Mug"] 或 ["Mug", "CoffeeMachine"]）
- `executable`: 可执行函数 `Callable[[SceneGraph], bool]`
- `step`: 绑定的动作索引
- `type`: 约束类型（PRE / POST）

### 2.5 模板化方法 vs. LLM 方法对比

| 特性 | 模板化方法（推荐） | LLM 方法 |
| :--- | :--- | :--- |
| **确定性** | ✅ 完全确定，可复现 | ❌ 依赖 LLM 输出，可能不一致 |
| **性能** | ✅ 快速，无需 API 调用 | ⚠️ 需要 LLM API 调用 |
| **可扩展性** | ✅ 易于添加新动作模板 | ⚠️ 需要更新 Prompt |
| **安全性** | ✅ 不依赖 eval，类型安全 | ⚠️ 需要解析 LLM 输出 |
| **成本** | ✅ 无 API 成本 | ❌ 每次调用产生 API 成本 |
| **适用场景** | 常见动作，标准模板 | 复杂或领域特定的约束 |

**推荐使用模板化方法**，因为：
1. 对于常见的机器人动作（pick_up, put_in, put_on, toggle_on 等），模板已经涵盖了所有物理先验
2. 确定性保证使得调试和复现更容易
3. 无需外部 API 依赖，可以离线运行

### 2.6 优势总结

该设计将失败检测从“目标状态一致性检查”升级为“**动作因果一致性验证**”，使系统能够在物理上不可能的动作发生时即时定位失败原因，实现与物理仿真环境对齐的精确归因。

**模板化方法的额外优势**：
- **零配置运行**：不需要 LLM API 密钥即可使用
- **类型安全**：使用函数而非字符串，编译器可以检查类型错误
- **易于测试**：每个 predicate 函数都可以独立测试
- **可扩展**：新动作模板可以轻松添加到 `ACTION_TEMPLATES` 字典中

⸻

#️⃣ 3. 环境记忆模块（Environment Memory）

**核心问题**：在真实环境中，如果没有 Environment Memory，CRAFT 会被感知噪声"玩死"。

## 3.1 环境记忆解决的真实问题

### 问题 1：遮挡（Occlusion）
- 杯子被机械臂挡住
- Detic 检测不到
- Scene Graph 突然"消失"
- **👉 不是任务失败，是感知失败**

### 问题 2：跳变（Teleportation）
- mug 瞬间从桌子 → 水槽
- 深度误差 / mask 错误
- **👉 不是动作 teleport，是感知异常**

### 问题 3：置信度波动
- 同一物体在不同帧的检测置信度波动（0.7 → 0.9 → 0.6）
- 关系检测不稳定（inside 关系时有时无）
- **👉 需要时间平滑**

## 3.2 Environment Memory 的核心思想

**用时间连续性，约束"世界不可能乱跳"**

EnvironmentMemory 使用：
	•	Kalman / Bayesian filter（位置 smoothing）
	•	last_seen state 存储
	•	occlusion prediction（根据机械臂与摄像头视锥）
	•	状态置信度衰减模型
	•	关系稳定性跟踪

✔ Memory 输出世界状态（WorldState）

WorldState:
    objects: {object_name → ObjectState}
        - position: smoothed 3D position
        - velocity: estimated velocity
        - confidence: temporal-averaged confidence
        - occluded: boolean flag
        - occlusion_duration: seconds since last seen
        - position_variance: uncertainty measure
    relations: {(a,b) → RelationState}
        - confidence: stable confidence over time
        - stable_count: consecutive frames with relation
    timestamp: current time
    frame_count: number of frames processed


⸻

✔ Memory 更新伪代码（论文级）

Algorithm UpdateEnvironmentMemory(detections, relations, memory, t):

    for each object o in memory:
        if o detected at time t:
            # Object visible: update with Kalman-like smoothing
            o.position = KalmanUpdate(o.position, detection.position)
            o.confidence = temporal_average(o.confidence_history)
            o.last_seen_ts = t
            o.occluded = False
            o.occlusion_duration = 0.0
            
            # Check for teleportation (sudden large movement)
            if distance(o.position, o.previous_position) > threshold:
                o.position_variance *= 2  # Increase uncertainty
        else:
            # Object not detected: predict and mark occlusion
            o.occlusion_duration = t - o.last_seen_ts
            if o.occlusion_duration > occlusion_threshold:
                o.occluded = True
            
            # Predict position using velocity
            if o.velocity is not None:
                o.position = Predict(o.position, o.velocity, dt)
            
            # Decay confidence
            o.confidence *= (1 - decay_rate * o.occlusion_duration)
    
    # Update relations with temporal stability
    for each relation r in relations:
        if r detected at time t:
            r.stable_count += 1
            r.confidence = temporal_average(r.confidence_history)
        else:
            r.stable_count = 0  # Reset if relation disappears
    
    return smoothed_world_state


⸻

#️⃣ 4. 可执行约束验证层（Constraint Execution Layer）

每个约束包含：
	•	可执行条件 AST
	•	类型（pre/post/invariant/goal）
	•	可执行函数（inside / eq / intersects / reachable 等）

✔ 4.1 约束编译改进（正确区分容器和对象状态）

**关键改进**：区分 "容器是否为空" 和 "对象是否被填充"

Algorithm CompileConstraint(constraint_description):

    if IsContainerEmptyCheck(constraint_description):
        # 容器是否为空：检查容器内是否有对象
        # 使用场景图中的边关系
        condition_expr = "len([e for e in scene_graph.edges.values() 
                              if e.end.name == container_name 
                              and e.edge_type == 'inside']) == 0"
    
    elif IsObjectFilledCheck(constraint_description):
        # 对象是否被填充：检查 isFilled 属性
        condition_expr = "node.attributes.get('isFilled', False)"
    
    else:
        # 其他约束类型
        condition_expr = ParseStandardConstraint(constraint_description)
    
    return condition_expr

**示例**：
	•	"coffee machine must be empty" → 检查容器内对象数量
	•	"mug must be filled" → 检查 mug.isFilled 属性

✔ 4.2 ValidateConstraint（改进版：动作级检查）

Algorithm ValidateConstraint(constraint, scene_graph, action_index, events):

    # 根据约束类型和绑定的动作选择正确的场景图
    if constraint.type == 'pre':
        # Precondition: 在动作执行前检查
        if action_index > 0:
            eval_scene_graph = GenerateSceneGraph(events[action_index - 1])
            eval_scene_graph = eval_scene_graph.extract_task_relevant_subgraph(task_info)
        else:
            eval_scene_graph = initial_scene_graph
        evaluation_time = f"before action {action_index + 1}"
    
    elif constraint.type == 'post':
        # Postcondition: 在动作执行后检查
        if action_index < len(events) - 1:
            eval_scene_graph = GenerateSceneGraph(events[action_index + 1])
            eval_scene_graph = eval_scene_graph.extract_task_relevant_subgraph(task_info)
        else:
            eval_scene_graph = final_scene_graph
        evaluation_time = f"after action {action_index + 1}"
    
    else:  # goal
        eval_scene_graph = final_scene_graph
        evaluation_time = "at task completion"

    if constraint.condition_ast == NULL:
        return UNCERTAIN

    (value, atom_conf) = EvalPredicate(constraint.condition_ast, eval_scene_graph, memory)

    confidence = Aggregate(atom_conf)

    if value == True and confidence > threshold:
        return SATISFIED

    if value == False and confidence > threshold:
        return VIOLATED

    return UNCERTAIN


⸻

#️⃣ 5. 整体流程（Complete Failure Detection Pipeline）

**改进版：动作级约束检查**

Algorithm CRAFT_Pipeline(events, task_info):

    memory = EnvironmentMemory()
    
    # 1. 生成场景图（裁剪任务相关子图）
    initial_sg = BuildSceneGraph(events[0], task_info)
    initial_sg = initial_sg.extract_task_relevant_subgraph(task_info)
    
    # 2. 生成约束（动作级）
    constraints = GenerateConstraints(initial_sg, task_info)
    # 约束已绑定到具体动作
    
    # 3. 按动作顺序执行检查
    actions = task_info.actions
    failures = []
    
    for action_idx, action in enumerate(actions):
        
        # 3.1 检查该动作的 Preconditions（动作执行前）
        action_preconditions = GetConstraintsForAction(constraints, action_idx, type='pre')
        
        # 获取动作执行前的场景图
        if action_idx > 0:
            pre_scene_graph = BuildSceneGraph(events[action_idx - 1], task_info)
            pre_scene_graph = pre_scene_graph.extract_task_relevant_subgraph(task_info)
        else:
            pre_scene_graph = initial_sg
        
        for constraint in action_preconditions:
            status = ValidateConstraint(constraint, pre_scene_graph, action_idx, 'pre')
            
            if status == VIOLATED:
                failures.append({
                    "step": action_idx + 1,
                    "action": action,
                    "failure_type": "Precondition Violation",
                    "constraint": constraint,
                    "scene": pre_scene_graph
                })
                return failures  # CRAFT：立即失败
        
        # 3.2 执行动作（模拟或实际执行）
        # 这里假设动作已执行，events[action_idx] 是执行后的状态
        
        # 3.3 检查该动作的 Postconditions（动作执行后）
        action_postconditions = GetConstraintsForAction(constraints, action_idx, type='post')
        
        # 获取动作执行后的场景图
        if action_idx < len(events) - 1:
            post_scene_graph = BuildSceneGraph(events[action_idx + 1], task_info)
            post_scene_graph = post_scene_graph.extract_task_relevant_subgraph(task_info)
        else:
            post_scene_graph = BuildSceneGraph(events[-1], task_info)
            post_scene_graph = post_scene_graph.extract_task_relevant_subgraph(task_info)
        
        for constraint in action_postconditions:
            status = ValidateConstraint(constraint, post_scene_graph, action_idx, 'post')
            
            if status == VIOLATED:
                failures.append({
                    "step": action_idx + 1,
                    "action": action,
                    "failure_type": "Postcondition Violation",
                    "constraint": constraint,
                    "scene": post_scene_graph
                })
                return failures
    
    # 4. 检查最终 Goal（任务完成时）
    final_sg = BuildSceneGraph(events[-1], task_info)
    final_sg = final_sg.extract_task_relevant_subgraph(task_info)
    
    goal_constraints = GetConstraintsForAction(constraints, None, type='goal')
    for constraint in goal_constraints:
        status = ValidateConstraint(constraint, final_sg, len(actions), 'goal')
        
        if status == VIOLATED:
            failures.append({
                "step": "final",
                "action": "task_completion",
                "failure_type": "Goal Not Achieved",
                "constraint": constraint,
                "scene": final_sg
            })
    
    return failures if failures else SUCCESS

**关键改进点（基于 improve2.md）**：
	•	按动作顺序检查：从第一个动作开始，依次检查每个动作的 Pre/Post 约束
	•	Precondition 失败立即停止：一旦检测到 Precondition Violation，立即返回失败，不再检查后续动作
	•	Goal 检查条件：只有在没有 Precondition 失败的情况下，才检查 Goal
	•	失败报告优先级：优先报告 Precondition Violation（真正的失败原因），Goal Not Achieved 只作为补充信息
	•	约束必须绑定到动作：每个约束必须明确绑定到具体动作，不能是"悬空的约束"


⸻

#️⃣ 6. 核心约束类型（Constraint Types）

类型	示例	说明
Precondition	machine must be open	动作前必须满足（绑定到具体动作）
Postcondition	cup inside machine	动作后必须满足（绑定到具体动作）
Invariant	kettle cannot teleport	始终适用
Causal Chain	fill → has_water → heat	跨动作因果依赖
Geometry Constraint	not intersect(cup, machine.wall)	真实几何检查
Occupancy Constraint	volume_free(machine)	容器不能被占满
Memory Constraint	must not disappear instantly	遮挡时不应判断为消失

⸻

#️⃣ 6.1 失败类型分类（Failure Type Classification）

**关键改进**：区分不同类型的失败，用于精确归因

类型	含义	检测时机	示例
Precondition Violation	执行动作时违反前置条件	动作执行前	容器不为空时尝试放入
Postcondition Violation	动作执行后未达到预期状态	动作执行后	放入后对象不在容器内
Goal Not Achieved	任务未完成	任务结束时	最终状态不满足目标
Physical Impossibility	物理上不可能	动作执行时	对象位置冲突
Perception Inconsistency	感知噪声导致误判	持续监控	对象状态跳变异常

**失败检测输出格式**：

{
  "step": 3,
  "action": "put_in(mug, coffee_machine)",
  "failure_type": "Precondition Violation",
  "violated_constraint": "C3",
  "constraint_type": "precondition",
  "description": "Coffee machine must be empty before inserting mug",
  "reason": "Container contains 1 object(s)",
  "scene_snapshot": {...}
}


⸻

#️⃣ 7. CRAFT++ 的优势（基于逻辑 + 几何 + 记忆）

问题	REFLECT	CRAFT++
遮挡导致假失败	✔ 容易误判	✘ Memory 自动识别 occlusion
靠近物体误判成功	✔ 可能错误	✘ 真实几何 & volume check
未加水却可加热	✔ 无因果链	✘ Pre/Post + Causal Chain
视觉噪声导致状态跳变	✔ 易 hallucinate	✘ Memory smoothing
难以复现、确定性差	✔ LLM 输出不稳定	✘ 可执行逻辑完全可复现


⸻

#️⃣ 8. 典型示例（执行失败检测）

**改进版：动作级约束检测**

⸻

例 1：咖啡机不为空时尝试放入（REFLECT 示例）

**REFLECT 描述**：
"The robot attempted to place the mug inside the coffee machine while there was already a cup inside it."

**CRAFT++ 检测流程**：

1. 动作：put_in(mug, coffee_machine) - Step 9

2. 检查 Precondition（动作执行前）：
   - Constraint C3: coffee_machine.contains == ∅
   - 场景图检查：len([e for e in scene_graph.edges.values() 
                      if e.end.name == 'CoffeeMachine' 
                      and e.edge_type == 'inside']) == 0
   - 结果：False（容器内有 1 个对象）

3. 输出：

```
Failure Detected at Step 9:
Action: put_in(mug, coffee_machine)

Violated Constraint:
- Type: Precondition
- Description: Coffee machine must be empty before inserting mug
- Condition: container_empty(coffee_machine)

Failure Type:
- Precondition Violation

Explanation:
- The robot attempted to insert the mug into a non-empty container.
- Container contains 1 object(s) (cup)
```

**关键改进**：
	•	失败在动作执行前就被检测到（不需要等到任务结束）
	•	失败位置唯一且确定（Step 9）
	•	失败类型明确（Precondition Violation）
	•	不依赖 LLM 主观判断（可执行逻辑验证）

⸻

例 2：水壶没加水却加热

**动作链**：
- A1: fill(pot) - Step 4
- A2: heat(pot) - Step 8

**CRAFT++ 检测流程**：

1. 检查 fill 动作的 Postcondition（动作执行后）：
   - Constraint C4: pot.isFilled == True
   - 场景图检查：pot.attributes.get('isFilled', False)
   - 结果：False（pot 未被填充）

2. 输出：

```
Failure Detected at Step 4:
Action: fill(pot)

Violated Constraint:
- Type: Postcondition
- Description: Pot must be filled with water after filling
- Condition: pot.isFilled == True

Failure Type:
- Postcondition Violation

Causal Chain:
- fill(pot) failed → pot.isFilled == False
- Cannot proceed to heat(pot) (precondition: pot.isFilled == True)
```

**关键改进**：
	•	检测到 fill 动作失败（Postcondition Violation）
	•	自动阻止后续 heat 动作（因果链检查）
	•	失败原因可追溯（fill 动作未成功）


⸻

#️⃣ 9. 完整系统结构图（概念）

## 9.1 仿真环境流程

+------------------+
|   AI2THOR States |
+------------------+
           |
           v
+---------------------------+
|      Scene Graph          |
+---------------------------+
           |
           v
+---------------------------+
|   LLM Constraint Compiler |
+---------------------------+
           |
           v
+---------------------------+
|  Constraint Executor      |
+---------------------------+
           |
           v
+---------------------------+
|   Failure Detection       |
+---------------------------+

## 9.2 真实环境流程（完整版）

+------------------+
|   RGB-D Stream   |
+------------------+
           |
           v
+---------------------------+
| Multi-Modal Perception   |
| (MDETR/CLIP/Detic/SAM)   |
+---------------------------+
           |
           v
+---------------------------+
| Multi-object Tracking     |
+---------------------------+
           |
           v
+---------------------------+
| Scene Graph Construction  |
| (with confidence scores)  |
+---------------------------+
           |
           v
+---------------------------+
|    Environment Memory     |
| (temporal smoothing,      |
|  occlusion handling)      |
+---------------------------+
           |
           v
+---------------------------+
|  Smoothed Scene Graph     |
+---------------------------+
           |
           v
+---------------------------+
|   LLM Constraint Compiler |
| (Action-aware)            |
+---------------------------+
           |
           v
+---------------------------+
|  Constraint Executor      |
| (with confidence          |
|  thresholds)              |
+---------------------------+
           |
           v
+---------------------------+
|   Failure Detection       |
| (distinguish perception   |
|  errors vs. failures)     |
+---------------------------+


⸻

#️⃣ 10. 总结（最凝练的论文式描述）

我们提出 CRAFT++，一个结合任务逻辑、可执行条件与环境记忆的失败检测框架。与依赖 LLM 概率性推理的现有方法相比，CRAFT++ 将任务知识转换为可执行逻辑表达式，通过时序建模与几何检查实现确定性、可解释的失败判定，从根本上解决遮挡、感知噪声、物理不一致与因果链缺失等真实场景中的核心问题。

---

#️⃣ 11. 优化方案（基于 demo1.ipynb 分析）

基于实际实现（`demo1.ipynb`）的分析，以下是高优先级和中优先级的优化方案：

## 11.1 高优先级优化

### 11.1.1 约束生成格式优化

**问题**：LLM 生成的是自然语言格式，缺少结构化 JSON 和可执行 AST。

**解决方案**：
- 改进 LLM Prompt，要求生成结构化 JSON 格式
- JSON 包含：`id`, `type`, `description`, `condition_expr`, `severity`, `eval_time`
- LLM 直接生成可执行的 `condition_expr`（AST 格式）

**实现位置**：
- `reasoning/llm_prompter.py`：更新 `constraint-generator` prompt
- `reasoning/constraint_generator.py`：更新 `_parse_constraints` 方法支持 JSON 解析

### 11.1.2 约束编译格式优化

**问题**：当前格式 `Mug is_inside Sink` 无法直接执行。

**解决方案**：
- 生成标准 AST 格式：`(inside mug sink)`
- 支持复杂逻辑组合：`(and (inside mug sink) (not (inside mug coffee_machine)))`
- 如果 LLM 已生成 `condition_expr`，直接使用

**实现位置**：
- `reasoning/constraint_generator.py`：改进 `compile_constraint` 方法

### 11.1.3 时序验证优化

**问题**：没有区分 pre/post 约束的评估时间，只在最终状态验证。

**解决方案**：
- 创建 `ConstraintEvaluator` 类评估 AST 表达式
- 在动作前验证 precondition
- 在动作后验证 postcondition
- 持续验证 invariant
- 在任务完成时验证 goal

**实现位置**：
- `reasoning/constraint_evaluator.py`：新建约束评估器
- `demo1.ipynb` Step 6：添加时序验证逻辑

## 11.2 中优先级优化

### 11.2.1 场景图属性完善

**问题**：缺少时间特征和几何属性。

**解决方案**：
- 更新 `Node` 类添加：`bbox`, `pose`, `confidence`, `last_seen_ts`, `velocity`
- 在场景图生成时填充这些属性

**实现位置**：
- `core/scene_graph.py`：更新 `Node` 类
- `demo1.ipynb` Step 3：填充属性

### 11.2.2 因果链约束支持

**问题**：缺少跨动作的因果依赖约束。

**解决方案**：
- 在 LLM Prompt 中添加因果链要求
- 添加 `causal_chain` 约束类型
- 验证时检查因果链依赖

**实现位置**：
- `reasoning/llm_prompter.py`：更新 prompt
- `reasoning/constraint_generator.py`：支持因果链类型
- `demo1.ipynb` Step 6：添加因果链验证

## 11.3 完整实现流程

```
1. 数据生成 (AI2THOR)
   ↓
2. 场景图生成（包含完整属性）
   ↓
3. 约束生成 (LLM) → 结构化 JSON + AST
   ↓
4. 约束编译（可选，如果 LLM 已生成则跳过）
   ↓
5. 时序验证（动作前后分别验证）
   ↓
6. 失败检测（使用 ConstraintEvaluator）
   ↓
7. 渐进式解释（包含因果链分析）
```

## 11.4 预期效果

- ✅ 约束质量提升：结构化 JSON + 可执行 AST
- ✅ 验证准确性提升：时序验证能够准确检测动作相关的违反
- ✅ 场景图信息完整性：包含时间和几何属性
- ✅ 因果链支持：能够检测因果违反

详细优化方案请参考：`Method_OPTIMIZATION.md`
