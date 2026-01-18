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

### 方案 A：DETIC + CLIP 集成（推荐）⭐

DETIC + CLIP 集成有两种方式：

#### 方案 A1：CLIP 文本嵌入替换分类器（推荐，Zero-shot Learning）⭐

**工作原理**：
1. 使用 CLIP 文本编码器生成自定义词汇表的文本嵌入
2. 将 CLIP 嵌入替换到 DETIC 模型的分类器中（`reset_cls_test`）
3. DETIC 直接输出自定义类名（如 "purple cup", "blue cup with handle"）

**核心优势**：
- ✅ **一步到位**：DETIC 直接输出自定义类名，无需后处理匹配
- ✅ **Zero-shot 学习**：能识别训练时未见过的类名（如 "purple cup"）
- ✅ **准确性高**：CLIP 语义理解能力直接集成到检测流程中
- ✅ **实现简单**：与 Detic_demo 官方实现一致

**实现代码**：
```python
from detic.modeling.text.text_encoder import build_text_encoder
from detic.modeling.utils import reset_cls_test

def get_clip_embeddings(vocabulary, prompt='a '):
    """Generate CLIP embeddings for custom vocabulary"""
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    texts = [prompt + x for x in vocabulary]
    emb = text_encoder(texts).detach().permute(1, 0).contiguous().cpu()
    return emb

# 设置自定义词汇表
metadata.thing_classes = ['coffee maker', 'purple cup', 'blue cup with handle', 'sink']
classifier = get_clip_embeddings(metadata.thing_classes)
reset_cls_test(predictor.model, classifier, len(metadata.thing_classes))

# 检测结果直接是自定义类名
outputs = predictor(image)  # 输出: ['purple cup', 'blue cup with handle', ...]
```

**工作流程**：
```
RGB-D Stream
    ↓
DETIC Detection (使用CLIP嵌入的分类器)
    ↓
直接输出自定义类名 (如 "purple cup", "coffee maker")
    ↓
ByteTrack Tracking (multi-object tracking, 可选)
    ↓
Environment Memory (temporal smoothing, occlusion handling)
    ↓
Scene Graph Construction (with confidence scores)
```

#### 方案 A2：DETIC 检测 + CLIP 后处理匹配（两步过程）

**工作原理**：
1. DETIC 使用 LVIS 词汇表检测基础类别（如 "cup", "sink"）
2. 使用 CLIP 进行语义匹配，将检测结果映射到原始对象描述
3. 通过相似度阈值过滤匹配

**核心问题**：
- ⚠️ **两步过程**：先检测基础类别，再匹配自定义描述，精度较低
- ⚠️ **匹配复杂**：需要额外的 CLIP 匹配逻辑，容易出错
- ⚠️ **信息损失**：属性信息（如 "purple", "blue"）可能丢失

**实现代码**（不推荐）：
```python
# 1. DETIC检测LVIS类别
outputs = predictor(image)  # 输出: ['cup', 'sink', ...]

# 2. CLIP匹配到自定义描述
for detection in detections:
    # 使用CLIP匹配到object_list
    matched = clip_match(detection.label, object_list)
    detection.label = matched  # 可能匹配失败
```

**对比总结**：

| 特性 | 方案 A1（推荐） | 方案 A2（不推荐） |
|------|----------------|------------------|
| **检测输出** | 直接是自定义类名 | 先输出基础类别，再匹配 |
| **准确性** | ⭐⭐⭐ 高 | ⭐⭐ 中等 |
| **实现复杂度** | ⭐ 简单 | ⭐⭐⭐ 复杂 |
| **信息保留** | ✅ 完整保留属性 | ⚠️ 可能丢失属性 |
| **Zero-shot能力** | ✅ 支持 | ⚠️ 有限 |
| **与官方Demo一致性** | ✅ 完全一致 | ❌ 不一致 |

**推荐使用方案 A1**，这是 Detic 官方 demo 采用的方法，准确性和实现简单性都更好。

**历史演进说明**：

- **原始实现（方案 A2）**：在早期实现中，我们尝试先使用 DETIC 检测 LVIS 类别，然后使用 CLIP 进行后处理匹配。这种方法虽然可行，但存在精度损失和信息丢失的问题。

- **改进实现（方案 A1）**：通过分析 Detic 官方 demo 的实现，我们采用了 CLIP 文本嵌入替换分类器的方法。这种方法将 CLIP 的语义理解能力直接集成到 DETIC 的检测流程中，实现了 Zero-shot 学习，能够直接识别自定义类名。

- **当前实现（demo4.ipynb）**：现在 demo4 使用方案 A1，与 Detic_demo 完全一致，确保了最佳的性能和准确性。

**与 REFLECT 方法的对比**：

详细对比 Demo4 当前实现（方案A1/A2）与 REFLECT 方法（MDETR + CLIP验证 + 点云处理）的区别，请参考：

📄 **详细对比文档**：`REAL_WORLD_METHOD_COMPARISON.md`

**快速对比**：

| 特性 | Demo4 (A1) | REFLECT |
|------|-----------|---------|
| **物体检测** | DETIC + CLIP嵌入 | MDETR + CLIP验证 |
| **检测输出** | 直接自定义类名 | 逐个类别检测后合并 |
| **CLIP验证** | ❌ 不需要 | ✅ 裁剪区域验证 (阈值>0.23) |
| **点云提取** | ❌ 仅3D位置 | ✅ 掩码提取点云 |
| **空间关系** | 3D位置计算 | 点云距离+边界框 |
| **Gripper状态** | ❌ 未实现 | ✅ 基于位置推断 |

**关键区别**：
- **Demo4优势**：Zero-shot能力强、速度快、实现简单
- **REFLECT优势**：可靠性高、点云精确、支持Gripper状态

详见对比文档了解完整差异和混合方案建议。

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

**混合方法改进**：
- 仿真环境：优先使用 metadata（置信度 1.0），位置信息作为补充（置信度 0.85）
- 真实环境：metadata 不可用时，使用位置/点云方法（置信度 0.75-0.85）
- 动态表面类型判断：避免写死对象类型，使用关键词匹配

## 1.5.3.5 混合空间关系判断方法（Hybrid Spatial Relation Detection）

**核心思想**：结合 CRAFT 和 REFLECT 的优势，采用优先级策略判断空间关系

### 优先级策略

1. **优先级 1：基于 Metadata（CRAFT 方法，置信度 1.0）**
   - 使用 AI2THOR 的 `parentReceptacles` 和 `receptacleObjectIds`
   - 动态判断关系类型（容器 vs 表面）
   - 完全可靠，基于 ground truth 数据

2. **优先级 2：基于位置信息（CRAFT 方法，置信度 0.85）**
   - 使用 3D position 计算 z_diff 和 horizontal_dist
   - 动态表面类型检测（关键词匹配）
   - 阈值：`0.05 < z_diff < 0.5m` 且 `horizontal_dist < 0.2m`

3. **优先级 3：基于点云（REFLECT 方法，置信度 0.75）**
   - 使用点云距离和边界框检查
   - 当 metadata 不可用且位置信息不足时使用
   - 需要点云数据可用

### 实现代码

```python
def determine_spatial_relation_hybrid(obj1, obj2, node1, node2, use_point_cloud=False):
    """
    混合方法：优先 metadata，备用位置/点云信息
    """
    # Priority 1: Metadata-based (confidence 1.0)
    if obj1.get('parentReceptacles'):
        for parent_id in obj1.get('parentReceptacles', []):
            if obj2.get('objectId') == parent_id:
                has_receptacle = bool(obj2.get('receptacleObjectIds', []))
                is_openable = obj2.get('openable', False) or 'isOpen' in obj2
                if is_openable or has_receptacle:
                    return ("inside", 1.0)
                else:
                    return ("on_top_of", 1.0)
    
    # Priority 2: Position-based (confidence 0.85)
    if node1.position and node2.position:
        z_diff = node1.position[2] - node2.position[2]
        horizontal_dist = np.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
        obj2_type = obj2.get('objectType', '').lower()
        is_surface = any(kw in obj2_type for kw in ['countertop', 'table', 'stoveburner', 'burner', 'sink'])
        
        if (0.05 < z_diff < 0.5 and horizontal_dist < 0.2 and is_surface):
            return ("on_top_of", 0.85)
    
    # Priority 3: Point cloud-based (confidence 0.75)
    if use_point_cloud and node1.pcd and node2.pcd:
        dist = get_point_cloud_distance(node1.pcd, node2.pcd)
        if dist < 0.1:
            if is_inside_point_cloud(node1.pcd, node2.pcd, 0.5):
                obj2_type = obj2.get('objectType', '').lower()
                is_surface = any(kw in obj2_type for kw in ['countertop', 'stoveburner', 'burner', 'sink'])
                if is_surface:
                    return ("on_top_of", 0.75)
                else:
                    return ("inside", 0.75)
    
    return None
```

### 优势

- ✅ **准确性**：优先使用可靠的 metadata，避免误判
- ✅ **可扩展性**：动态判断，不依赖写死的类型列表
- ✅ **鲁棒性**：当 metadata 不可用时，自动回退到位置/点云方法
- ✅ **置信度分数**：每个关系都有置信度，便于后续处理

## 1.5.4 真实环境完整流程

### DETIC + CLIP 方案 A1（推荐：CLIP嵌入替换分类器）

```
RGB-D Stream
    ↓
DETIC Detection (使用CLIP嵌入的分类器)
    - 自定义词汇表: ['coffee maker', 'purple cup', 'blue cup with handle', 'sink']
    - 直接输出自定义类名（Zero-shot learning）
    ↓
ByteTrack Multi-object Tracking (可选)
    - Track IDs across frames
    - Handle occlusion and ID switches
    ↓
Scene Graph Construction (with confidence)
    - Nodes: objects with DETIC confidence
    - Edges: spatial relations with confidence
    ↓
Environment Memory (temporal smoothing, occlusion handling)
    - Kalman-like position smoothing
    - Occlusion prediction
    - Confidence decay for unseen objects
    ↓
Smoothed Scene Graph (for constraint validation)
```

**关键改进**：
- ✅ **Zero-shot 学习**：使用 CLIP 文本嵌入替换 DETIC 分类器，直接识别自定义类名
- ✅ **一步到位**：检测结果直接是自定义类名，无需后处理匹配
- ✅ **准确性高**：CLIP 的语义理解能力直接集成到检测流程中

### DETIC + CLIP 方案 A2（不推荐：两步匹配过程）

```
RGB-D Stream
    ↓
DETIC Detection (LVIS词汇表)
    - 输出基础类别: ['cup', 'sink', ...]
    ↓
CLIP Semantic Matching (后处理)
    - 将检测结果匹配到object_list
    - Semantic similarity filtering (threshold: 0.25)
    - 可能匹配失败或信息丢失
    ↓
ByteTrack Multi-object Tracking
    ↓
Scene Graph Construction
    ↓
Environment Memory
    ↓
Smoothed Scene Graph
```

**关键问题**：
- ⚠️ **两步过程**：先检测基础类别，再匹配自定义描述，精度较低
- ⚠️ **信息损失**：属性信息（如颜色）可能在匹配过程中丢失

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

**混合方法应用**：
- 仿真环境：优先使用 metadata（parentReceptacles），置信度 1.0
- 真实环境：metadata 不可用时，使用位置/点云方法，置信度 0.75-0.85
- 动态判断：所有方法都使用动态表面类型检测，避免写死对象类型


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
        # Postcondition: 使用 Postcondition Temporal Window 检查
        # 关键改进：不在 immediate next frame 检查，而是在时间窗口内检查
        # 避免延迟更新导致的误报（如物理下落、状态更新延迟）
        
        # 根据动作类型确定窗口大小 K
        if action_type == 'toggle':
            K = 5  # toggle 动作：3-5 帧
        elif action_type in ['put_in', 'put_on']:
            K = 8  # put_in/put_on 动作：5-10 帧
        elif action_type == 'pick_up':
            K = 3  # pick_up 动作：2-3 帧
        else:
            K = 5  # 默认：5 帧
        
        # Postcondition Temporal Window: [f_end(i), f_end(i)+1, ..., f_end(i)+K]
        start_frame = action_index + 1
        end_frame = min(start_frame + K, len(events))
        
        post_satisfied = False
        for check_frame in range(start_frame, end_frame):
            eval_scene_graph = GenerateSceneGraph(events[check_frame])
            eval_scene_graph = eval_scene_graph.extract_task_relevant_subgraph(task_info)
            
            (value, atom_conf) = EvalPredicate(constraint.condition_ast, eval_scene_graph, memory)
            if value == True:
                post_satisfied = True
                break  # 只要窗口内任何一帧满足，就认为满足
        
        if post_satisfied:
            return SATISFIED
        else:
            return VIOLATED
        
        evaluation_time = f"after action {action_index + 1} (temporal window [{start_frame}-{end_frame-1}])"
    
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

⸻

#️⃣ 12. 核心优化方案（基于失败检测实践）

基于实际失败检测实践，CRAFT++ 在失败检测、错误聚合、因果链解释这三件核心事情上是"成立的"，但目前存在以下问题需要优化：

## 12.1 问题分析

### 12.1.1 约束噪声过大

**问题描述**：
- 一个早期失败 → 后续所有 postcondition 全部失败 → 数量级膨胀
- 例如：`makeCoffee` 任务，真实"致命失败"应该只有 1–2 个，但现在检测到大量级联失败

**根本原因**：
- 现在正确地检测到了级联失败，但没有收敛它
- 所有后续动作的 postcondition 都因为前置动作失败而失败

### 12.1.2 错误源未分层

**问题描述**：
- Robot 相关 Warning 过多：`Node 'Robot' not found in scene graph`、`'NoneType' object has no attribute 'name'`
- 工程上必须优化（否则 reviewer 会盯）

**根本原因**：
- 场景图是 object-centric，但约束假设存在 `holding(robot, mug)`
- 场景图中没有 robot 节点，导致大量警告

### 12.1.3 仿真伪失败未被充分隔离

**问题描述**：
- CRAFT 把"动作执行失败"和"感知未更新"混在了一起
- 例如：Mug 实际被放上去了，但 scene graph 没更新 → postcondition false

**根本原因**：
- 没有区分 Execution failure 和 Perception failure
- 所有失败都被当作执行失败处理

### 12.1.4 LLM 分析存在幻觉

**问题描述**：
- LLM 分析逻辑正确，但"话太多 + 有轻微幻觉"
- 例如："CoffeeMachine was not empty because mug was dirty" 这个推断链条有点 speculative，不是严格从约束推出来的

**根本原因**：
- LLM 可以"补故事"，而不是"解释已验证的约束失败"
- CRAFT 的哲学是：LLM 不能"补故事"，只能"解释已验证的约束失败"

⸻

## 12.2 优化方案

### 12.2.1 Failure Root Collapsing 机制

**目标**：将级联失败收敛到根失败，减少约束噪声

**实现方法**：

```python
def collapse_failures(violations):
    """
    收敛级联失败到根失败
    
    Args:
        violations: List[Dict] - 所有违反的约束
        
    Returns:
        Dict - 包含 root violation 和 derived violations
    """
    # 找到最早的致命违反（precondition violation）
    root = None
    earliest_step = float('inf')
    
    for v in violations:
        # 只考虑 precondition violation（致命失败）
        if v.get('failure_type') == 'PRECONDITION_VIOLATION':
            step = v.get('step', float('inf'))
            if step < earliest_step:
                earliest_step = step
                root = v
    
    if root is None:
        # 如果没有 precondition violation，返回第一个 violation
        root = violations[0] if violations else None
        earliest_step = root.get('step', 0) if root else 0
    
    # 分离根失败和派生失败
    derived = [v for v in violations if v.get('step', 0) > earliest_step]
    
    return {
        "root": root,
        "derived": derived,
        "root_step": earliest_step,
        "total_violations": len(violations),
        "collapsed_count": len(derived)
    }
```

**效果**：
- `makeCoffee` 任务从 15+ 个违反收敛到 1–2 个根失败
- 后续所有 postcondition 失败被标记为"派生失败"，不单独报告

⸻

### 12.2.2 虚拟 Robot 节点（Dummy Agent Node）

**目标**：解决 Robot 相关警告，消除 `Node 'Robot' not found` 错误

**实现方法**：

```python
def add_virtual_robot_node(scene_graph: SceneGraph):
    """
    添加虚拟 Robot 节点到场景图
    
    即使 Robot 节点在感知中不存在，也创建一个虚拟节点
    用于约束评估（如 holding(robot, mug)）
    """
    # 检查是否已存在 Robot 节点
    robot_exists = False
    for node in scene_graph.nodes:
        if node.name.lower() == 'robot' or node.object_type.lower() == 'robot':
            robot_exists = True
            break
    
    if not robot_exists:
        # 创建虚拟 Robot 节点
        robot_node = Node(
            name="Robot",
            object_type="agent",
            attributes={
                "gripper": "empty",  # 默认 gripper 为空
                "is_virtual": True   # 标记为虚拟节点
            },
            pose=None,  # 位置未知（模拟环境）
            confidence=1.0
        )
        scene_graph.add_node(robot_node)
    
    return scene_graph
```

**效果**：
- 消除所有 `Node 'Robot' not found` 警告
- `holding(robot, mug)` 等约束可以正常评估
- 虚拟节点的 `gripper` 状态可以通过约束评估动态更新

⸻

### 12.2.3 Execution vs Perception Failure Distinction

**目标**：区分动作执行失败和感知未更新

**实现方法**：

```python
def classify_failure_type(violation: Dict, action_result: Optional[Dict] = None) -> str:
    """
    分类失败类型：Execution failure vs Perception failure
    
    Args:
        violation: 约束违反信息
        action_result: 动作执行结果（如果可用）
        
    Returns:
        str: "execution_failure" | "perception_failure" | "unknown"
    """
    # 如果动作执行成功，但 postcondition 失败，可能是感知问题
    if action_result and action_result.get('status') == 'SUCCESS':
        if violation.get('failure_type') == 'POSTCONDITION_VIOLATION':
            # 检查是否是感知未更新导致的
            # 例如：对象实际被移动了，但 scene graph 没更新
            return "perception_failure"
    
    # 如果 precondition 失败，通常是执行失败
    if violation.get('failure_type') == 'PRECONDITION_VIOLATION':
        return "execution_failure"
    
    # 如果动作执行失败，明确是执行失败
    if action_result and action_result.get('status') == 'FAILED':
        return "execution_failure"
    
    return "unknown"

# 在失败检测中使用
for violation in violations:
    failure_type = classify_failure_type(violation, action_result)
    violation['failure_category'] = failure_type
    
    if failure_type == "perception_failure":
        # 标记为感知失败，不当作致命错误
        violation['is_warning'] = True
        violation['reason'] += " (可能由感知未更新导致)"
```

**效果**：
- 区分真实的执行失败和感知噪声
- 感知失败被标记为警告，不影响根因分析
- 提高失败检测的准确性

⸻

### 12.2.4 LLM 分析优化（限制因果来源）

**目标**：限制 LLM 的"因果来源"，避免幻觉，只解释已验证的约束失败

**实现方法**：

```python
def build_llm_analysis_prompt(task_info: Dict, root_violation: Dict, 
                              all_violations: List[Dict]) -> Tuple[str, str]:
    """
    构建优化的 LLM 分析 Prompt
    
    关键改进：
    1. 显式指定 Root Violation
    2. 限制 LLM 只能基于列出的约束失败进行解释
    3. 禁止引入假设
    """
    system_prompt = """You are a robot task failure analyzer. Analyze constraint violations and identify the root cause of failures.

CRITICAL RULES:
1. Only explain failures strictly based on the listed constraint violations.
2. Do NOT introduce assumptions beyond the constraints.
3. Do NOT speculate about causes not mentioned in the constraint violations.
4. Focus on the ROOT VIOLATION as the primary cause.
5. Explain derived failures as consequences of the root violation.

Your task is to:
1. Identify the PRIMARY root cause (the root violation)
2. Explain WHY the root violation occurred (based on the constraint description)
3. Explain how derived failures are consequences of the root violation
4. Suggest what should have been done differently (based on the root violation only)"""
    
    # 构建错误描述（只包含真实错误，排除警告）
    real_errors = [v for v in all_violations if not v.get('is_warning', False)]
    
    error_descriptions = []
    for i, error in enumerate(real_errors, 1):
        constraint = error.get('constraint', {})
        action = error.get('action', 'N/A')
        step = error.get('step', '?')
        reason = error.get('reason', 'N/A')
        description = constraint.get('description', 'N/A')
        failure_type = error.get('failure_type', 'Unknown')
        
        error_descriptions.append(
            f"{i}. [{failure_type}]\n"
            f"   动作: {action} (Step {step})\n"
            f"   约束: {description}\n"
            f"   失败原因: {reason}"
        )
    
    # 显式指定 Root Violation
    root_desc = f"""
ROOT FAILURE (Primary Cause):
- Step {root_violation.get('step', '?')}: {root_violation.get('action', 'N/A')}
- Constraint: {root_violation.get('constraint', {}).get('description', 'N/A')}
- Reason: {root_violation.get('reason', 'N/A')}

All later failures are consequences of this root failure.
"""
    
    user_prompt = f"""Task: {task_info.get('name', 'Unknown')}

Task Goal: {task_info.get('success_condition', 'N/A')}

{root_desc}

All Constraint Violations ({len(real_errors)} errors):
{chr(10).join(error_descriptions)}

Please analyze:
1. What is the PRIMARY root cause? (The root violation listed above)
2. Why did the root violation occur? (Explain based ONLY on the constraint description and reason)
3. How are the derived failures consequences of the root violation?
4. What should have been done differently? (Based on the root violation only)

IMPORTANT: Do NOT introduce assumptions beyond what is stated in the constraint violations above."""
    
    return system_prompt, user_prompt
```

**效果**：
- LLM 只基于已验证的约束失败进行解释
- 避免幻觉式推断（如 "mug was dirty"）
- 显式指定 Root Violation，让 LLM 聚焦核心问题
- 提高分析的可信度和可解释性

⸻

## 12.3 完整优化流程

优化后的失败检测流程：

```
1. 场景图构建
   ↓
2. 添加虚拟 Robot 节点（如果不存在）
   ↓
3. 约束生成和编译
   ↓
4. 失败检测（按动作顺序）
   ↓
5. 失败分类（Execution vs Perception）
   ↓
6. Failure Root Collapsing（收敛级联失败）
   ↓
7. LLM 分析（基于 Root Violation，限制因果来源）
```

⸻

## 12.4 预期效果

优化后，CRAFT++ 应该能够：

1. **约束噪声大幅降低**：
   - `makeCoffee` 任务从 15+ 个违反收敛到 1–2 个根失败
   - 级联失败被正确标记为"派生失败"

2. **Robot 相关警告消除**：
   - 不再出现 `Node 'Robot' not found` 错误
   - `holding(robot, mug)` 等约束可以正常评估

3. **失败类型准确分类**：
   - 区分 Execution failure 和 Perception failure
   - 感知失败被标记为警告，不影响根因分析

4. **LLM 分析更可信**：
   - 只基于已验证的约束失败进行解释
   - 避免幻觉式推断
   - 显式指定 Root Violation，聚焦核心问题

⸻

## 12.5 动作分类与约束生成策略（分层设计）

### 12.5.1 问题分析

**当前问题**：
- `toggle_on(StoveBurner-4)` 等状态切换动作被标记为"未知动作类型"，未生成约束
- 这导致 `boilWater` 任务中关键的状态变化无法被检测

**根本原因**：
- 将所有未知动作统一处理，没有区分动作类型
- 状态切换动作（如 `toggle_on`）需要状态变量约束，但当前系统未生成

### 12.5.2 动作分类设计

CRAFT++ 采用**分层 + 保守 + 受控的 LLM 参与**策略，将动作分为三类：

#### A类：纯导航/定位动作（可安全忽略）

**特点**：
- 不改变世界状态
- 失败通常只影响后续动作是否可达
- 例如：`navigate_to_obj`, `look_at`, `turn_to`

**处理策略**：
- ✅ 不生成约束（可安全忽略）
- ✅ 为后续动作提供 `reachable` 先验（软约束）

**论文表述**：
> Navigation actions are treated as latent enabling actions and do not introduce explicit constraints.

#### B类：状态切换动作（必须建模，但不需要复杂几何）

**特点**：
- 改变对象的离散状态（on/off, open/closed）
- 不需要复杂的几何检查
- 例如：`toggle_on(Faucet)`, `toggle_on(StoveBurner-4)`, `open(Cabinet)`

**处理策略**：
- ✅ **必须生成状态变量约束**
- ✅ 至少生成 `POST: state(obj) == ON` 或 `POST: toggled_on(obj) == True`
- ✅ 不需要写"火一定要热""水一定要沸腾"，只要："这个对象被改变了状态"

**实现方法**：

```python
def _generate_state_variable_constraints(action, action_type, action_args):
    """
    为状态切换动作生成基本状态变量约束
    """
    if action_type == "toggle_on":
        return [{
            'type': 'postcondition',
            'template': f"toggled_on({obj_name})",
            'description': f"{obj_name} must be toggled on",
            'condition_expr': f"node.attributes.get('isToggled', False)"
        }]
    # ... 其他状态切换动作
```

**关键改进**：
- `toggle_on` 等动作**绝对不能**当"未知动作类型"
- 必须生成状态变量约束，即使没有完整的模板

#### C类：语义复合动作（可由 LLM 辅助生成"保守约束"）

**特点**：
- 高层语义强
- 低层实现差异大
- 规则很难写全
- 例如：`pour`, `wash`, `heat`, `clean`

**处理策略**：
- ✅ **LLM 只填空，不造规则**
- ✅ LLM 从预定义的约束类型中选择，不能自由创造
- ✅ 系统再把它翻译成代码约束

**实现方法**：

```python
def _generate_constraints_for_action_llm_constrained(action, action_type, action_args):
    """
    受限的 LLM 辅助生成（C类动作）
    
    LLM 只从预定义的约束类型中选择：
    1. State change (on/off, clean/dirty)
    2. Containment relation (inside, on_top_of)
    3. Spatial relation (near, in_contact)
    4. No constraint needed
    """
    prompt = f"""Given an action: {action}

Select applicable constraints from:
1. State change (on/off, clean/dirty)
2. Containment relation
3. Spatial relation
4. No constraint needed

Output only the selected types."""
    
    # LLM 返回选择，系统生成对应约束
```

**关键原则**：
- ❌ **错误方式**：让 LLM 自己判断这个动作应该有什么约束
- ✅ **正确方式**：LLM 只分类器/选择器，不是规则创造者

### 12.5.3 完整动作分类表

| 动作类型 | 分类 | 约束生成策略 | 示例 |
|---------|------|-------------|------|
| `navigate_to_obj` | A类（导航） | 不生成约束 | `navigate_to_obj(Pot)` |
| `toggle_on` | B类（状态切换） | 生成状态变量约束 | `toggle_on(Faucet)` → `POST: toggled_on(Faucet)` |
| `toggle_off` | B类（状态切换） | 生成状态变量约束 | `toggle_off(Faucet)` → `POST: toggled_off(Faucet)` |
| `open` | B类（状态切换） | 生成状态变量约束 | `open(Cabinet)` → `POST: container_open(Cabinet)` |
| `pour` | C类（语义复合） | LLM 受限选择 | `pour(Mug, Sink)` → LLM 选择约束类型 |
| `pick_up` | 标准模板 | 使用预定义模板 | `pick_up(Mug)` → 模板生成 |
| `put_in` | 标准模板 | 使用预定义模板 | `put_in(Mug, CoffeeMachine)` → 模板生成 |

### 12.5.4 论文表述

**推荐表述（硕士论文风格）**：

> Not all actions in embodied task execution require explicit geometric constraints. CRAFT categorizes actions into navigation, state-changing, and semantic composite actions. Navigation actions are treated as latent enablers and do not introduce explicit constraints. For state-changing actions such as `toggle_on`, CRAFT introduces lightweight state-variable postconditions to capture discrete world state transitions. For high-level semantic actions whose effects are difficult to manually enumerate, CRAFT employs a constrained LLM-assisted selection mechanism, where the language model selects from a predefined constraint schema rather than generating free-form rules. This design ensures both extensibility and determinism, avoiding the hallucination issues observed in prior LLM-only approaches.

### 12.5.5 预期效果

优化后，CRAFT++ 应该能够：

1. **正确识别状态切换动作**：
   - `toggle_on(StoveBurner-4)` 不再被标记为"未知动作类型"
   - 自动生成 `POST: toggled_on(StoveBurner-4)` 约束

2. **boilWater 任务完整检测**：
   - `toggle_on(Faucet)` → `POST: Faucet.state == ON`
   - `toggle_on(StoveBurner-4)` → `POST: StoveBurner-4.state == ON`
   - 结合 `put_in(Pot, Sink)` → `IF Faucet.state == ON ∧ inside(Pot, Sink) → Pot.filled == True`

3. **导航动作正确忽略**：
   - `navigate_to_obj` 不生成约束，不产生 false positive

4. **模拟环境错误硬性剔除**：
   - Robot/NoneType 相关错误不出现在 root-cause 候选集中

### 12.5.8 Postcondition Temporal Window（关键改进）⭐

**问题分析**：
- 当前实现在动作执行后的**下一帧（immediate next frame）**就检查 postcondition
- 但物理仿真中，状态更新往往是**延迟的**：
  - `put_in(Pot, Sink)` → `inside(pot, sink)` 需要物理下落（可能需要 3-7 帧）
  - `toggle_on(Faucet)` → `isToggled` 状态更新延迟（可能需要 2-5 帧）
- 这导致大量**假 postcondition violation**（false positives）

**根本原因**：
- AI2THOR 等仿真环境的物理更新和状态同步需要时间
- 不是 scene graph 生成错误，也不是 constraint 错误
- **是 postcondition evaluation timing 错误**

**解决方案：Postcondition Temporal Window**

**核心思想**：
不在 immediate next frame 检查，而是在一个**时间窗口**内检查。只要在窗口内任何一帧满足，就认为 postcondition 满足。

**定义**：

```
PostFrames(action_i) = [f_end(i), f_end(i)+1, ..., f_end(i)+K]

post_satisfied = any(
    check_postcondition(state(f)) 
    for f in PostFrames(action_i)
)
```

**窗口大小（K）的经验值（simulation）**：

| Action Type | K (frames) | 原因 |
|------------|-----------|------|
| `toggle` | 3-5 | 状态更新延迟 |
| `put_in` / `put_on` | 5-10 | 物理下落需要时间 |
| `pick_up` | 2-3 | 抓取状态更新较快 |
| 默认 | 5 | 保守估计 |

**实现伪代码**：

```python
# 根据动作类型确定窗口大小 K
if 'toggle' in action.lower():
    K = 5  # toggle 动作：3-5 帧
elif 'put_in' in action.lower() or 'put_on' in action.lower():
    K = 8  # put_in/put_on 动作：5-10 帧
elif 'pick_up' in action.lower():
    K = 3  # pick_up 动作：2-3 帧
else:
    K = 5  # 默认：5 帧

# Postcondition Temporal Window 检查
start_frame = action_idx + 1
end_frame = min(start_frame + K, len(events))

post_satisfied = False
satisfied_frame = None

for check_frame in range(start_frame, end_frame):
    eval_sg = GenerateSceneGraph(events[check_frame])
    is_valid, reason, _, diagnostics = EvaluateConstraint(eval_sg, constraint)
    
    if is_valid:
        post_satisfied = True
        satisfied_frame = check_frame
        break  # 只要窗口内任何一帧满足，就认为满足

if not post_satisfied:
    # 窗口内所有帧都不满足，才判定为 violation
    return VIOLATION
else:
    return SATISFIED
```

**效果**：
- ✅ **大幅减少假 postcondition violation**
- ✅ **准确反映真实的执行失败**（而不是时间延迟）
- ✅ **提高失败检测的准确性**

**论文级表述**：

> We observe that evaluating postconditions on the immediate next frame after action execution often leads to false positives due to delayed physical and state updates in simulation environments. Therefore, we adopt a **temporal postcondition evaluation strategy**, where a postcondition is considered satisfied if it emerges within a short temporal window following the action. The window size is dynamically determined based on the action type (e.g., 5-10 frames for manipulation actions, 3-5 frames for toggle actions), accounting for the varying latency of physical simulation and state synchronization.

**关键洞察**：
- 这不是 scene graph 生成错误，也不是 constraint 错误
- **是 postcondition evaluation timing 的错误**
- 这个问题正是 CRAFT 方法和原论文（如 REFLECT）真正拉开差距的地方

**预期改进**：
- `put_in` / `put_on` 动作的 postcondition violation 大幅减少
- `toggle` 动作的 postcondition violation 大幅减少
- 更准确的根因分析（不会因为时间延迟而误判）

### 12.5.9 Postcondition 违反的进一步诊断

**问题分析**：
即使使用了 Postcondition Temporal Window，仍然可能有 postcondition 违反。这些问题通常不是时间延迟的问题，而是：

1. **节点匹配问题**：约束中提到的对象名称（如 "Mug"）可能无法匹配到场景图中的节点（如 "Mug_0b3dbbd3"）
2. **空间关系未建立**：`on_top_of`、`inside` 等关系可能在场景图中没有正确建立
3. **状态属性同步延迟**：`isToggled`、`isFilled` 等属性虽然 metadata 更新了，但 scene graph 中可能还没有同步
4. **filled 约束语义不清**：对于容器（如 Sink），"filled" 的含义可能需要特别处理

**诊断方法**：

Step 5.5 排查 cell 提供了详细的诊断信息，包括：
- 约束中提到的对象是否在场景图中找到
- 相关节点和边的详细信息
- 状态属性与 metadata 的对比
- 场景图的完整信息（节点列表、边列表）

**可能的问题和解决方案**：

#### 问题 1：节点匹配失败

**症状**：约束中提到的对象未在场景图中找到

**解决方案**：
- 改进 `find_node_by_name` 函数，使用更健壮的匹配逻辑
- 提取对象名称的基础部分（去掉 ID），进行匹配
- 同时匹配对象名称和对象类型

#### 问题 2：空间关系未建立

**症状**：约束中提到的对象都在场景图中找到，但相关边不存在

**解决方案**：
- 检查 `parentReceptacles` metadata 是否存在
- 检查位置信息是否准确
- 调整位置判断的阈值（z_diff, horizontal_dist）
- 扩展表面类型关键词列表（确保包含所有可能的表面类型）

#### 问题 3：状态属性同步延迟

**症状**：节点存在，但状态属性（isToggled, isFilled）与 metadata 不一致

**解决方案**：
- 确保每个 frame 都正确同步状态属性
- 检查 metadata 字段名是否正确（如 `isToggledOn` vs `isToggled`）
- 使用多个字段作为回退（如 `obj.get('isToggledOn', False) or obj.get('isToggled', False)`）

#### 问题 4：filled 约束语义不清

**症状**：Sink 的 filled 检查失败，但 Sink 内部有液体对象

**解决方案**：
- 对于容器（如 Sink），filled 应该检查：
  1. 容器本身的 `isFilled` 属性
  2. 容器内部是否有液体对象
  3. 容器的 `fillLiquid` 属性

**改进的 filled 检查逻辑**：
```python
def check_filled(sg, obj_name):
    """检查对象是否 filled（改进版）"""
    node = find_node_by_name(sg, obj_name)
    if not node:
        return False, "Node not found"
    
    # 1. 检查 isFilled 属性
    if node.attributes.get('isFilled', False):
        return True, "isFilled attribute is True"
    
    # 2. 对于容器（如 Sink），检查内部是否有液体对象
    if 'sink' in obj_name.lower() or 'container' in node.object_type.lower():
        inside_liquid_objects = []
        for edge in sg.edges.values():
            if edge.end.name == node.name and edge.edge_type == 'inside':
                inside_obj = edge.start
                if inside_obj.attributes.get('isFilled', False):
                    inside_liquid_objects.append(inside_obj.name)
        if inside_liquid_objects:
            return True, f"Contains filled objects: {inside_liquid_objects}"
    
    # 3. 检查 fillLiquid 属性
    if node.attributes.get('fillLiquid'):
        return True, f"fillLiquid: {node.attributes.get('fillLiquid')}"
    
    return False, "Not filled"
```

### 12.5.6 模拟环境中的 Robot 交互约束过滤

**问题分析**：
- 在模拟环境中，robot 交互相关的约束（`holding`, `gripper_empty`）不可靠
- 这些约束不是任务失败的根本原因，而是模拟环境的实现细节
- 例如：`holding(mug)` 在模拟环境中可能因为 Robot 节点不存在或状态不同步而失败

**解决方案**：
- 在失败检测阶段，自动跳过 robot 交互相关的约束
- 这些约束被标记为 `SKIPPED`，不视为失败
- 只关注任务相关的物理约束（如 `container_empty`, `inside`, `on_top_of`）

**实现方法**：
```python
# 在约束检查循环中
is_robot_interaction = (
    'holding' in constraint_template or 
    'holding' in constraint_desc or
    'gripper_empty' in constraint_template or
    'gripper' in constraint_desc or
    'robot must be holding' in constraint_desc
)
if is_robot_interaction:
    skipped_constraints.append({
        'reason': "SKIPPED: Robot interaction constraint in simulation environment"
    })
    continue  # 跳过 robot 交互约束
```

**效果**：
- 减少模拟环境中的误报
- 聚焦任务相关的物理约束
- 提高失败检测的准确性

### 12.5.7 状态属性同步优化（每帧更新）

**问题分析**：
- 状态属性（如 `isToggled`, `isOpen`, `isFilled`）可能只在 final frame 读取
- 导致中间帧的状态不正确，约束验证失败
- 例如：`Faucet must be toggled on` 在动作执行后应该为 True，但 scene graph 中仍为 False

**解决方案**：
- **每个 event frame 都同步更新状态属性**
- 在 scene graph 构建时，直接从 `obj_metadata` 中读取状态
- 确保每个时间步的状态都正确反映

**实现方法**：
```python
# 在 generate_scene_graph_from_event_enhanced 中
node = Node(
    name=obj.get('name', 'unknown'),
    attributes={
        # 状态属性：从 metadata 中直接读取，每个 frame 都更新
        'isToggled': obj.get('isToggledOn', False) or obj.get('isToggled', False),
        'isOpen': obj.get('isOpen', False),
        'isFilled': obj.get('isFilledWithLiquid', False) or obj.get('isFilled', False),
        # ... 其他状态属性
    }
)
```

**关键原则**：
- ✅ **每个 event frame 都生成新的 scene graph**
- ✅ **状态属性从当前 frame 的 metadata 中读取**
- ✅ **不使用缓存或之前 frame 的状态**

**效果**：
- 状态属性在每个时间步都正确
- 约束验证能够准确检测状态变化
- 避免"状态未更新"导致的误报

### 12.5.10 put_on 约束语义自适应改进

**问题分析**：
- `put_on` 动作模板默认生成 `on_top_of` 约束，但对于容器类型（如 Sink, SinkBasin），应该生成 `inside` 约束
- 实际情况：`put_on(Mug, SinkBasin)` 在场景图中显示为 `Mug --[inside]--> SinkBasin`，但约束生成的是 `Mug must be on top of SinkBasin`
- 语义不匹配导致约束验证失败

**解决方案**：
- 在约束生成时，根据目标对象类型判断应该生成 `inside` 还是 `on_top_of` 约束
- 如果目标对象是容器类型（Sink, SinkBasin, Bowl 等），`put_on` 应该生成 `inside` 约束
- **Sink 和 SinkBasin 视为等价**（都是容器类型）

**实现方法**：
```python
# 在 _generate_constraints_for_action 方法中
# 特殊处理：put_on 动作根据目标对象类型判断应该生成 inside 还是 on_top_of 约束
if action_type == "put_on" and predicate == "on_top_of" and len(bound_args) >= 2:
    target_obj = bound_args[1]
    target_obj_lower = target_obj.lower()
    
    # 定义容器类型（包括 Sink/SinkBasin 的等价处理）
    CONTAINER_TYPES = {
        'sink', 'sinkbasin',  # Sink 和 SinkBasin 视为等价
        'bowl', 'pot', 'pan', 'mug', 'cup', 'coffeemachine',
        'fridge', 'cabinet', 'drawer', 'microwave', 'oven'
    }
    
    # 判断目标对象是否为容器类型
    is_container = any(container_type in target_obj_lower for container_type in CONTAINER_TYPES)
    
    if is_container:
        # 对于容器类型，生成 inside 约束而不是 on_top_of
        predicate = "inside"
```

**关键原则**：
- ✅ **动态判断**：根据目标对象类型动态选择约束类型
- ✅ **语义对齐**：确保生成的约束与场景图中的实际关系一致
- ✅ **等价处理**：Sink 和 SinkBasin 视为等价（都是容器类型）

**效果**：
- `put_on(Mug, SinkBasin)` 现在会生成 `inside(Mug, SinkBasin)` 约束
- 约束与场景图中的实际关系一致
- 避免语义不匹配导致的约束验证失败

### 12.5.11 Faucet Toggle 状态检查改进

**问题分析**：
- `toggle_on(Faucet)` 执行后，`isToggled` 属性仍然是 `False`
- metadata 中 `isToggledOn` 是 `None`，`isToggled` 是 `False`
- AI2THOR 中，某些对象（如 Faucet）的 toggle 状态可能使用不同的字段名

**根本原因**：
- AI2THOR 的 metadata 中，不同对象的 toggle 状态可能使用不同的字段名
- 某些对象使用 `isToggledOn`，某些对象使用 `isToggled`，某些对象使用 `isOn`
- 只检查单一字段会导致状态检测失败

**解决方案**：
- 改进 `isToggled` 属性的提取逻辑，检查多个可能的字段
- 优先使用 `isToggledOn`，回退到 `isToggled`，再检查 `toggleable` 属性和 `isOn` 字段
- 在所有 scene graph 生成的地方统一使用修复后的逻辑

**实现方法**：
```python
# 在 generate_scene_graph_from_event_enhanced 和所有 scene graph 生成函数中
'isToggled': (
    obj.get('isToggledOn', False) or 
    obj.get('isToggled', False) or
    (obj.get('toggleable', False) and obj.get('isOn', False)) or
    obj.get('isOn', False)
),
```

**关键原则**：
- ✅ **多字段检查**：检查多个可能的 toggle 状态字段
- ✅ **统一逻辑**：在所有 scene graph 生成的地方使用相同的逻辑
- ✅ **容错处理**：即使某些字段不存在，也能正确读取状态

**效果**：
- Faucet 等对象的 toggle 状态现在能正确检测
- 支持多种 toggle 状态字段格式
- 避免"状态未检测到"导致的约束验证失败
