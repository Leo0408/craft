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

#️⃣ 1.1 场景图裁剪（Task-Relevant Subgraph Extraction）

**核心思想**：从完整场景图中裁剪出"与当前子任务相关的最小子图"，减少复杂度，提高约束生成和验证的效率。

**问题**：
	•	完整场景图可能包含大量无关对象（例如：90个节点，36条边）
	•	约束生成和验证时只需要关注与任务相关的对象
	•	减少场景图大小可以：
		- 降低 LLM 输入长度
		- 提高约束生成质量
		- 加快约束验证速度

**解决方案**：
	•	从任务信息中提取相关对象（从 actions、success_condition、preactions）
	•	只保留相关对象及其直接关系
	•	支持精确匹配和部分匹配（处理对象名变体）

✔ 伪代码

Algorithm ExtractTaskRelevantSubgraph(full_scene_graph, task_info):

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
    
    # 2. 创建子图
    subgraph = SceneGraph()
    
    # 3. 查找相关节点（支持精确匹配和部分匹配）
    for node in full_scene_graph.nodes:
        if IsRelevant(node, relevant_objects):
            subgraph.add_node(node)
    
    # 4. 添加相关边
    for edge in full_scene_graph.edges:
        if (edge.start in subgraph.nodes OR edge.end in subgraph.nodes):
            # 如果边的端点至少有一个在子图中，保留该边
            # 如果端点不在子图中，也添加端点节点（保留直接关系）
            if edge.start not in subgraph.nodes:
                subgraph.add_node(edge.start)
            if edge.end not in subgraph.nodes:
                subgraph.add_node(edge.end)
            subgraph.add_edge(edge)
    
    return subgraph

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
	•	`core/scene_graph.py`：`SceneGraph.extract_task_relevant_subgraph()` 方法
	•	使用方式：`task_relevant_sg = full_sg.extract_task_relevant_subgraph(task_info)`


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

### 2.3 主算法：GenerateActionAwareConstraints
该算法遍历机器人的动作序列，为每一个动作实例化对应的物理约束。

```python
Algorithm GenerateActionAwareConstraints:
Input: scene_graph, action_sequence
Output: action_bound_constraints

BEGIN
    constraints = []
    FOR each action IN action_sequence:
        # 1. 查找动作模板
        template = ACTION_TEMPLATE_LIBRARY.get(action.name)
        
        # 2. 生成前置条件 (Preconditions)
        FOR each pre_template IN template["pre"]:
            constraints.append(Instantiate(pre_template, action, type='PRE'))
            
        # 3. 生成后置条件 (Postconditions)
        FOR each post_template IN template["post"]:
            constraints.append(Instantiate(post_item, action, type='POST'))
            
    RETURN constraints
END
```

### 2.4 约束实例化与编译（模板 → 可执行代码）
生成的结构化约束被直接映射为可执行的判定函数（Executable Predicates），避免了自然语言解析的不确定性。

*   **`holding(X)`**：`sg.robot_state["holding"] == X` (或 `node.attributes["isPickedUp"]`)
*   **`container_empty(Y)`**：`len([e for e in sg.edges.values() if e.end.name == Y and e.edge_type == 'inside']) == 0`
*   **`is_on(X, Y)`**：`sg.has_edge(X, Y, 'on_top_of')`
*   **`inside(X, Y)`**：`sg.has_edge(X, Y, 'inside')`

### 2.5 优势总结
该设计将失败检测从“目标状态一致性检查”升级为“**动作因果一致性验证**”，使系统能够在物理上不可能的动作发生时即时定位失败原因，实现与物理仿真环境对齐的精确归因。

⸻

#️⃣ 3. 环境记忆模块（Environment Memory）

为解决遮挡、跳变、噪声等现实问题：

EnvironmentMemory 使用：
	•	Kalman / Bayesian filter（位置 smoothing）
	•	last_seen state 存储
	•	occlusion prediction（根据机械臂与摄像头视锥）
	•	状态置信度衰减模型

✔ Memory 输出世界状态（WorldState）

WorldState:
    objects: {object_name → ObjectState}
    relations: {(a,b) → RelationState}
    occlusion_flags
    smoothed_positions
    last_seen
    velocity


⸻

✔ Memory 更新伪代码

Algorithm MemoryUpdate(raw_state):

    for each object in raw_state:
        if object.visible:
            apply_kalman_filter(object)
            update_last_seen(object)
        else:
            predict_position(object)
            mark_possible_occlusion(object)

    update_relations()
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

+------------------+
|   Perception     |
+------------------+
           |
           v
+---------------------------+
|    Environment Memory     |
+---------------------------+
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
|  (logic + geometry + mem)|
+---------------------------+
           |
           v
+---------------------------+
|   Failure Detection       |
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
