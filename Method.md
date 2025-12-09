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

框架包含三层：

(Perception + Memory) → Scene Graph → Constraint Compiler → Constraint Executor


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

#️⃣ 2. 约束生成（Constraint Generation）

LLM 负责将场景图 + 任务目标转换为：
	•	结构化 JSON 约束
	•	每个约束包含 Pre / Post / Invariants / Goal
	•	每个约束包含 condition_expr（可执行 DSL / AST）

⸻

✔ 2.1 LLM 生成的目标格式（结构化 JSON）

{
  "constraints": [
    {
      "id": "C1",
      "type": "pre",
      "description": "Machine must be open before inserting a cup",
      "condition_expr": "(eq machine.door 'open')",
      "severity": "hard",
      "eval_time": "pre"
    },
    {
      "id": "C2",
      "type": "post",
      "description": "Cup must be inside machine after insertion",
      "condition_expr": "(inside cup machine)",
      "severity": "hard",
      "eval_time": "post"
    }
  ]
}


⸻

✔ 2.2 Constraint Generation 伪代码

Algorithm GenerateConstraints(scene_graph, task_info):

    scene_text = scene_graph.to_text()

    prompt = BuildPrompt(scene_text, task_info)

    llm_output = LLMQuery(prompt)

    constraint_list = ParseConstraintJSON(llm_output)

    ast_constraints = CompileConstraintsToAST(constraint_list)

    return ast_constraints


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

✔ ValidateConstraint（核心）

Algorithm ValidateConstraint(constraint, world_state, evaluation_time, memory):

    if constraint.type == 'pre' and evaluation_time != 'pre':
        return SKIP

    if constraint.type == 'post' and evaluation_time != 'post':
        return SKIP

    if constraint.condition_ast == NULL:
        return UNCERTAIN  # 防止 LLM 错误导致漏判

    (value, atom_conf) = EvalPredicate(constraint.condition_ast, world_state, memory)

    confidence = Aggregate(atom_conf)

    if value == True and confidence > threshold:
        return SATISFIED

    if value == False and confidence > threshold:
        return VIOLATED

    return UNCERTAIN


⸻

#️⃣ 5. 整体流程（Complete Failure Detection Pipeline）

Algorithm CRAFT_Pipeline(video_stream, task_info):

    memory = EnvironmentMemory()
    constraints = GenerateConstraints(initial_scene_graph, task_info)
    
    prev_state = None

    for frame in video_stream:

        raw_state = Perception(frame)
        world_state = memory.update(raw_state)

        event = DetectCurrentEvent(world_state, action_log)

        if ShouldTriggerValidation(prev_state, world_state, event):

            for c in constraints.for_event(event) ∪ global_invariants:

                status = ValidateConstraint(c, world_state, eval_time_for(c), memory)

                Log(c.id, status)

                if status == VIOLATED:
                    return FAILURE_DETECTED(c)

        prev_state = world_state

    return SUCCESS


⸻

#️⃣ 6. 核心约束类型（Constraint Types）

类型	示例	说明
Precondition	machine must be open	动作前必须满足
Postcondition	cup inside machine	动作后必须满足
Invariant	kettle cannot teleport	始终适用
Causal Chain	fill → has_water → heat	跨动作因果依赖
Geometry Constraint	not intersect(cup, machine.wall)	真实几何检查
Occupancy Constraint	volume_free(machine)	容器不能被占满
Memory Constraint	must not disappear instantly	遮挡时不应判断为消失


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

例：柜门关闭却“放入成功”

LLM summary 模糊 → “cup near cabinet” → REFLECT误判成功
CRAFT++：

Pre: cabinet.door == 'open'
Post: inside(cup, cabinet)

可执行验证输出：

PreconditionViolation: cabinet not open


⸻

例：水壶没加水却加热

Pre(fill): kettle.position == faucet
Post(fill): has_water == True
Pre(heat): has_water == True

输出：

Violation: cannot heat kettle with no water


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
