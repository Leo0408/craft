# CRAFT++ 框架优化方案

本文档详细说明了基于 `demo1.ipynb` 分析的高优先级和中优先级优化方案。

---

## 一、高优先级优化

### 1.1 约束生成格式优化

#### 问题
- LLM 生成的是自然语言格式，缺少结构化 JSON
- 缺少 `id`, `severity`, `eval_time` 等关键字段
- LLM 未直接生成 `condition_expr`（可执行 AST）

#### 解决方案

**1. 改进 LLM Prompt**

更新 `reasoning/llm_prompter.py` 中的 `constraint-generator` prompt：

```python
'constraint-generator': {
    'template-system': 'You are a constraint generator for robot tasks. Generate structured logical constraints in JSON format with executable AST expressions.',
    'template-user': '''Task: {task}
Scene Graph: {scene_graph}
Task Goal: {goal}

Generate logical constraints in the following JSON format:
{
  "constraints": [
    {
      "id": "C1",
      "type": "pre",  // "pre" (precondition), "post" (postcondition), "invariant", or "goal"
      "description": "Machine must be open before inserting a cup",
      "condition_expr": "(eq machine.door 'open')",  // Executable AST/DSL expression in LISP format
      "severity": "hard",  // "hard" or "soft"
      "eval_time": "pre"  // "pre" (before action), "post" (after action), "now" (current state), or "final" (task completion)
    }
  ]
}

AST Expression Format:
- Location: (inside obj container), (on_top_of obj surface)
- State: (eq obj.state 'open'), (eq obj.state 'closed'), (eq obj.state 'empty')
- Combined: (and (inside mug sink) (not (inside mug coffee_machine)))
- Negation: (not (inside obj container))

Generate constraints covering:
1. Preconditions (must be satisfied before actions)
2. Postconditions (must be satisfied after actions)
3. Invariants (must always be satisfied)
4. Goal constraints (final success conditions)
5. Causal chains (e.g., fill → has_water → heat)

Return ONLY valid JSON, no additional text.'''
}
```

**2. 更新约束解析器**

更新 `reasoning/constraint_generator.py` 中的 `_parse_constraints` 方法：

- 优先解析 JSON 格式
- 如果 JSON 解析失败，回退到文本解析（向后兼容）
- 自动规范化约束类型（pre → precondition, post → postcondition）
- 为每个约束添加默认字段（id, severity, eval_time）

**3. 实现效果**

- ✅ LLM 直接生成结构化 JSON
- ✅ 包含完整的约束元数据（id, type, severity, eval_time）
- ✅ LLM 直接生成可执行的 `condition_expr`（AST 格式）
- ✅ 向后兼容旧的文本格式

---

### 1.2 约束编译格式优化

#### 问题
- 当前格式：`Mug is_inside Sink`（自然语言风格）
- 期望格式：`(inside mug sink)`（AST 格式）
- 无法直接执行

#### 解决方案

**1. 改进 `compile_constraint` 方法**

更新 `reasoning/constraint_generator.py` 中的 `compile_constraint` 方法：

- 如果约束已有 `condition_expr`，直接使用（LLM 已生成）
- 否则，从描述中解析并生成标准 AST 格式：
  - 位置关系：`(inside mug sink)`, `(on_top_of mug countertop)`
  - 状态检查：`(eq machine.state 'open')`, `(eq mug.state 'empty')`
  - 组合表达式：`(and (inside mug sink) (not (inside mug coffee_machine)))`

**2. AST 格式规范**

```
原子谓词:
  (inside obj container)
  (on_top_of obj surface)
  (eq obj.attr 'value')
  (empty obj)
  (open obj)
  (closed obj)

逻辑组合:
  (and expr1 expr2 ...)
  (or expr1 expr2 ...)
  (not expr)

示例:
  (inside mug sink)
  (eq coffee_machine.state 'open')
  (and (inside mug coffee_machine) (not (inside mug sink)))
```

**3. 实现效果**

- ✅ 生成标准 AST 格式
- ✅ 可直接执行，无需额外解析
- ✅ 支持复杂逻辑组合

---

### 1.3 时序验证优化

#### 问题
- 没有区分 pre/post 约束的评估时间
- 只在最终状态验证，没有在动作前后分别验证
- 无法检测动作相关的约束违反

#### 解决方案

**1. 创建约束评估器**

新建 `reasoning/constraint_evaluator.py`：

- `ConstraintEvaluator` 类：评估 AST 表达式
- 支持原子谓词（inside, on_top_of, eq, empty 等）
- 支持逻辑组合（and, or, not）
- 返回 `(is_satisfied, reason, confidence)` 三元组

**2. 添加时序验证逻辑**

在 Step 6 失败检测中：

```python
# 在动作前验证 precondition
for constraint in constraints:
    if constraint['type'] == 'precondition' and constraint['eval_time'] == 'pre':
        is_valid, reason, conf = evaluator.evaluate(
            constraint['condition_expr'],
            scene_graph_before_action
        )
        if not is_valid:
            return FAILURE_DETECTED(constraint, reason)

# 执行动作
action_result = execute_action(action)

# 在动作后验证 postcondition
for constraint in constraints:
    if constraint['type'] == 'postcondition' and constraint['eval_time'] == 'post':
        is_valid, reason, conf = evaluator.evaluate(
            constraint['condition_expr'],
            scene_graph_after_action
        )
        if not is_valid:
            return FAILURE_DETECTED(constraint, reason)
```

**3. 实现效果**

- ✅ 在动作前验证 precondition
- ✅ 在动作后验证 postcondition
- ✅ 持续验证 invariant
- ✅ 在任务完成时验证 goal

---

## 二、中优先级优化

### 2.1 场景图属性完善

#### 问题
- 缺少时间特征（last_seen_ts, velocity）
- 缺少几何属性（bbox, pose）
- 缺少置信度（confidence）

#### 解决方案

**1. 更新 Node 类**

更新 `core/scene_graph.py` 中的 `Node` 类：

```python
@dataclass
class Node:
    """Represents an object or entity in the scene"""
    name: str
    object_type: str
    state: Optional[str] = None
    position: Optional[Tuple[float, float, float]] = None
    attributes: Dict = None
    # Enhanced attributes for CRAFT++
    bbox: Optional[Dict] = None  # Bounding box: {"min": [x,y,z], "max": [x,y,z]}
    pose: Optional[Dict] = None  # Pose: {"position": [x,y,z], "rotation": [x,y,z]}
    confidence: float = 1.0  # Detection confidence (0.0-1.0)
    last_seen_ts: Optional[float] = None  # Timestamp when last seen
    velocity: Optional[Tuple[float, float, float]] = None  # Velocity vector
```

**2. 在场景图生成时填充属性**

在 `demo1.ipynb` 的 Step 3 中：

```python
node = Node(
    name=obj_name,
    object_type=obj_type,
    state=state,
    position=obj.get("position"),
    bbox=obj.get("axisAlignedBoundingBox"),  # 添加
    pose={"position": obj.get("position"), "rotation": obj.get("rotation")},  # 添加
    confidence=1.0,  # AI2THOR 中为 1.0
    last_seen_ts=time.time(),  # 添加
    velocity=None  # 可以计算
)
```

**3. 实现效果**

- ✅ 场景图包含完整的时间和几何属性
- ✅ 支持 Environment Memory 模块使用
- ✅ 为真实环境应用做好准备

---

### 2.2 因果链约束支持

#### 问题
- 缺少跨动作的因果依赖约束
- 无法检测"未加水却加热"这类因果违反

#### 解决方案

**1. 在 LLM Prompt 中添加因果链要求**

在 `constraint-generator` prompt 中：

```
Generate constraints covering:
...
5. Causal chains (e.g., fill → has_water → heat)
```

**2. 约束类型扩展**

添加 `causal_chain` 类型：

```python
{
  "id": "C5",
  "type": "causal_chain",
  "description": "Mug must be filled with water before heating",
  "condition_expr": "(and (eq mug.state 'filled') (eq mug.contents 'water'))",
  "severity": "hard",
  "eval_time": "pre",  // Before heat action
  "depends_on": ["C3"]  // Depends on fill action constraint
}
```

**3. 验证逻辑**

在验证时检查因果链：

```python
# 检查因果链依赖
for constraint in constraints:
    if constraint['type'] == 'causal_chain':
        # 检查依赖的约束是否满足
        for dep_id in constraint.get('depends_on', []):
            dep_constraint = find_constraint_by_id(dep_id, constraints)
            if dep_constraint:
                is_satisfied, _, _ = evaluator.evaluate(
                    dep_constraint['condition_expr'],
                    scene_graph
                )
                if not is_satisfied:
                    return FAILURE_DETECTED(constraint, f"Dependency {dep_id} not satisfied")
```

**4. 实现效果**

- ✅ 支持因果链约束
- ✅ 能够检测因果违反
- ✅ 提供详细的因果链分析

---

## 三、完整实现流程

### 3.1 更新后的工作流程

```
1. 数据生成 (AI2THOR)
   ↓
2. 场景图生成
   - 包含完整属性（bbox, pose, confidence, last_seen_ts, velocity）
   ↓
3. 约束生成 (LLM)
   - 生成结构化 JSON
   - 包含 condition_expr (AST 格式)
   - 包含 id, type, severity, eval_time
   ↓
4. 约束编译（可选，如果 LLM 已生成则跳过）
   - 确保所有约束都有有效的 condition_expr
   ↓
5. 时序验证
   - 动作前：验证 precondition
   - 动作后：验证 postcondition
   - 持续：验证 invariant
   - 最终：验证 goal
   ↓
6. 失败检测
   - 使用 ConstraintEvaluator 评估 AST
   - 返回详细的违反原因
   ↓
7. 渐进式解释
   - 包含因果链分析
```

### 3.2 关键代码文件

1. **`reasoning/llm_prompter.py`**
   - 更新 `constraint-generator` prompt

2. **`reasoning/constraint_generator.py`**
   - 更新 `_parse_constraints` 方法（JSON 解析）
   - 改进 `compile_constraint` 方法（AST 格式）

3. **`reasoning/constraint_evaluator.py`**（新建）
   - `ConstraintEvaluator` 类
   - AST 表达式评估

4. **`core/scene_graph.py`**
   - 更新 `Node` 类（添加属性）

5. **`demo1.ipynb`**
   - Step 3: 场景图生成时填充属性
   - Step 6: 添加时序验证逻辑

---

## 四、预期效果

### 4.1 约束质量提升

**之前：**
```
约束: "Mug must be inside the Sink"
编译: "Mug is_inside Sink"  (无法执行)
```

**之后：**
```json
{
  "id": "C1",
  "type": "precondition",
  "description": "Mug must be inside the Sink",
  "condition_expr": "(inside mug sink)",
  "severity": "hard",
  "eval_time": "pre"
}
```

### 4.2 验证准确性提升

**之前：**
- 只在最终状态验证
- 无法检测动作相关的违反

**之后：**
- 动作前验证 precondition
- 动作后验证 postcondition
- 能够准确检测动作相关的违反

### 4.3 场景图信息完整性

**之前：**
- 只有 name, type, state, position

**之后：**
- 包含 bbox, pose, confidence, last_seen_ts, velocity
- 支持 Environment Memory 模块

---

## 五、实施步骤

1. ✅ 更新 LLM prompt（已完成）
2. ✅ 更新约束解析器（已完成）
3. ✅ 改进约束编译（已完成）
4. ✅ 创建约束评估器（已完成）
5. ✅ 更新场景图节点（已完成）
6. ⏳ 更新 demo1.ipynb 使用新功能
7. ⏳ 测试完整流程
8. ⏳ 更新 Method.md

---

## 六、向后兼容性

所有优化都保持向后兼容：

1. **约束解析**：如果 JSON 解析失败，回退到文本解析
2. **约束编译**：如果已有 condition_expr，直接使用
3. **场景图节点**：新属性都是可选的，旧代码仍可工作
4. **时序验证**：如果 eval_time 未指定，使用默认值

---

## 七、总结

通过实施这些优化，CRAFT++ 框架将能够：

- ✅ 生成更高质量的约束（结构化 JSON + AST）
- ✅ 更准确地检测失败（时序验证）
- ✅ 提供更完整的场景信息（增强属性）
- ✅ 支持因果链分析（因果约束）

这些优化使 CRAFT++ 更接近 Method.md 中描述的理想框架。

