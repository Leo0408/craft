# CRAFT++ 完整流程与优化方案说明文档

本文档详细说明 CRAFT++ 框架的完整流程以及基于 `method_add.md` 的优化方案实现。

---

## 一、CRAFT++ 框架概述

CRAFT++ 是一个基于可执行逻辑约束与环境记忆的机器人失败检测框架，旨在解决零样本 LLM 系统在真实场景失败检测中的三个核心问题：

1. **感知噪声导致的错误判断**（遮挡、不稳定检测 → 假失败/假成功）
2. **缺乏物理可验证性**（LLM"看图编故事" → 幻觉式成功判断）
3. **缺乏因果链/动作前后逻辑**（例如：未加水却被判定能加热水壶）

### 核心思想

让 LLM 生成可执行约束（Executable Constraints），并通过逻辑引擎与时序记忆进行验证，从而实现与视觉无关、与场景无关的确定性失败检测。

### 框架结构

```
(Perception + Memory) → Scene Graph → Constraint Compiler → Constraint Executor
```

---

## 二、完整流程（基于 Method.md）

### 2.1 数据生成（AI2THOR）

**目标**：在模拟环境中生成机器人执行数据

**实现**：
- 使用 AI2THOR Controller 执行任务动作序列
- 记录每个动作的事件（event）和结果（action_result）
- 保存 RGB 帧用于可视化

**输出**：
- `events_craft`: 事件列表
- `action_results`: 动作结果列表
- `task_info_craft`: 任务信息

### 2.2 场景图生成（Scene Graph Construction）

**目标**：构建结构化场景表示

**实现**（Method.md Section 1）：
- 从 AI2THOR 事件中提取对象和状态
- 推断空间关系（inside, on_top_of）
- 创建 SceneGraph，包含：
  - **节点（Nodes）**：对象及其属性
  - **边（Edges）**：对象间关系

**节点属性**（优化后包含完整属性）：
- `name`: 对象名称
- `object_type`: 对象类型
- `state`: 对象状态（open/closed/filled/empty）
- `position`: 3D 位置
- `bbox`: 边界框（优化新增）
- `pose`: 位姿（优化新增）
- `confidence`: 检测置信度（优化新增）
- `last_seen_ts`: 最后看到的时间戳（优化新增）
- `velocity`: 速度向量（优化新增）

**输出**：
- `scene_graphs_craft`: 场景图列表（每个事件一个场景图）

### 2.3 约束生成（Constraint Generation）

**目标**：使用 LLM 生成结构化约束

**实现**（Method.md Section 2）：
- 将场景图转换为文本描述
- 使用 LLM 生成结构化 JSON 约束
- 每个约束包含：
  - `id`: 约束 ID
  - `type`: 约束类型（pre/post/invariant/goal）
  - `description`: 自然语言描述
  - `condition_expr`: 可执行 AST/DSL 表达式（优化重点）
  - `severity`: 严重程度（hard/soft）
  - `eval_time`: 评估时间（pre/post/now/final）

**LLM Prompt 优化**（基于 method_add.md）：
- 要求生成结构化 JSON 格式
- 直接生成可执行的 `condition_expr`（LISP 格式 AST）
- 支持复杂逻辑组合：`(and (inside mug sink) (not (inside mug coffee_machine)))`

**输出**：
- `constraints_craft`: 约束列表

### 2.4 约束编译（Constraint Compilation）

**目标**：确保所有约束都有可执行的 AST 表达式

**实现**（Method.md Section 2.2）：
- 如果 LLM 已生成 `condition_expr`，直接使用
- 否则，从 `description` 编译生成 AST
- 验证 AST 语法正确性

**输出**：
- `compiled_constraints`: 编译后的约束列表（每个约束都有有效的 `condition_expr`）

### 2.5 时序验证（Timing Validation）

**目标**：在正确的时机验证不同类型的约束

**实现**（优化新增，基于 method_add.md）：
- **Precondition**：在动作前验证（`evaluation_time='pre'`）
- **Postcondition**：在动作后验证（`evaluation_time='post'`）
- **Invariant**：持续验证（每个状态）
- **Goal**：在任务完成时验证（`evaluation_time='final'`）

**验证逻辑**：
```python
if constraint_type == 'precondition' and evaluation_time not in ['pre', 'now']:
    return SKIP
if constraint_type == 'postcondition' and evaluation_time not in ['post', 'final']:
    return SKIP
```

### 2.6 失败检测（Failure Detection）

**目标**：使用可执行逻辑验证约束

**实现**（Method.md Section 4）：
- 使用 `ConstraintEvaluator` 评估 AST 表达式
- 返回详细结果：
  - `status`: SATISFIED / VIOLATED / UNCERTAIN / SKIP
  - `confidence`: 置信度（0.0-1.0）
  - `reason`: 解释
  - `atom_traces`: 原子级追踪（优化新增）

**Atom-level Trace**（优化新增，基于 method_add.md）：
- 每个原子谓词（如 `inside(mug, sink)`）返回：
  - `value`: True/False
  - `confidence`: 置信度
  - `source`: 来源（edge_relation, geometry_iou, state_check）
  - `reason`: 详细原因

**置信度聚合**：
- AND: 取最小值（保守策略）
- OR: 取最大值
- 如果置信度 < `min_confidence_threshold`，返回 UNCERTAIN

**输出**：
- `validation_results`: 验证结果列表
- `violated_constraints`: 违反的约束列表

### 2.7 渐进式解释（Progressive Explanation）

**目标**：生成详细的失败分析

**实现**（Method.md Section 5）：
- 如果检测到违反，生成可追溯解释
- 包含：
  - 哪个 pre/post 失败
  - 哪个原子谓词导致失败
  - 因果链分析（如果适用）
  - 修复建议

---

## 三、优化方案（基于 method_add.md）

### 3.1 高优先级优化

#### 3.1.1 约束生成格式优化 ✅

**问题**：
- LLM 生成的是自然语言格式，缺少结构化 JSON
- 缺少 `condition_expr`（可执行 AST）

**解决方案**：
- 更新 `reasoning/llm_prompter.py` 中的 `constraint-generator` prompt
- 要求生成结构化 JSON，包含 `condition_expr`
- 更新 `reasoning/constraint_generator.py` 的 `_parse_constraints` 方法支持 JSON 解析

**实现位置**：
- `reasoning/llm_prompter.py`: 已更新 prompt
- `reasoning/constraint_generator.py`: 已支持 JSON 解析

#### 3.1.2 约束编译格式优化 ✅

**问题**：
- 当前格式 `Mug is_inside Sink` 无法直接执行

**解决方案**：
- 生成标准 AST 格式：`(inside mug sink)`
- 支持复杂逻辑组合：`(and (inside mug sink) (not (inside mug coffee_machine)))`

**实现位置**：
- `reasoning/constraint_generator.py`: `compile_constraint` 方法已改进

#### 3.1.3 时序验证优化 ✅

**问题**：
- 没有区分 pre/post 约束的评估时间
- 只在最终状态验证

**解决方案**：
- 创建 `ConstraintEvaluator.validate_constraint` 方法
- 在动作前验证 precondition
- 在动作后验证 postcondition
- 持续验证 invariant
- 在任务完成时验证 goal

**实现位置**：
- `reasoning/constraint_evaluator.py`: 已实现 `validate_constraint` 方法

#### 3.1.4 Atom-level Trace ✅

**问题**：
- 验证器没有打印 atom-level trace
- 无法看到执行器是如何判断的

**解决方案**：
- 增强 `ConstraintEvaluator`，添加 `AtomTrace` 类
- 每个原子谓词返回详细追踪信息
- 包含 value, confidence, source, reason

**实现位置**：
- `reasoning/constraint_evaluator.py`: 已实现 `AtomTrace` 和 atom-level trace

### 3.2 中优先级优化

#### 3.2.1 场景图属性完善 ✅

**问题**：
- 缺少时间特征和几何属性

**解决方案**：
- 更新 `Node` 类添加：`bbox`, `pose`, `confidence`, `last_seen_ts`, `velocity`
- 在场景图生成时填充这些属性

**实现位置**：
- `core/scene_graph.py`: `Node` 类已包含所有属性
- `demo1.ipynb` Step 3: 需要在生成时填充属性（待更新）

#### 3.2.2 因果链约束支持

**问题**：
- 缺少跨动作的因果依赖约束

**解决方案**：
- 在 LLM Prompt 中添加因果链要求
- 添加 `causal_chain` 约束类型
- 验证时检查因果链依赖

**实现位置**：
- `reasoning/llm_prompter.py`: prompt 已包含因果链要求
- `reasoning/constraint_generator.py`: 支持因果链类型（待完善）

---

## 四、代码更新说明

### 4.1 已更新的文件

1. **`reasoning/constraint_evaluator.py`**
   - 添加 `AtomTrace` 类
   - 增强 `evaluate` 方法，支持 atom-level trace
   - 添加 `validate_constraint` 方法，支持时序验证
   - 所有评估方法返回 `(value, reason, confidence, atom_traces)`

2. **`reasoning/constraint_generator.py`**
   - `_parse_constraints` 方法已支持 JSON 解析
   - `compile_constraint` 方法已改进，生成标准 AST 格式

3. **`reasoning/llm_prompter.py`**
   - `constraint-generator` prompt 已更新，要求生成结构化 JSON

4. **`core/scene_graph.py`**
   - `Node` 类已包含完整属性（bbox, pose, confidence, last_seen_ts, velocity）

### 4.2 待更新的文件

1. **`demo1.ipynb`**
   - Step 3: 场景图生成时填充完整属性
   - Step 4: 显示生成的 JSON 约束和 AST
   - Step 6: 使用 `ConstraintEvaluator.validate_constraint` 进行时序验证
   - Step 6: 显示 atom-level trace

---

## 五、使用示例

### 5.1 约束生成示例

```python
from craft.reasoning import ConstraintGenerator, LLMPrompter

llm_prompter = LLMPrompter(gpt_version="gpt-3.5-turbo", api_key=API_KEY, base_url=BASE_URL)
generator = ConstraintGenerator(llm_prompter)

constraints = generator.generate_constraints(
    scene_graph=scene_graph,
    task_info=task_info,
    goal="a clean mug filled with coffee is on top of the countertop"
)

# 约束格式：
# {
#   "id": "C1",
#   "type": "pre",
#   "description": "Machine must be open before inserting a cup",
#   "condition_expr": "(eq machine.door 'open')",
#   "severity": "hard",
#   "eval_time": "pre"
# }
```

### 5.2 约束验证示例

```python
from craft.reasoning import ConstraintEvaluator

evaluator = ConstraintEvaluator(min_confidence_threshold=0.7)

# 验证约束（动作前）
result = evaluator.validate_constraint(
    constraint=constraint,
    scene_graph=scene_graph,
    evaluation_time="pre"  # 动作前验证
)

# 结果格式：
# {
#   "id": "C1",
#   "status": "VIOLATED",  # SATISFIED / VIOLATED / UNCERTAIN / SKIP
#   "confidence": 0.95,
#   "reason": "Machine door is 'closed' but required 'open'",
#   "atom_traces": [
#     AtomTrace(
#       atom_expr="(eq machine.door 'open')",
#       value=False,
#       confidence=0.95,
#       source="state_equality",
#       reason="Machine.door == 'closed' vs required 'open'"
#     )
#   ]
# }
```

### 5.3 Atom-level Trace 示例

```python
# 打印 atom-level trace
for trace in result['atom_traces']:
    print(f"[{trace.atom_expr}]: {trace.value} (conf={trace.confidence:.2f}, source={trace.source})")
    print(f"  Reason: {trace.reason}")

# 输出示例：
# [(eq machine.door 'open')]: False (conf=0.95, source=state_equality)
#   Reason: Machine.door == 'closed' vs required 'open'
```

---

## 六、完整工作流程

```
1. 数据生成 (AI2THOR)
   ↓
2. 场景图生成（包含完整属性：bbox, pose, confidence, last_seen_ts, velocity）
   ↓
3. 约束生成 (LLM) → 结构化 JSON + condition_expr (AST)
   ↓
4. 约束编译（可选，如果 LLM 已生成则跳过）
   ↓
5. 时序验证
   - 动作前：验证 precondition (evaluation_time='pre')
   - 动作后：验证 postcondition (evaluation_time='post')
   - 持续：验证 invariant (evaluation_time='now')
   - 最终：验证 goal (evaluation_time='final')
   ↓
6. 失败检测（使用 ConstraintEvaluator）
   - 评估 AST 表达式
   - 返回 atom-level trace
   - 聚合置信度
   ↓
7. 渐进式解释（包含因果链分析）
```

---

## 七、预期效果

### 7.1 约束质量提升
- ✅ 结构化 JSON 格式
- ✅ 可执行 AST 表达式
- ✅ 完整的元数据（id, type, severity, eval_time）

### 7.2 验证准确性提升
- ✅ 时序验证能够准确检测动作相关的违反
- ✅ Atom-level trace 提供可解释的判定过程
- ✅ 置信度聚合支持不确定性处理

### 7.3 场景图信息完整性
- ✅ 包含时间和几何属性
- ✅ 支持置信度跟踪

### 7.4 可解释性提升
- ✅ Atom-level trace 显示每个原子谓词的判定过程
- ✅ 详细的违反原因和修复建议

---

## 八、后续优化建议

### 8.1 环境记忆模块（Environment Memory）

**当前状态**：在 AI2THOR 模拟环境中简化（因为确定性状态）

**未来增强**：
- 实现 Kalman/Bayesian filter 进行位置平滑
- 处理遮挡和传感器噪声
- 状态置信度衰减模型

### 8.2 几何检查模块

**当前状态**：主要基于 edge 关系

**未来增强**：
- 实现 `geometry.is_inside(obj, container)` 使用 bbox/pose/volume
- 实现 `iou(bbox1, bbox2)` 用于几何验证
- 在边缘情况下返回 UNCERTAIN

### 8.3 因果链验证

**当前状态**：LLM prompt 已包含因果链要求

**未来增强**：
- 实现因果链依赖检查
- 支持跨动作的因果约束验证

---

## 九、总结

CRAFT++ 框架通过以下优化实现了确定性、可解释的失败检测：

1. **可执行约束**：LLM 生成结构化 JSON 和可执行 AST
2. **时序验证**：在正确的时机验证不同类型的约束
3. **Atom-level Trace**：提供详细的判定过程追踪
4. **置信度聚合**：支持不确定性处理
5. **完整属性**：场景图包含所有必要的几何和时间信息

这些优化确保了 CRAFT++ 能够：
- 准确检测动作执行时的违反
- 提供可解释的判定过程
- 处理感知噪声和不确定性
- 支持因果链分析

---

## 十、参考文档

- `Method.md`: CRAFT++ 框架整体设计
- `method_add.md`: 优化方案详细说明
- `demo1.ipynb`: 完整工作流程演示

