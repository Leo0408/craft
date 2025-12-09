# CRAFT++ 框架流程分析与优化建议

## 一、流程概述

根据 `demo1.ipynb` 的执行流程，整个 CRAFT++ 框架包含以下步骤：

1. **环境设置** - 模块导入和 LLM API 配置
2. **Step 1: 数据生成** - AI2THOR 模拟环境中的任务执行
3. **Step 2: 视频显示** - 提取帧并生成标注视频
4. **Step 3: 场景图生成** - 从 AI2THOR events 构建场景图
5. **Step 4: 约束生成** - LLM 生成逻辑约束
6. **Step 5: 约束代码生成** - 编译约束为可执行 AST/DSL
7. **Step 6: 失败检测** - 使用可执行逻辑验证约束
8. **Step 7: 渐进式解释** - 生成详细的失败分析

---

## 二、各步骤详细分析与 CRAFT 思想对照

### Step 1: 数据生成 (AI2THOR)

**当前实现：**
- ✅ 使用真实的 AI2THOR Controller 执行任务
- ✅ 记录每个动作的执行结果（SUCCESS/FAILED）
- ✅ 保存 events 和 action_results

**输出示例：**
```
✅ Executed 9 actions
   Successful: 9
   Failed: 0
   Errors: 0
```

**CRAFT 思想符合度：** ✅ **符合**
- 使用真实模拟环境生成数据
- 记录完整的执行轨迹

**优化建议：**
- 当前所有动作都成功，建议添加**失败注入机制**来测试失败检测能力
- 可以模拟常见的失败场景（如：咖啡机未打开就放入杯子）

---

### Step 2: 视频显示

**当前实现：**
- ✅ 从 AI2THOR events 提取 RGB 帧
- ✅ 添加动作标注（动作名称、状态）
- ✅ 生成视频文件

**CRAFT 思想符合度：** ✅ **符合**
- 可视化执行过程，便于调试和分析

**优化建议：**
- 当前视频生成是简单的帧序列，可以添加**场景图可视化**到视频中
- 可以在关键帧标注约束验证结果

---

### Step 3: 场景图生成

**当前实现：**
- ✅ 从 AI2THOR metadata 提取对象（节点）
- ✅ 提取对象状态（open/closed, on/off, filled/empty）
- ✅ 推断空间关系（inside, on_top_of）
- ✅ 为每个 event 生成一个场景图

**输出示例：**
```
✅ Generated 9 scene graphs
📊 Scene Graph Summary:
   Initial State:
   - Nodes: 6
   - Edges: 6
   Final State:
   - Nodes: 6
   - Edges: 5
```

**CRAFT 思想符合度：** ⚠️ **部分符合**

**问题分析：**

1. **缺少时间特征** (Method.md Section 1)
   - ❌ 没有 `last_seen_ts`（最后看到的时间戳）
   - ❌ 没有 `velocity`（速度信息）
   - ❌ 没有 `confidence`（置信度）

2. **缺少几何属性** (Method.md Section 1)
   - ❌ 没有 `bbox`（边界框）
   - ❌ 没有 `pose`（位姿）

3. **关系推断过于简单**
   - ⚠️ 只检查了 `parentReceptacles` 和位置关系
   - ⚠️ 没有考虑遮挡、可见性等复杂情况

**优化建议：**

```python
# 应该添加的属性
node = SceneNode(
    name=obj_name,
    type=obj_type,
    state=state,
    bbox=obj.get("axisAlignedBoundingBox"),  # 添加
    pose=obj.get("position"),  # 添加
    confidence=1.0,  # AI2THOR 中为 1.0，真实环境需要计算
    last_seen_ts=current_time(),  # 添加
    velocity=None  # 可以计算
)
```

---

### Step 4: 约束生成

**当前实现：**
- ✅ 使用 LLM 从初始场景图生成约束
- ✅ 约束包含 type（precondition/postcondition）
- ✅ 约束包含 description

**输出示例：**
```
✅ Generated 7 constraints
📋 Generated Constraints:
   1. 🔒 [precondition]: Mug must be inside the Sink....
   2. 🔒 [precondition]: CoffeeMachine must be on top of the CounterTop....
   3. 🔒 [precondition]: Mug must be on top of the CounterTop....
   4. 🔒 [precondition]: The Faucet must be on top of the CounterTop....
   5. 🔒 [precondition]: The Sink must be on top of the CounterTop....
   6. 🔒 [precondition]: The Mug must be clean....
   7. 🔒 [precondition]: The Mug must be filled with coffee....
```

**CRAFT 思想符合度：** ⚠️ **部分符合**

**问题分析：**

1. **缺少结构化 JSON 格式** (Method.md Section 2.1)
   - ❌ 约束没有 `id`（唯一标识符）
   - ❌ 约束没有 `condition_expr`（可执行表达式）
   - ❌ 约束没有 `severity`（严重程度）
   - ❌ 约束没有 `eval_time`（评估时间）

2. **约束类型不完整**
   - ⚠️ 只有 precondition，缺少 postcondition、invariant、goal
   - ⚠️ 缺少因果链约束（Causal Chain）

3. **约束质量**
   - ⚠️ 约束 1-5 都是位置约束，可能过于冗余
   - ⚠️ 约束 6-7 是状态约束，但缺少动作相关的约束

**期望格式（Method.md Section 2.1）：**
```json
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
```

**优化建议：**

1. **改进 LLM Prompt**，要求生成结构化 JSON
2. **添加约束类型验证**，确保包含 pre/post/invariant/goal
3. **添加因果链约束**，例如：
   - `fill → has_water → heat`（加水后才能加热）

---

### Step 5: 约束代码生成

**当前实现：**
- ✅ 使用 `compile_constraint` 方法将约束描述编译为 AST/DSL
- ✅ 生成可执行表达式

**CRAFT 思想符合度：** ✅ **符合**
- 将自然语言约束转换为可执行代码，符合 CRAFT 核心思想

**优化建议：**
- 当前编译是基于模式匹配，可能不够准确
- 建议让 LLM 在生成约束时直接生成 `condition_expr`，而不是后续编译

---

### Step 6: 失败检测

**当前实现：**
- ✅ 对最终场景图评估编译后的约束
- ✅ 使用简单的规则引擎检查约束是否满足

**CRAFT 思想符合度：** ⚠️ **部分符合**

**问题分析：**

1. **缺少时序验证** (Method.md Section 4)
   - ❌ 没有区分 pre/post 约束的评估时间
   - ❌ 没有在动作前后分别验证约束

2. **缺少 Environment Memory 集成**
   - ❌ 没有使用 Memory 的平滑状态
   - ❌ 没有处理遮挡情况

3. **验证逻辑过于简单**
   - ⚠️ 只检查最终状态，没有检查动作序列中的中间状态
   - ⚠️ 没有考虑置信度和不确定性

**期望流程（Method.md Section 5）：**
```
for frame in video_stream:
    raw_state = Perception(frame)
    world_state = memory.update(raw_state)  # 使用 Memory
    
    event = DetectCurrentEvent(world_state, action_log)
    
    if ShouldTriggerValidation(prev_state, world_state, event):
        for c in constraints.for_event(event):
            status = ValidateConstraint(
                c, 
                world_state, 
                eval_time_for(c),  # pre/post
                memory
            )
```

**优化建议：**

1. **添加时序验证**
   - 在动作前验证 precondition
   - 在动作后验证 postcondition
   - 持续验证 invariant

2. **集成 Environment Memory**
   - 使用 Memory 的平滑状态进行验证
   - 处理遮挡和不确定性

3. **添加置信度评估**
   - 返回约束满足的置信度
   - 区分确定违反和不确定情况

---

### Step 7: 渐进式解释

**当前实现：**
- ⚠️ 可能未完全实现或输出不完整

**CRAFT 思想符合度：** ❓ **需要确认**

**期望功能（Method.md Section 5）：**
- 生成根因分析
- 创建因果链
- 提供可操作的修正建议

**优化建议：**
- 确保 FailureAnalyzer 能够生成详细的失败分析
- 包含因果链和修正建议

---

## 三、整体流程与 CRAFT 思想对照

### ✅ 符合 CRAFT 思想的部分

1. **可执行约束** - Step 5 将约束编译为可执行代码 ✅
2. **场景图构建** - Step 3 构建了结构化场景表示 ✅
3. **逻辑验证** - Step 6 使用逻辑引擎验证约束 ✅
4. **真实环境数据** - Step 1 使用 AI2THOR 生成数据 ✅

### ⚠️ 不符合或需要改进的部分

1. **Environment Memory** - 完全缺失
   - ❌ 没有 Kalman/Bayesian filter
   - ❌ 没有 last_seen 跟踪
   - ❌ 没有遮挡处理
   - **影响**：在真实环境中无法处理感知噪声和遮挡

2. **约束生成格式** - 不符合 Method.md 要求
   - ❌ 缺少结构化 JSON 格式
   - ❌ 缺少 condition_expr（LLM 应直接生成）
   - ❌ 缺少 severity 和 eval_time
   - **影响**：约束质量不高，难以精确验证

3. **时序验证** - 验证逻辑不完整
   - ❌ 没有区分 pre/post 的评估时间
   - ❌ 没有在动作前后分别验证
   - **影响**：无法准确检测动作前后的约束违反

4. **场景图属性** - 缺少关键属性
   - ❌ 缺少时间特征（last_seen_ts, velocity）
   - ❌ 缺少几何属性（bbox, pose）
   - **影响**：场景图信息不完整，影响约束验证准确性

5. **因果链** - 缺少因果链约束
   - ❌ 没有跨动作的因果依赖约束
   - **影响**：无法检测"未加水却加热"这类因果违反

---

## 四、优化优先级建议

### 🔴 高优先级（核心功能）

1. **改进约束生成格式**
   - 让 LLM 直接生成结构化 JSON，包含 `condition_expr`
   - 添加 `id`, `severity`, `eval_time` 字段
   - **预期效果**：约束质量显著提升，验证更准确

2. **添加时序验证**
   - 在动作前验证 precondition
   - 在动作后验证 postcondition
   - **预期效果**：能够准确检测动作相关的约束违反

3. **完善场景图属性**
   - 添加 `last_seen_ts`, `confidence` 等时间特征
   - 添加 `bbox`, `pose` 等几何属性
   - **预期效果**：场景图信息更完整，支持更精确的约束验证

### 🟡 中优先级（增强功能）

4. **添加因果链约束**
   - 生成跨动作的因果依赖约束
   - 例如：`fill → has_water → heat`
   - **预期效果**：能够检测因果链违反

5. **改进约束类型分布**
   - 确保包含 pre/post/invariant/goal 所有类型
   - 减少冗余约束
   - **预期效果**：约束更全面，覆盖更多失败场景

### 🟢 低优先级（扩展功能）

6. **集成 Environment Memory**
   - 在模拟环境中可以简化，但在真实环境中必需
   - 添加 Kalman filter、遮挡处理等
   - **预期效果**：在真实环境中处理感知噪声和遮挡

7. **添加失败注入机制**
   - 模拟常见失败场景
   - 测试失败检测能力
   - **预期效果**：验证框架的失败检测能力

---

## 五、具体优化代码示例

### 1. 改进约束生成 Prompt

```python
prompt = f"""
Generate constraints in the following JSON format:
{{
  "constraints": [
    {{
      "id": "C1",
      "type": "pre",  // pre/post/invariant/goal
      "description": "...",
      "condition_expr": "(eq machine.door 'open')",  // Executable AST/DSL
      "severity": "hard",  // hard/soft
      "eval_time": "pre"  // pre/post/now/final
    }}
  ]
}}

Scene Graph: {scene_text}
Task: {task_name}
Goal: {goal_text}
"""
```

### 2. 添加时序验证

```python
# 在动作前验证 precondition
for constraint in constraints:
    if constraint['type'] == 'precondition':
        status = validate_constraint(
            constraint, 
            scene_graph_before_action, 
            eval_time='pre',
            memory=memory
        )
        if status == VIOLATED:
            return FAILURE_DETECTED(constraint)

# 执行动作
action_result = execute_action(action)

# 在动作后验证 postcondition
for constraint in constraints:
    if constraint['type'] == 'postcondition':
        status = validate_constraint(
            constraint, 
            scene_graph_after_action, 
            eval_time='post',
            memory=memory
        )
        if status == VIOLATED:
            return FAILURE_DETECTED(constraint)
```

### 3. 完善场景图节点

```python
node = SceneNode(
    name=obj_name,
    type=obj_type,
    state=state,
    bbox=obj.get("axisAlignedBoundingBox"),
    pose=obj.get("position"),
    confidence=1.0,  # AI2THOR 中为 1.0
    last_seen_ts=current_time(),
    velocity=calculate_velocity(obj, prev_obj) if prev_obj else None
)
```

---

## 六、总结

### 当前状态
- ✅ **基础框架完整**：所有核心步骤都已实现
- ✅ **可执行约束**：符合 CRAFT 核心思想
- ⚠️ **细节不完善**：多个关键功能需要优化

### 关键改进点
1. **约束生成格式** - 需要结构化 JSON 和 condition_expr
2. **时序验证** - 需要区分 pre/post 评估时间
3. **场景图属性** - 需要添加时间和几何属性
4. **因果链** - 需要添加跨动作的因果约束

### 预期效果
完成这些优化后，CRAFT++ 框架将能够：
- ✅ 更准确地检测失败
- ✅ 提供更详细的失败分析
- ✅ 支持真实环境的感知噪声和遮挡
- ✅ 检测因果链违反

