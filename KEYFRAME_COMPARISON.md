# 关键帧方法对比：CRAFT vs REFLECT

## 快速回答

### Q1: CRAFT 现在是否采用关键帧？

**答案：部分采用，但不是最优的关键帧方法**

- ✅ 为**每个动作事件**生成一个场景图（9 个动作 → 9 个场景图）
- ⚠️ **不是真正的关键帧选择**，而是"每个动作一帧"
- ❌ 没有根据状态变化选择关键帧

### Q2: 与 REFLECT 的关键帧方法有什么不同？

**主要区别**：

| 方面 | CRAFT (当前) | REFLECT |
|------|-------------|---------|
| **关键帧定义** | 每个动作一帧 | 子目标达成时 |
| **选择标准** | 固定（每个动作） | 动态（状态变化） |
| **验证方式** | 可执行约束（但时机不对） | LLM 子目标验证 |
| **验证时机** | 最终状态 | 每个子目标后 |
| **优势** | 可执行、可复现 | 细粒度检测 |

---

## 详细分析

### CRAFT 当前实现

#### 场景图生成

```python
# demo1.ipynb Step 3
for event_idx, (event, action_result) in enumerate(zip(events_craft, action_results)):
    # 为每个动作生成场景图
    scene_graphs_craft.append(sg)
```

**特点**：
- 9 个动作 → 9 个场景图
- 固定模式：每个动作一帧
- 不区分关键动作和非关键动作

**问题**：
- `navigate_to_obj` 可能不改变场景图，但仍然生成
- 没有选择真正关键的状态变化点

#### 约束验证

```python
# demo1.ipynb Step 6
final_scene_graph = scene_graphs_craft[-1]
# 只在最终状态验证
for comp_const in compiled_constraints:
    is_valid, reason = evaluate_constraint(condition_expr, final_scene_graph)
```

**特点**：
- ❌ 只在最终状态验证
- ❌ 没有在动作执行时验证
- ❌ 没有使用关键帧触发验证

---

### REFLECT 的关键帧方法

#### 子目标作为关键帧

REFLECT 使用**子目标（subgoal）**作为关键帧：

```python
# REFLECT 的关键帧选择
subgoals = [
    {"goal": "pick up purple cup", "frame_idx": 50},
    {"goal": "put cup in coffee machine", "frame_idx": 100},
    {"goal": "coffee machine is open", "frame_idx": 80}
]

# 在子目标时间点验证
for subgoal in subgoals:
    frame_idx = subgoal['frame_idx']
    scene_graph = scene_graphs[frame_idx]
    is_success = verify_subgoal(subgoal['goal'], scene_graph)
```

**特点**：
- ✅ 关键帧 = 子目标应该达成的时间点
- ✅ 不是每帧检查，而是**在关键状态变化时检查**
- ✅ 基于 LLM 验证子目标是否达成

#### 验证流程示例

```
Action 1: navigate_to_obj(mug)
  → 子目标：到达 mug 附近
  → 关键帧：Action 1 后（frame 50）
  → 验证：mug 是否可见/可达

Action 2: pick_up(mug)
  → 子目标：mug 被拿起
  → 关键帧：Action 2 后（frame 100）
  → 验证：mug 是否在机器人手中

Action 3: put_in(mug, coffee_machine)
  → 子目标：mug 在咖啡机内
  → 关键帧：Action 3 后（frame 150）
  → 验证：mug 是否在咖啡机内，咖啡机是否为空（precondition）
```

---

## 核心区别

### 1. 关键帧选择策略

**CRAFT (当前)**：
- 固定：每个动作一帧
- 不区分关键动作和非关键动作
- 可能包含冗余场景图

**REFLECT**：
- 动态：子目标达成时
- 只选择关键状态变化点
- 更高效，减少冗余

### 2. 验证时机

**CRAFT (当前)**：
- ❌ 只在最终状态验证
- ❌ 无法检测动作执行时的违反

**REFLECT**：
- ✅ 每个子目标后验证
- ✅ 能够及时检测失败

### 3. 验证方式

**CRAFT**：
- ✅ 可执行约束（AST 表达式）
- ✅ 可复现、确定性
- ❌ 但验证时机不对

**REFLECT**：
- ✅ LLM 子目标验证
- ✅ 细粒度检测
- ❌ 依赖 LLM，可能不稳定

---

## CRAFT 的问题

### 问题 1: 不是真正的关键帧

**当前**：
- 每个动作一帧（9 个动作 → 9 个场景图）
- 不区分关键动作

**应该**：
- 只在状态显著变化时生成场景图
- 例如：`navigate_to_obj` 可能不需要生成场景图

### 问题 2: 缺少 ShouldTriggerValidation

**Method.md 中的理想实现**：
```python
if ShouldTriggerValidation(prev_state, world_state, event):
    # 验证约束
```

**当前实现**：
- ❌ 没有 `ShouldTriggerValidation` 函数
- ❌ 固定验证，不根据状态变化触发

### 问题 3: 验证时机不对

**当前**：
- 只在最终状态验证
- 无法检测动作执行时的违反

**应该**：
- 在动作前验证 precondition
- 在动作后验证 postcondition

---

## 优化方案

### 方案 1: 智能关键帧选择

```python
def should_generate_scene_graph(prev_state, current_state, action):
    """决定是否生成场景图（关键帧选择）"""
    # 1. 关键动作（必须生成）
    key_actions = ['put_in', 'put_on', 'pick_up', 'toggle_on', 'toggle_off']
    if action in key_actions:
        return True
    
    # 2. 状态显著变化
    if state_changed_significantly(prev_state, current_state):
        return True
    
    # 3. 对象关系变化
    if relations_changed(prev_state, current_state):
        return True
    
    # 4. 导航动作不生成（状态通常不变）
    if action == 'navigate_to_obj':
        return False
    
    return False
```

### 方案 2: 动作相关的关键帧（已实施）

在 `TIMING_VALIDATION_UPDATE.md` 中：

```python
# 在动作执行时验证
for action_idx, action_result in enumerate(action_results):
    # 关键帧 1: 动作前（验证 precondition）
    scene_graph_before = scene_graphs_craft[action_idx]
    
    # 执行动作
    action_result = execute_action(action)
    
    # 关键帧 2: 动作后（验证 postcondition）
    scene_graph_after = scene_graphs_craft[action_idx + 1]
    
    # 验证约束
    validate_constraints_at_keyframe(...)
```

### 方案 3: ShouldTriggerValidation 实现

```python
def should_trigger_validation(prev_state, current_state, event, action_log):
    """决定是否触发约束验证"""
    
    # 1. 关键动作执行时（必须验证）
    key_actions = ['put_in', 'put_on', 'pick_up', 'toggle_on', 'toggle_off']
    if event.action_type in key_actions:
        return True
    
    # 2. 状态显著变化
    if state_changed_significantly(prev_state, current_state):
        return True
    
    # 3. 对象关系变化
    if relations_changed(prev_state, current_state):
        return True
    
    return False
```

---

## 推荐实施步骤

### 短期（快速改进）

1. **在动作执行时验证**（已在 `TIMING_VALIDATION_UPDATE.md`）
   - 使用现有的场景图（每个动作一帧）
   - 在动作前后分别验证 pre/post 约束

2. **只对关键动作验证**
   - 跳过 `navigate_to_obj` 的验证
   - 只验证 `put_in`, `put_on`, `pick_up`, `toggle` 等

### 长期（完整实现）

1. **实现智能关键帧选择**
   - 只在状态显著变化时生成场景图
   - 减少场景图数量，提高效率

2. **实现 ShouldTriggerValidation**
   - 根据状态变化触发验证
   - 避免冗余验证

3. **子目标级别的验证**（可选）
   - 类似 REFLECT，在子目标达成时验证
   - 但使用可执行约束而非 LLM

---

## 总结

### CRAFT 当前状态

- ✅ 为每个动作生成场景图（9 个动作 → 9 个场景图）
- ⚠️ 不是真正的关键帧选择（固定模式）
- ❌ 只在最终状态验证（时机不对）
- ❌ 没有 ShouldTriggerValidation

### REFLECT 的方法

- ✅ 使用子目标作为关键帧（动态选择）
- ✅ 在子目标达成时验证（及时检测）
- ✅ 基于 LLM 验证子目标

### CRAFT 优化方向

1. **短期**：在动作执行时验证（时序验证）✅ 已提供方案
2. **中期**：智能关键帧选择（减少冗余）
3. **长期**：实现 ShouldTriggerValidation（动态触发）

### 关键区别总结

| 方面 | CRAFT (当前) | REFLECT | CRAFT (优化后) |
|------|-------------|---------|----------------|
| 关键帧定义 | 每个动作一帧 | 子目标达成时 | 状态变化时 |
| 选择方式 | 固定 | 动态 | 动态 |
| 验证方式 | 可执行约束 | LLM 验证 | 可执行约束 |
| 验证时机 | 最终状态 | 每个子目标后 | 动作前后 |
| 优势 | 可执行、可复现 | 细粒度检测 | 可执行 + 及时检测 |

**CRAFT 的优势在于可执行约束，但需要改进关键帧选择和验证时机。**

