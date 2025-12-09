# 关键帧方法分析：CRAFT vs REFLECT

## 问题回答

### 1. CRAFT 现在是否采用关键帧？

**答案：部分采用，但不是最优的关键帧方法**

**当前实现**：
- ✅ 为**每个动作事件**生成一个场景图
- ✅ 总共生成 9 个场景图（对应 9 个动作）
- ⚠️ **不是真正的关键帧选择**，而是"每个动作一帧"

**问题**：
- 没有选择真正关键的状态变化点
- 可能包含冗余的场景图（状态没有变化）
- 例如：`navigate_to_obj` 动作可能不会改变场景图，但仍然生成

### 2. 与 REFLECT 的关键帧方法有什么不同？

**主要区别**：

| 方面 | CRAFT (当前) | REFLECT |
|------|-------------|---------|
| **关键帧定义** | 每个动作一帧 | 子目标达成时 |
| **选择标准** | 固定（每个动作） | 动态（状态变化） |
| **验证方式** | 可执行约束（但时机不对） | LLM 子目标验证 |
| **验证时机** | 最终状态 | 每个子目标后 |
| **优势** | 可执行、可复现 | 细粒度检测 |

---

## 详细对比

### CRAFT 当前实现

#### 场景图生成方式

从 `demo1.ipynb` Step 3 可以看到：

```python
# 为每个 event 生成场景图
for event_idx, (event, action_result) in enumerate(zip(events_craft, action_results)):
    print(f"\nProcessing Event {event_idx + 1}/{len(events_craft)}...")
    # ... 生成场景图
    scene_graphs_craft.append(sg)
```

**特点**：
- ✅ 为**每个动作事件**生成一个场景图
- ✅ 总共生成 9 个场景图（对应 9 个动作）
- ⚠️ **不是真正的关键帧**，而是"每个动作一帧"

### 约束验证方式

从 `demo1.ipynb` Step 6 可以看到：

```python
# 只在最终状态验证
final_scene_graph = scene_graphs_craft[-1]
# 验证所有约束
for comp_const in compiled_constraints:
    is_valid, reason = evaluate_constraint(condition_expr, final_scene_graph)
```

**特点**：
- ❌ **只在最终状态验证**
- ❌ 没有在动作执行时验证
- ❌ 没有使用关键帧触发验证

---

## REFLECT 的关键帧方法

### REFLECT 的关键帧选择

REFLECT 使用**子目标（subgoal）**作为关键帧：

1. **子目标定义**：每个子目标对应一个关键状态变化
   - 例如：`pick_up(mug)` → 子目标：mug 被拿起
   - 例如：`put_in(mug, coffee_machine)` → 子目标：mug 在咖啡机内

2. **关键帧选择**：
   - 在**子目标应该达成的时间点**检查
   - 不是每帧检查，而是**在关键状态变化时检查**

3. **验证方式**：
   - 使用 LLM 验证子目标是否达成
   - 基于场景描述（scene description）判断

### REFLECT 的验证流程

```
Action 1: navigate_to_obj(mug)
  → 子目标：到达 mug 附近
  → 关键帧：Action 1 后
  → 验证：mug 是否可见/可达

Action 2: pick_up(mug)
  → 子目标：mug 被拿起
  → 关键帧：Action 2 后
  → 验证：mug 是否在机器人手中

Action 3: put_in(mug, coffee_machine)
  → 子目标：mug 在咖啡机内
  → 关键帧：Action 3 后
  → 验证：mug 是否在咖啡机内，咖啡机是否为空（precondition）
```

---

## CRAFT 与 REFLECT 的对比

| 方面 | CRAFT (当前) | REFLECT | CRAFT (理想) |
|------|-------------|---------|--------------|
| **场景图生成** | 每个动作一帧 | 关键状态变化时 | 关键状态变化时 |
| **关键帧选择** | 每个动作 | 子目标达成时 | 动作前后 + 状态变化 |
| **验证时机** | 只在最终状态 | 每个子目标后 | 动作前后分别验证 |
| **验证方式** | 可执行约束 | LLM 子目标验证 | 可执行约束 + 时序验证 |
| **触发条件** | 无（固定验证） | 子目标时间点 | ShouldTriggerValidation |

---

## CRAFT 的问题

### 1. 不是真正的关键帧

**当前实现**：
- 为每个动作生成场景图（9 个动作 → 9 个场景图）
- 这是"每个动作一帧"，不是"关键帧"

**问题**：
- 没有选择真正关键的状态变化点
- 可能包含冗余的场景图（状态没有变化）

### 2. 验证时机不对

**当前实现**：
- 只在最终状态验证所有约束
- 没有在动作执行时验证

**问题**：
- 无法检测动作执行时的违反
- 例如：put_in 前应该检查"容器必须为空"，但当前没有检查

### 3. 缺少 ShouldTriggerValidation

**Method.md 中的理想实现**：
```python
if ShouldTriggerValidation(prev_state, world_state, event):
    # 验证约束
```

**当前实现**：
- ❌ 没有 `ShouldTriggerValidation` 函数
- ❌ 固定验证，不根据状态变化触发

---

## 优化方案

### 方案 1: 真正的关键帧选择

```python
def should_generate_scene_graph(prev_state, current_state, action):
    """决定是否生成场景图（关键帧选择）"""
    # 1. 动作执行前后（必须）
    if action is not None:
        return True
    
    # 2. 状态发生显著变化
    if state_changed_significantly(prev_state, current_state):
        return True
    
    # 3. 对象可见性变化
    if visibility_changed(prev_state, current_state):
        return True
    
    return False

def state_changed_significantly(prev_state, current_state):
    """检查状态是否发生显著变化"""
    # 检查对象位置变化
    # 检查对象状态变化（open/closed, empty/filled）
    # 检查关系变化（inside, on_top_of）
    ...
```

### 方案 2: 动作相关的关键帧

```python
# 在动作执行时生成关键帧
for action_idx, action_result in enumerate(action_results):
    action_name = action_result.get('action_name', '')
    
    # 关键帧 1: 动作前（验证 precondition）
    scene_graph_before = get_scene_graph_before_action(action_idx)
    
    # 执行动作
    action_result = execute_action(action)
    
    # 关键帧 2: 动作后（验证 postcondition）
    scene_graph_after = get_scene_graph_after_action(action_idx)
    
    # 验证约束
    validate_constraints_at_keyframe(
        scene_graph_before, 
        scene_graph_after, 
        action_name
    )
```

### 方案 3: ShouldTriggerValidation 实现

```python
def should_trigger_validation(prev_state, current_state, event, action_log):
    """决定是否触发约束验证"""
    
    # 1. 动作执行时（必须验证）
    if event.action_type in ['put_in', 'put_on', 'pick_up', 'toggle_on', 'toggle_off']:
        return True
    
    # 2. 状态显著变化
    if state_changed_significantly(prev_state, current_state):
        return True
    
    # 3. 对象关系变化
    if relations_changed(prev_state, current_state):
        return True
    
    # 4. 周期性验证（每 N 帧）
    if frame_idx % validation_interval == 0:
        return True
    
    return False
```

---

## 推荐方案

### 短期优化（快速实施）

1. **在动作执行时验证**（已在 `TIMING_VALIDATION_UPDATE.md` 中）
   - 动作前验证 precondition
   - 动作后验证 postcondition
   - 使用现有的场景图（每个动作一帧）

2. **选择关键动作验证**
   - 只对关键动作（put_in, put_on, toggle 等）验证
   - 跳过导航动作的验证

### 长期优化（完整实现）

1. **实现 ShouldTriggerValidation**
   - 根据状态变化触发验证
   - 避免冗余验证

2. **智能关键帧选择**
   - 只在状态显著变化时生成场景图
   - 减少场景图数量，提高效率

3. **子目标级别的验证**
   - 类似 REFLECT，在子目标达成时验证
   - 但使用可执行约束而非 LLM 验证

---

## 总结

### CRAFT 当前实现

- ✅ 为每个动作生成场景图（9 个动作 → 9 个场景图）
- ❌ 不是真正的关键帧选择
- ❌ 只在最终状态验证
- ❌ 没有 ShouldTriggerValidation

### REFLECT 的方法

- ✅ 使用子目标作为关键帧
- ✅ 在子目标达成时验证
- ✅ 基于 LLM 验证子目标

### CRAFT 优化方向

1. **短期**：在动作执行时验证（时序验证）
2. **长期**：实现 ShouldTriggerValidation 和智能关键帧选择

### 关键区别

| 方面 | CRAFT (当前) | REFLECT |
|------|-------------|---------|
| 关键帧定义 | 每个动作 | 子目标达成时 |
| 验证方式 | 可执行约束（但时机不对） | LLM 子目标验证 |
| 验证时机 | 最终状态 | 每个子目标后 |
| 优势 | 可执行、可复现 | 细粒度检测 |

**CRAFT 的优势在于可执行约束，但需要改进验证时机。**

