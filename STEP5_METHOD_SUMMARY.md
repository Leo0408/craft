# Step 5 失败检测方法总结

## 一、当前 Step 5 方法概述

### 1.1 核心流程

Step 5 实现了基于 **动作级约束验证** 的失败检测，主要包括以下步骤：

```
1. 约束分组
   ↓
2. 逐帧检测（对每个动作）
   ├─ Precondition 检查（动作执行前）
   └─ Postcondition 检查（动作执行后）
   ↓
3. 错误过滤（排除 Robot/NoneType）
   ↓
4. Failure Root Collapsing（收敛级联失败）
   ↓
5. LLM 分析（汇总所有错误）
```

### 1.2 关键特性

#### ✅ 1. 逐帧检测
- **Precondition**：使用 `events[action_idx]` 的场景图（动作执行前）
- **Postcondition**：使用 `events[action_idx + 1]` 的场景图（动作执行后）
- **优势**：能够在正确的时刻检查约束，符合因果顺序

#### ✅ 2. 虚拟 Robot 节点
- 自动为所有场景图添加虚拟 Robot 节点
- 避免 `Node 'Robot' not found` 错误
- **目的**：模拟环境中 Robot 节点不在场景图中，但约束需要它

#### ✅ 3. 硬性错误过滤
- **排除规则**：
  - `'Robot'` 相关错误
  - `'NoneType'` 相关错误
  - `'gripper'` 相关错误
  - 标记为 `'excluded'` 的错误
- **原因**：模拟环境中这些错误是伪失败，不是真实的执行失败

#### ✅ 4. Failure Root Collapsing
- 收敛级联失败到根失败
- **策略**：优先选择最早的 **precondition violation**
- **效果**：减少噪声，聚焦真正的根因

#### ✅ 5. LLM 分析
- 汇总所有真实错误后发送给 LLM
- **限制**：只基于已验证的约束失败进行解释，禁止引入假设
- **明确排除**：Robot 相关问题不在 LLM 分析范围内

---

## 二、为什么会有这么多 Pre/Post 违反？

### 2.1 问题现象

之前可能只检测到 **1-2 个**违反，但现在可能检测到 **10+ 个**违反。

### 2.2 可能的原因分析

#### 🔴 **原因 1：逐帧检测 vs 只检测最终状态**

**之前的方法**（可能）：
- 只在最终场景图上检查所有约束
- 一个早期失败可能被后续状态"掩盖"
- 只看到最终结果，看不到中间过程

**当前的方法**：
- **每一帧都检查**对应的约束
- 如果 Step 3 的 postcondition 失败，会被检测到
- 如果 Step 5 的 precondition 失败，也会被检测到
- **结果**：能看到所有中间过程的失败

**示例**：
```
任务：makeCoffee，10 个动作
- 之前：只在最终状态检查 → 可能只看到最终失败 → 1-2 个违反
- 现在：每帧检查 → Step 3、5、7、9 的失败都被检测到 → 4-5 个违反
```

#### 🔴 **原因 2：约束数量增加**

每个动作的约束数量：

| 动作类型 | Pre 约束数 | Post 约束数 | 总计 |
|---------|-----------|------------|------|
| `put_in(X, Y)` | 3 | 1 | 4 |
| `put_on(X, Y)` | 1 | 1 | 2 |
| `toggle_on(X)` | 1 | 1 | 2 |
| `pick_up(X)` | 2 | 1 | 3 |

**示例**：如果任务有 10 个动作，平均每个动作 2-3 个约束：
- 总约束数：**20-30 个**
- 如果其中 5 个约束失败 → 5 个违反

**之前可能的问题**：
- 可能只生成了部分约束
- 或者某些约束没有被检查

#### 🔴 **原因 3：场景图生成不稳定**

**当前实现**：
```python
# 每次检查都重新生成场景图
eval_sg = generate_scene_graph_from_event_enhanced(
    events[action_idx],
    task_info,
    timestep=action_idx,
    action=action
)
```

**问题**：
- `generate_scene_graph_from_event_enhanced` 可能每次生成略有不同
- 如果场景图生成不稳定，可能导致：
  - 同一约束在不同时刻检测结果不一致
  - 检测到"感知不一致"而不是"真实失败"

**建议**：
- 预先缓存所有关键帧的场景图
- 避免重复生成

#### 🔴 **原因 4：场景图状态延迟（AI2THOR 特性）**

**问题**：
- AI2THOR 中，动作执行后场景图可能**还没有完全更新**
- 例如：`put_on(Pot, StoveBurner-4)` 执行后，场景图可能还没反映 `Pot` 已经放在 `StoveBurner-4` 上
- 导致 **postcondition 失败**（实际是感知延迟，不是执行失败）

**示例**：
```
Step 8: put_on(Pot, StoveBurner-4)
→ 检查 postcondition: on_top_of(Pot, StoveBurner-4)
→ 场景图还没更新 → 失败
→ 但实际上动作已经成功执行
```

**这解释了为什么会有这么多 postcondition 违反**。

#### 🔴 **原因 5：级联失败（但已被 Collapse）**

**问题**：
- 一个早期失败（如 Step 3 的 precondition 失败）会导致后续所有相关约束失败
- 例如：
  - Step 3 失败 → `holding(Pot)` 不满足
  - Step 4-10 的所有 `put_in`/`put_on` 的 precondition（需要 `holding(Pot)`）都会失败
  - **结果**：1 个根失败 → 7 个派生失败 = 8 个违反

**当前处理**：
- `collapse_failures` 已经收敛到根失败
- 但**仍然会显示所有违反**（用于完整诊断）

#### 🔴 **原因 6：节点匹配可能不准确**

**当前实现**：
```python
def find_node_by_name(sg: SceneGraph, name: str):
    # 部分匹配：name_lower in n.name.lower()
```

**问题**：
- 部分匹配可能匹配到错误节点
- 或者匹配不到节点（节点名称格式不一致）
- 导致约束评估失败

**示例**：
```
约束：empty(CoffeeMachine)
场景图节点：CoffeeMachine_43b68f52
→ 如果匹配失败 → 节点未找到 → 约束失败
```

---

## 三、如何验证和调试

### 3.1 检查约束数量

```python
# 在 Step 5 开始时添加
print(f"总约束数: {len(constraints)}")
for action_idx in sorted(constraints_by_action.keys()):
    action_constraints = constraints_by_action[action_idx]
    print(f"动作 {action_idx}: {len(action_constraints['pre'])} pre + {len(action_constraints['post'])} post")
```

### 3.2 检查场景图生成

```python
# 检查场景图是否稳定
for i in range(3):
    sg1 = generate_scene_graph_from_event_enhanced(events[5], task_info)
    sg2 = generate_scene_graph_from_event_enhanced(events[5], task_info)
    print(f"场景图一致性: {sg1.nodes == sg2.nodes}")
```

### 3.3 检查节点匹配

```python
# 在 evaluate_constraint 中添加调试输出
node = find_node_by_name(sg, container_name)
print(f"查找节点 '{container_name}': {node.name if node else 'NOT FOUND'}")
```

### 3.4 区分真实失败 vs 感知延迟

```python
# 检查是否是场景图未更新导致的失败
# 如果动作执行成功，但 postcondition 失败 → 可能是感知延迟
```

---

## 四、建议的改进方向

### 4.1 立即改进（高优先级）

#### 1. **缓存场景图**
```python
# 预先生成所有关键帧的场景图
scene_graphs_by_frame = {}
for i, event in enumerate(events):
    scene_graphs_by_frame[i] = generate_scene_graph_from_event_enhanced(
        event, task_info, timestep=i
    )
    scene_graphs_by_frame[i] = add_virtual_robot_node(scene_graphs_by_frame[i])
```

#### 2. **改进节点匹配**
```python
# 精确匹配优先，部分匹配作为回退
def find_node_by_name(sg, name):
    # 1. 精确匹配
    node = sg.get_node(name)
    if node:
        return node
    
    # 2. 部分匹配（但要求更高相似度）
    # 例如：要求 name 是节点名称的前缀或完整匹配
    for n in sg.nodes:
        if name.lower() in n.name.lower():
            # 检查相似度
            similarity = calculate_similarity(name, n.name)
            if similarity > 0.8:  # 阈值
                return n
    
    return None
```

#### 3. **区分感知延迟 vs 真实失败**
```python
# 如果场景图在动作执行后 N 帧内仍无更新，标记为感知延迟
if action_result.status == "SUCCESS" and postcondition_failed:
    # 等待下一帧再检查
    next_frame_sg = generate_scene_graph_from_event_enhanced(events[action_idx + 2])
    if constraint_satisfied_in(next_frame_sg):
        # 标记为感知延迟，不是真实失败
        violation['is_perception_delay'] = True
```

### 4.2 长期改进（中优先级）

#### 1. **优化约束生成**
- 检查是否所有约束都是必要的
- 考虑合并一些相关的约束

#### 2. **提前终止**
```python
# 如果检测到 precondition 失败，提前终止
if precondition_failed:
    print("⚠️ Precondition 失败，跳过后续动作的检查")
    break  # 不再检查后续动作
```

#### 3. **置信度阈值**
```python
# 为约束评估添加置信度
if constraint_confidence < 0.7:
    # 标记为不确定，不判定为失败
    violation['is_uncertain'] = True
```

---

## 五、总结

### 5.1 为什么会看到更多违反？

1. **逐帧检测**：能看到所有中间过程的失败，而不是只看最终结果
2. **约束数量增加**：每个动作都有多个约束，增加了失败检测的覆盖度
3. **场景图状态延迟**：AI2THOR 中场景图更新可能有延迟，导致 postcondition 失败
4. **级联失败**：一个早期失败会导致多个后续失败（虽然已被收敛）
5. **节点匹配问题**：部分匹配可能导致误判

### 5.2 这是好事还是坏事？

**好事**：
- ✅ 能发现所有中间过程的失败
- ✅ 更全面的失败诊断
- ✅ 符合 Method.md 的设计理念（逐帧检测）

**需要改进**：
- ⚠️ 需要区分"真实失败"和"感知延迟"
- ⚠️ 需要优化节点匹配的准确性
- ⚠️ 需要缓存场景图，避免重复生成

### 5.3 下一步行动

1. **验证约束数量**：检查每个动作的约束数是否合理
2. **检查场景图稳定性**：验证场景图生成是否一致
3. **调试节点匹配**：确认节点匹配是否正确
4. **区分感知延迟**：添加逻辑区分真实失败和感知延迟
5. **缓存场景图**：预先生成所有关键帧的场景图

---

## 六、参考代码位置

- Step 5 代码：`demo3.ipynb` Cell 29
- 约束生成：`reasoning/constraint_generator.py`
- 场景图生成：`core/enhanced_generate_scene_graph.py`
- 方法文档：`Method.md` Section 12

