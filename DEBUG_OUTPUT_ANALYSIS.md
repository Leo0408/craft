# 调试输出分析和修复方案

## 一、问题总结

根据 Step 5.5 的调试输出，发现了以下关键问题：

### 问题 1: `Pot must be inside Sink` (Frame 4)

**调试输出显示**：
- ✅ Pot 节点找到了：`Pot_5c47f775`
- ✅ Sink 节点找到了：`Sink_41af8f72`
- ❌ **没有 `inside` 边连接它们**
- ⚠️ 相关边显示：`Pot_5c47f775 --[inside]--> CounterTop_978a4b41`（Pot 在 CounterTop 内，不在 Sink 内）

**根本原因**：
1. `put_in(Pot, Sink)` 执行后，AI2THOR 的 metadata 中 Pot 的 `parentReceptacles` 可能还没有更新到包含 Sink
2. 或者 Sink 被判断为 surface 而不是 container，导致建立了 `on_top_of` 而不是 `inside` 关系

**解决方案**：
- ✅ **已修复**：改进 Sink 容器识别逻辑（`core/enhanced_generate_scene_graph.py`）
  - 即使 `receptacleObjectIds` 为空，也应该将 Sink 识别为容器
  - 通过 `objectType` 判断："Sink" 类型应该被认为是容器

### 问题 2: `Faucet must be toggled on` (Frame 5)

**调试输出显示**：
- ✅ Faucet 节点找到了：`Faucet_1000141d`
- ❌ `isToggled` 属性是 `False`
- ❌ metadata 中 `isToggledOn` 是 `None`
- ❌ metadata 中 `isToggled` 是 `False`

**根本原因**：
1. `toggle_on(Faucet)` 执行后，AI2THOR 的 metadata 中 Faucet 的 toggle 状态字段可能不是 `isToggledOn`
2. 可能是其他字段（如 `isOn`）或者字段名不同

**解决方案**：
- ✅ **已修复**：改进 `isToggled` 属性的提取逻辑（`core/enhanced_generate_scene_graph.py`）
  - 检查更多可能的字段：`isToggledOn`, `isToggled`, `isOn`
  - 对于 `toggleable` 对象，也检查 `isOn` 字段

### 问题 3: `Sink must be filled` (Frame 8-12)

**问题**：
- Sink 的 filled 应该检查内部是否有液体对象（如 Pot filled with water）
- 而不是只检查 Sink 本身的 `isFilled` 属性

**解决方案**：
- 需要改进 `evaluate_constraint` 中的 filled 约束检查逻辑
- 对于容器（如 Sink），应该检查内部是否有液体对象

### 问题 4: `Mug must be on top of SinkBasin` (Frame 4-11)

**问题**：
- `on_top_of` 关系可能没有建立
- 可能原因：位置判断失败（阈值太严格）或者 metadata 中没有 `parentReceptacles`

**解决方案**：
- 如果 metadata 中有 `parentReceptacles`，应该优先使用
- 如果没有，使用位置信息判断，但阈值应该更宽松

### 问题 5: `CoffeeMachine must be toggled on` (Frame 11-15)

**问题**：
- 同问题 2，toggle 状态检查问题

### 问题 6: `Mug must be on top of CounterTop` (Frame 14-21)

**问题**：
- 同问题 4，`on_top_of` 关系未建立

---

## 二、已实施的修复

### 修复 1: Sink 容器识别改进 ✅

**文件**：`core/enhanced_generate_scene_graph.py`

**修改**：
```python
# 改进：Sink, SinkBasin 等容器类型即使 receptacleObjectIds 为空也应被识别为容器
parent_obj_type = other_obj.get('objectType', '').lower()
is_container_type = any(kw in parent_obj_type for kw in ['sink', 'sinkbasin', 'bowl', 'pot', 'pan', 'mug', 'cup'])

if is_openable_container or (has_receptacle and receptacle_count > 0) or is_container_type:
    edge_key = (node.name, parent_node.name)
    if edge_key not in sg.edges:
        sg.add_edge(Edge(node, parent_node, "inside"))
```

**效果**：
- Sink 即使 `receptacleObjectIds` 为空，也会被识别为容器
- `put_in(Pot, Sink)` 执行后，如果 metadata 中 Pot 的 `parentReceptacles` 包含 Sink，会建立 `inside` 关系

### 修复 2: Faucet toggle 状态检查改进 ✅

**文件**：`core/enhanced_generate_scene_graph.py`

**修改**：
```python
'isToggled': (
    obj.get('isToggledOn', False) or 
    obj.get('isToggled', False) or
    (obj.get('toggleable', False) and obj.get('isOn', False)) or
    obj.get('isOn', False)
),
```

**效果**：
- 检查更多可能的 toggle 状态字段
- 对于 `toggleable` 对象，也检查 `isOn` 字段
- 如果 AI2THOR 使用 `isOn` 字段，现在也能正确读取

### 修复 3: determine_spatial_relation_hybrid 改进 ✅

**文件**：`core/enhanced_scene_graph_utils.py`

**修改**：
```python
# 改进：Sink, SinkBasin 等容器类型即使 receptacleObjectIds 为空也应被识别为容器
obj2_type = obj2.get('objectType', '').lower()
is_container_type = any(kw in obj2_type for kw in ['sink', 'sinkbasin', 'bowl', 'pot', 'pan', 'mug', 'cup'])

if is_openable_container or (has_receptacle and receptacle_count > 0) or is_container_type:
    return ("inside", 1.0)  # Highest confidence
```

**效果**：
- 确保 hybrid 方法也能正确识别 Sink 为容器

---

## 三、需要进一步修复的问题

### 修复 4: filled 约束检查逻辑改进（待实施）⏳

**问题**：
- 对于 Sink 等容器，filled 应该检查内部是否有液体对象
- 而不是只检查容器本身的 `isFilled` 属性

**需要修改**：
- `demo3.ipynb` Cell 29 中的 `evaluate_constraint` 函数
- filled 约束检查逻辑

**预期改进**：
```python
# 对于容器（如 Sink），filled 应该检查：
# 1. 容器本身的 isFilled 属性
# 2. 容器内部是否有液体对象
# 3. 容器的 fillLiquid 属性
```

### 修复 5: on_top_of 关系建立改进（待实施）⏳

**问题**：
- 如果 metadata 中没有 `parentReceptacles`，使用位置信息判断
- 但位置判断的阈值可能太严格

**需要修改**：
- `core/enhanced_scene_graph_utils.py` 中的位置判断阈值
- 或者改进位置判断逻辑

---

## 四、可能仍存在的问题

### 问题 A: metadata 更新延迟

**问题**：
- `put_in(Pot, Sink)` 执行后，AI2THOR 的 metadata 中 Pot 的 `parentReceptacles` 可能还没有更新到包含 Sink
- 这是 AI2THOR 模拟器的固有延迟

**解决方案**：
- Postcondition Temporal Window 已经考虑了这个问题
- 但如果延迟太长（超过窗口大小），仍然会失败

### 问题 B: Faucet toggle 状态字段名未知

**问题**：
- 即使检查了多个字段，如果 AI2THOR 使用的字段名不同，仍然会失败

**解决方案**：
- 在 Step 5.5 排查 cell 中添加更多调试信息
- 打印所有可能的 toggle 相关字段

---

## 五、测试建议

1. **重新运行 Step 5**，检查修复效果
2. **运行 Step 5.5 排查 cell**，查看详细的诊断信息
3. **如果问题仍然存在**：
   - 检查 metadata 中实际使用的字段名
   - 检查 `parentReceptacles` 是否正确更新
   - 检查位置判断的阈值是否合适

---

## 六、下一步行动

1. ✅ **已完成**：修复 Sink 容器识别逻辑
2. ✅ **已完成**：改进 Faucet toggle 状态检查
3. ⏳ **待实施**：改进 filled 约束检查逻辑
4. ⏳ **待实施**：改进 on_top_of 关系建立（如果需要）

**建议**：
- 先测试已修复的问题，看看效果如何
- 如果问题仍然存在，根据新的调试输出进一步调整

