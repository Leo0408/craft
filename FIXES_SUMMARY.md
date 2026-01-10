# Postcondition Violation 修复总结

## 一、已完成的修复 ✅

### 修复 1: Sink 容器识别问题

**问题**：
- `put_in(Pot, Sink)` 执行后，Pot 和 Sink 之间没有建立 `inside` 关系
- Sink 被判断为 surface 而不是 container，导致建立了 `on_top_of` 而不是 `inside` 关系

**根本原因**：
- Sink 的 `receptacleObjectIds` 可能为空或不存在
- 当前的判断逻辑：`is_openable_container or (has_receptacle and receptacle_count > 0)`
- Sink 既不是 `openable_container`，也没有 `receptacleObjectIds`，导致被判断为 surface

**修复方案**：
- ✅ **文件**：`core/enhanced_generate_scene_graph.py`
- ✅ **修改**：在容器判断逻辑中添加 `objectType` 检查
  ```python
  # 改进：Sink, SinkBasin 等容器类型即使 receptacleObjectIds 为空也应被识别为容器
  parent_obj_type = other_obj.get('objectType', '').lower()
  is_container_type = any(kw in parent_obj_type for kw in ['sink', 'sinkbasin', 'bowl', 'pot', 'pan', 'mug', 'cup'])
  
  if is_openable_container or (has_receptacle and receptacle_count > 0) or is_container_type:
      edge_key = (node.name, parent_node.name)
      if edge_key not in sg.edges:
          sg.add_edge(Edge(node, parent_node, "inside"))
  ```

- ✅ **文件**：`core/enhanced_scene_graph_utils.py`（`determine_spatial_relation_hybrid` 函数）
- ✅ **修改**：同样添加 `objectType` 检查

**效果**：
- Sink 现在会被正确识别为容器
- `put_in(Pot, Sink)` 执行后，如果 metadata 中 Pot 的 `parentReceptacles` 包含 Sink，会建立 `inside` 关系

---

### 修复 2: Faucet toggle 状态检查问题

**问题**：
- `toggle_on(Faucet)` 执行后，`isToggled` 属性仍然是 `False`
- metadata 中 `isToggledOn` 是 `None`，`isToggled` 是 `False`

**根本原因**：
- AI2THOR 的 metadata 中，Faucet 的 toggle 状态可能使用不同的字段名
- 可能使用 `isOn` 而不是 `isToggledOn`

**修复方案**：
- ✅ **文件**：`core/enhanced_generate_scene_graph.py`
- ✅ **修改**：改进 `isToggled` 属性的提取逻辑
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

---

## 二、需要进一步诊断的问题 ⏳

### 问题 1: `Sink must be filled` 约束检查

**问题**：
- Sink 的 filled 应该检查内部是否有液体对象（如 Pot filled with water）
- 而不是只检查 Sink 本身的 `isFilled` 属性

**当前检查逻辑**：
- 只检查 `node.attributes.get('isFilled', False)`

**需要改进**：
- 对于容器（如 Sink），应该检查：
  1. 容器本身的 `isFilled` 属性
  2. 容器内部是否有液体对象（如 Pot filled with water）
  3. 容器的 `fillLiquid` 属性

**需要修改**：
- `demo3.ipynb` Cell 29 中的 `evaluate_constraint` 函数
- filled 约束检查逻辑

**注意**：
- 这个问题在 Step 5.5 排查 cell 中已经有部分诊断逻辑
- 但 `evaluate_constraint` 函数中还没有实现

---

### 问题 2: `Mug must be on top of SinkBasin` 关系未建立

**问题**：
- `on_top_of` 关系可能没有建立
- 可能原因：
  1. metadata 中 Mug 的 `parentReceptacles` 不包含 SinkBasin
  2. 位置判断失败（阈值太严格）

**需要诊断**：
- 检查 metadata 中 Mug 的 `parentReceptacles`
- 检查两个节点的位置信息
- 检查位置判断的阈值（z_diff, horizontal_dist）

**可能需要修复**：
- 如果 metadata 中没有 `parentReceptacles`，位置判断的阈值可能需要更宽松
- 或者 SinkBasin 应该被识别为 surface（而不是 container）

---

### 问题 3: `CoffeeMachine must be toggled on` 约束检查

**问题**：
- 同问题 2（Faucet toggle 状态检查）

**解决方案**：
- 应该已经被修复 2 解决
- 但需要进一步测试确认

---

### 问题 4: `Mug must be on top of CounterTop` 关系未建立

**问题**：
- 同问题 2（`on_top_of` 关系未建立）

**解决方案**：
- 需要进一步诊断

---

## 三、可能的根本原因分析

### 原因 1: metadata 更新延迟

**问题**：
- `put_in(Pot, Sink)` 执行后，AI2THOR 的 metadata 中 Pot 的 `parentReceptacles` 可能还没有更新到包含 Sink
- 这是 AI2THOR 模拟器的固有延迟

**解决方案**：
- Postcondition Temporal Window 已经考虑了这个问题
- 但如果延迟太长（超过窗口大小），仍然会失败

**检查方法**：
- 在 Step 5.5 排查 cell 中，检查 Frame 4-11 中 Pot 的 `parentReceptacles` 是否包含 Sink

---

### 原因 2: Faucet toggle 状态字段名未知

**问题**：
- 即使检查了多个字段，如果 AI2THOR 使用的字段名不同，仍然会失败

**解决方案**：
- 在 Step 5.5 排查 cell 中添加更多调试信息
- 打印所有可能的 toggle 相关字段

**检查方法**：
- 在 Step 5.5 排查 cell 中，打印 Faucet 对象的所有 metadata 字段
- 查找包含 "toggle"、"on"、"state" 等关键词的字段

---

### 原因 3: 空间关系建立时机

**问题**：
- `on_top_of` 关系可能需要在动作执行后的多个帧中才能建立
- 如果只检查单个帧，可能错过

**解决方案**：
- Postcondition Temporal Window 已经考虑了这个问题
- 但需要确保场景图生成逻辑在每个帧中都正确执行

---

## 四、下一步行动

1. ✅ **已完成**：修复 Sink 容器识别逻辑
2. ✅ **已完成**：改进 Faucet toggle 状态检查
3. ⏳ **待实施**：改进 filled 约束检查逻辑（需要修改 `evaluate_constraint` 函数）
4. ⏳ **待测试**：重新运行 Step 5，检查修复效果
5. ⏳ **待诊断**：如果问题仍然存在，使用 Step 5.5 排查 cell 进一步诊断

**建议**：
- 先测试已修复的问题（Sink 容器识别和 Faucet toggle 状态），看看效果如何
- 如果 `Pot must be inside Sink` 和 `Faucet must be toggled on` 问题仍然存在：
  - 检查 metadata 中实际使用的字段名
  - 检查 `parentReceptacles` 是否正确更新
  - 检查 toggle 状态字段名
- 然后处理 filled 约束检查和 on_top_of 关系建立问题

