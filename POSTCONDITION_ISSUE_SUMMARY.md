# Postcondition Violation 问题诊断总结

## 一、问题现象

即使使用了 Postcondition Temporal Window，仍然有 postcondition 违反：

```
❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be on top of SinkBasin
❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
```

但实际上这些应该是满足的。

---

## 二、可能的问题原因（按优先级排序）

### 2.1 节点匹配问题（最可能）⚠️

**问题描述**：
- 约束中提到的对象名称（如 `Mug`, `SinkBasin`, `Faucet`）可能无法匹配到场景图中的节点
- 场景图中的节点名称通常包含 ID（如 `Mug_0b3dbbd3`, `SinkBasin_3138b92f`, `Faucet_4105d586`）
- `find_node_by_name` 使用部分匹配，但可能无法正确匹配

**检查方法**：
1. 在 Step 5.5 排查 cell 中，检查约束中提到的对象是否在场景图中找到
2. 查看场景图中的节点列表，确认是否有类似名称的节点
3. 检查 `find_node_by_name` 的匹配逻辑

**可能的问题**：
- 部分匹配逻辑不够健壮
- 对象名称格式不一致（如 "SinkBasin" vs "SinkBasin_3138b92f"）
- 节点名称中的 ID 导致匹配失败

### 2.2 空间关系未建立（很可能）⚠️

**问题描述**：
- `on_top_of(Mug, SinkBasin)` 关系可能在场景图中没有建立
- 可能原因：
  1. `parentReceptacles` metadata 中没有该关系
  2. 位置信息不足以建立关系（位置判断失败）
  3. `determine_spatial_relation_hybrid` 函数判断失败

**检查方法**：
1. 在 Step 5.5 排查 cell 中，检查场景图中的所有边
2. 确认是否存在 `Mug_xxx --[on_top_of]--> SinkBasin_xxx` 的边
3. 检查两个节点的位置信息，手动计算是否符合 `on_top_of` 条件

**可能的问题**：
- 位置判断的阈值不合适（z_diff, horizontal_dist）
- SinkBasin 的表面类型判断失败（可能不在关键词列表中）
- 关系建立时跳过了某些对象对

### 2.3 状态属性同步延迟（很可能）⚠️

**问题描述**：
- `isToggled` 属性可能没有正确同步
- 即使 metadata 中有 `isToggledOn: true`，scene graph 中的 `isToggled` 可能仍是 `False`
- 可能是因为 metadata 读取逻辑有问题

**检查方法**：
1. 在 Step 5.5 排查 cell 中，对比 scene graph 中的 `isToggled` 和 metadata 中的 `isToggledOn`
2. 检查 `obj.get('isToggledOn', False) or obj.get('isToggled', False)` 是否正确读取

**可能的问题**：
- metadata 字段名不对（可能不是 `isToggledOn`，而是其他字段名）
- 属性同步时机不对（虽然每个 frame 都读取，但可能读取的是错误的 frame）

### 2.4 filled 约束检查逻辑错误

**问题描述**：
- 对于 Sink 来说，"filled" 可能不是通过 `isFilled` 属性检查
- Sink 的 filled 可能意味着：
  1. Sink 内部有液体对象（如 Pot filled with water）
  2. Sink 的 `fillLiquid` 属性不为空
  3. Sink 的 `isFilledWithLiquid` 为 true

**当前问题**：
- `evaluate_constraint` 函数中可能只检查了 `isFilled` 属性
- 没有检查 Sink 内部是否有液体对象
- 没有检查 `fillLiquid` 属性

**检查方法**：
1. 在 Step 5.5 排查 cell 中，检查 Sink 内部是否有液体对象
2. 检查 metadata 中的 `fillLiquid` 和 `isFilledWithLiquid`

**可能的问题**：
- filled 约束检查逻辑不正确（只检查 `isFilled`，不检查内部液体）

### 2.5 任务相关对象过滤问题（不太可能）

**问题**：
- 如果在某个地方使用了 `extract_task_relevant_subgraph`，可能会过滤掉某些对象
- 例如：SinkBasin 可能不在任务相关对象列表中，导致被过滤掉

**检查方法**：
- 确认 Step 5 中是否使用了任务相关子图提取
- 如果没有使用，说明问题不在这里

---

## 三、诊断步骤

### 3.1 运行 Step 5.5 排查 Cell

运行 Step 5.5 排查 cell，它会输出：
1. 每个 violation 对应的帧号
2. 场景图中的节点和边信息
3. 约束中提到的对象是否在场景图中找到
4. 相关边的详细信息
5. 状态属性的详细信息（isToggled, isFilled 等）

### 3.2 检查关键信息

对于每个违反的 postcondition，检查：

#### 对于 `on_top_of(Mug, SinkBasin)`：
1. ✅ Mug 节点是否存在？节点名称是什么？
2. ✅ SinkBasin 节点是否存在？节点名称是什么？
3. ✅ 是否存在 `Mug_xxx --[on_top_of]--> SinkBasin_xxx` 的边？
4. ✅ 如果不存在，为什么？是 metadata 中没有，还是位置判断失败？
5. ✅ 两个节点的位置信息是什么？手动计算是否符合 `on_top_of` 条件？

#### 对于 `toggled_on(Faucet)`：
1. ✅ Faucet 节点是否存在？节点名称是什么？
2. ✅ Faucet 节点的 `isToggled` 属性值是多少？
3. ✅ metadata 中的 `isToggledOn` 值是多少？
4. ✅ 两者是否一致？如果不一致，为什么？

#### 对于 `filled(Sink)`：
1. ✅ Sink 节点是否存在？节点名称是什么？
2. ✅ Sink 节点的 `isFilled` 属性值是多少？
3. ✅ Sink 内部是否有液体对象？如果有，是哪些？
4. ✅ metadata 中的 `isFilledWithLiquid` 和 `fillLiquid` 值是多少？
5. ✅ Sink 的 filled 应该通过什么判断？是 `isFilled` 属性，还是内部液体对象？

---

## 四、可能的问题根源总结

### 4.1 节点匹配问题

**症状**：
- 约束中提到的对象未在场景图中找到
- 场景图中有类似名称的节点，但匹配失败

**解决方案**：
- 改进 `find_node_by_name` 函数，使其更健壮
- 添加更详细的匹配日志

### 4.2 空间关系未建立

**症状**：
- 约束中提到的对象都在场景图中找到
- 但相关边（on_top_of, inside）不存在

**解决方案**：
- 检查关系建立的逻辑
- 改进 `determine_spatial_relation_hybrid` 函数
- 降低位置判断的阈值（如果太严格）

### 4.3 状态属性同步延迟

**症状**：
- 约束中提到的对象都在场景图中找到
- 但状态属性（isToggled, isFilled）与 metadata 不一致

**解决方案**：
- 检查 metadata 字段名是否正确
- 确保每个 frame 都正确同步状态属性

### 4.4 filled 约束检查逻辑错误

**症状**：
- Sink 节点存在
- 但 `isFilled` 属性为 False，即使 Sink 内部有液体对象

**解决方案**：
- 改进 filled 约束检查逻辑
- 对于容器（如 Sink），检查内部是否有液体对象，而不只是 `isFilled` 属性

---

## 五、建议的修复方案

### 5.1 改进节点匹配（高优先级）

```python
def find_node_by_name(sg, name):
    """查找节点（改进版，更健壮的匹配）"""
    name_lower = name.lower().strip()
    
    # 1. 精确匹配
    for n in sg.nodes:
        if n.name.lower() == name_lower:
            return n
    
    # 2. 提取对象名称（去掉 ID，只比较基础名称）
    name_base = name_lower.split('_')[0] if '_' in name_lower else name_lower
    for n in sg.nodes:
        node_base = n.name.lower().split('_')[0] if '_' in n.name.lower() else n.name.lower()
        if name_base == node_base:
            return n
    
    # 3. 部分匹配（对象名称或类型）
    for n in sg.nodes:
        if name_lower in n.name.lower() or n.name.lower() in name_lower:
            return n
        if name_lower in n.object_type.lower() or n.object_type.lower() in name_lower:
            return n
    
    return None
```

### 5.2 改进 filled 约束检查（高优先级）

```python
def check_filled(sg, obj_name):
    """检查对象是否 filled（改进版）"""
    node = find_node_by_name(sg, obj_name)
    if not node:
        return False, "Node not found"
    
    # 1. 检查 isFilled 属性
    if node.attributes.get('isFilled', False):
        return True, "isFilled attribute is True"
    
    # 2. 对于容器（如 Sink），检查内部是否有液体对象
    if 'sink' in obj_name.lower() or 'container' in node.object_type.lower():
        inside_liquid_objects = []
        for edge in sg.edges.values():
            if edge.end.name == node.name and edge.edge_type == 'inside':
                inside_obj = edge.start
                if inside_obj.attributes.get('isFilled', False):
                    inside_liquid_objects.append(inside_obj.name)
        if inside_liquid_objects:
            return True, f"Contains filled objects: {inside_liquid_objects}"
    
    # 3. 检查 fillLiquid 属性
    if node.attributes.get('fillLiquid'):
        return True, f"fillLiquid: {node.attributes.get('fillLiquid')}"
    
    return False, "Not filled"
```

### 5.3 改进空间关系建立（中优先级）

如果 `on_top_of` 关系未建立，可能需要：
1. 降低位置判断的阈值（z_diff, horizontal_dist）
2. 扩展表面类型关键词列表（确保包含 "sinkbasin"）
3. 改进 `determine_spatial_relation_hybrid` 函数的判断逻辑

---

## 六、下一步行动

1. **运行 Step 5.5 排查 cell**，获取详细的诊断信息
2. **根据诊断结果**，确定具体是哪个问题（节点匹配、关系未建立、状态属性同步、filled 检查逻辑）
3. **修复对应的问题**：
   - 如果是节点匹配问题，改进 `find_node_by_name`
   - 如果是关系未建立，改进关系建立逻辑
   - 如果是状态属性同步问题，确保每个 frame 都正确同步
   - 如果是 filled 检查逻辑问题，改进 filled 约束检查

4. **重新运行 Step 5**，验证修复效果

---

## 七、关键问题总结

### 7.1 为什么会出现这些问题？

1. **节点名称格式不一致**：约束中使用的是对象名称（如 "Mug"），但场景图中是带 ID 的（如 "Mug_0b3dbbd3"）
2. **空间关系建立依赖位置信息**：如果位置信息不准确，可能导致关系未建立
3. **状态属性同步时机**：虽然每个 frame 都读取，但如果 metadata 中字段名不对，仍然会失败
4. **filled 约束语义不清**：对于容器（如 Sink），"filled" 的含义可能不同（是容器本身 filled，还是容器内有液体对象）

### 7.2 这是 scene graph 生成的问题吗？

**部分是的**：
- 如果节点匹配失败 → scene graph 生成时可能过滤掉了某些对象
- 如果空间关系未建立 → scene graph 生成时关系建立逻辑有问题

**部分不是**：
- 如果状态属性同步延迟 → 这是 metadata 读取的问题，不是 scene graph 生成的问题
- 如果 filled 检查逻辑错误 → 这是约束检查的问题，不是 scene graph 生成的问题

### 7.3 需要检查的关键点

1. ✅ **节点是否在场景图中**：约束中提到的对象是否都在场景图中？
2. ✅ **关系是否正确建立**：相关的空间关系（on_top_of, inside）是否正确建立？
3. ✅ **状态属性是否正确同步**：isToggled, isFilled 等属性是否与 metadata 一致？
4. ✅ **约束检查逻辑是否正确**：filled 约束的检查逻辑是否正确？

