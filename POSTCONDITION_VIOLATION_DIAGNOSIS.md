# Postcondition Violation 诊断总结

## 一、问题现象

即使使用了 Postcondition Temporal Window，仍然有 postcondition 违反：

```
❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be on top of SinkBasin
❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
```

但这些实际上应该是满足的。

---

## 二、可能的问题原因

### 2.1 节点匹配问题（最可能）⚠️

**问题**：
- 约束中提到的对象名称（如 `Mug`, `SinkBasin`, `Faucet`）可能无法匹配到场景图中的节点
- 场景图中的节点名称通常包含 ID（如 `Mug_0b3dbbd3`, `SinkBasin_3138b92f`, `Faucet_4105d586`）
- `find_node_by_name` 使用部分匹配，可能无法正确匹配

**示例**：
```python
# 约束：on_top_of(Mug, SinkBasin)
# 场景图节点：Mug_0b3dbbd3, SinkBasin_3138b92f
# find_node_by_name("Mug") 应该能匹配到 "Mug_0b3dbbd3"
# 但如果匹配逻辑有问题，可能返回 None
```

**检查方法**：
- 在 Step 5.5 排查 cell 中，检查约束中提到的对象是否在场景图中找到
- 检查 `find_node_by_name` 的匹配逻辑是否正确

### 2.2 空间关系未建立 ⚠️

**问题**：
- `on_top_of(Mug, SinkBasin)` 关系可能在场景图中没有建立
- 可能原因：
  1. `parentReceptacles` metadata 中没有该关系
  2. 位置信息不足以建立关系（位置判断失败）
  3. 关系建立的逻辑有问题

**检查方法**：
- 在 Step 5.5 排查 cell 中，检查场景图中的所有边
- 确认是否存在 `Mug_xxx --[on_top_of]--> SinkBasin_xxx` 的边

### 2.3 状态属性同步问题 ⚠️

**问题**：
- `isToggled` 属性可能没有正确同步
- 即使 metadata 中有 `isToggledOn: true`，scene graph 中的 `isToggled` 可能仍是 `False`
- 可能原因：
  1. `obj.get('isToggledOn', False) or obj.get('isToggled', False)` 逻辑有问题
  2. metadata 中字段名不对

**检查方法**：
- 在 Step 5.5 排查 cell 中，对比 scene graph 中的 `isToggled` 和 metadata 中的 `isToggledOn`
- 确认属性是否正确同步

### 2.4 filled 约束检查逻辑问题 ⚠️

**问题**：
- 对于 Sink 来说，"filled" 可能不是通过 `isFilled` 属性检查
- Sink 的 filled 可能意味着：
  1. Sink 内部有液体对象（如 Pot filled with water）
  2. Sink 的 `fillLiquid` 属性不为空
  3. Sink 的 `isFilledWithLiquid` 为 true

**当前检查**：
- 只检查了 `node.attributes.get('isFilled', False)`
- 可能没有检查 Sink 内部是否有液体对象

**检查方法**：
- 在 Step 5.5 排查 cell 中，检查 Sink 内部是否有液体对象
- 检查 metadata 中的 `fillLiquid` 和 `isFilledWithLiquid`

### 2.5 任务相关对象过滤问题

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

#### 对于 `toggled_on(Faucet)`：
1. ✅ Faucet 节点是否存在？节点名称是什么？
2. ✅ Faucet 节点的 `isToggled` 属性值是多少？
3. ✅ metadata 中的 `isToggledOn` 值是多少？
4. ✅ 两者是否一致？如果不一致，为什么？

#### 对于 `filled(Sink)`：
1. ✅ Sink 节点是否存在？节点名称是什么？
2. ✅ Sink 节点的 `isFilled` 属性值是多少？
3. ✅ Sink 内部是否有液体对象？
4. ✅ metadata 中的 `isFilledWithLiquid` 和 `fillLiquid` 值是多少？

---

## 四、可能的解决方案

### 4.1 改进节点匹配逻辑

如果节点匹配失败，改进 `find_node_by_name`：
```python
def find_node_by_name(sg, name):
    """查找节点（改进版，更健壮的匹配）"""
    name_lower = name.lower().strip()
    
    # 1. 精确匹配
    for n in sg.nodes:
        if n.name.lower() == name_lower:
            return n
    
    # 2. 提取对象名称（去掉 ID）
    name_base = name_lower.split('_')[0]
    for n in sg.nodes:
        node_base = n.name.lower().split('_')[0]
        if name_base == node_base:
            return n
    
    # 3. 部分匹配
    for n in sg.nodes:
        if name_lower in n.name.lower() or n.name.lower() in name_lower:
            return n
        if name_lower in n.object_type.lower() or n.object_type.lower() in name_lower:
            return n
    
    return None
```

### 4.2 改进 filled 约束检查

对于 Sink 等容器，filled 应该检查内部是否有液体：
```python
def check_filled(sg, obj_name):
    """检查对象是否 filled（改进版）"""
    node = find_node_by_name(sg, obj_name)
    if not node:
        return False, "Node not found"
    
    # 1. 检查 isFilled 属性
    if node.attributes.get('isFilled', False):
        return True, "isFilled attribute is True"
    
    # 2. 对于容器，检查内部是否有液体对象
    if 'sink' in obj_name.lower() or 'container' in node.object_type.lower():
        for edge in sg.edges.values():
            if edge.end.name == node.name and edge.edge_type == 'inside':
                inside_obj = edge.start
                if inside_obj.attributes.get('isFilled', False):
                    return True, f"Contains filled object: {inside_obj.name}"
    
    # 3. 检查 fillLiquid 属性
    if node.attributes.get('fillLiquid'):
        return True, f"fillLiquid: {node.attributes.get('fillLiquid')}"
    
    return False, "Not filled"
```

### 4.3 确保场景图包含所有对象

**关键**：确认 `generate_scene_graph_from_event_enhanced` 是否包含了所有对象，还是只包含了任务相关对象。

如果只包含了任务相关对象，可能需要：
1. 确保任务相关对象提取包含所有必要的对象（如 SinkBasin, Faucet）
2. 或者在 postcondition 检查时使用完整场景图，而不是任务相关子图

---

## 五、总结

### 5.1 最可能的问题

根据描述，最可能的问题是：

1. **节点匹配失败**：约束中的对象名称无法匹配到场景图中的节点
2. **空间关系未建立**：`on_top_of` 关系在场景图中没有正确建立
3. **状态属性同步延迟**：`isToggled` 虽然 metadata 更新了，但 scene graph 中还没有

### 5.2 检查清单

运行 Step 5.5 排查 cell 后，检查：
- [ ] 约束中提到的对象是否都在场景图中找到？
- [ ] 场景图中是否存在相关的边（on_top_of, inside）？
- [ ] 状态属性（isToggled, isFilled）是否正确同步？
- [ ] metadata 中的原始状态是什么？

### 5.3 下一步行动

1. **运行 Step 5.5 排查 cell**，获取详细的诊断信息
2. **根据诊断结果**，确定具体是哪个问题
3. **修复对应的问题**：
   - 如果是节点匹配问题，改进 `find_node_by_name`
   - 如果是关系未建立，改进关系建立逻辑
   - 如果是状态属性同步问题，确保每个 frame 都正确同步

---

## 六、代码位置

- **Scene Graph 生成**：`core/enhanced_generate_scene_graph.py`
- **约束检查**：`demo3.ipynb` Cell 29 (Step 5)
- **诊断 Cell**：`demo3.ipynb` Cell 30 (Step 5.5)
- **节点查找函数**：`demo3.ipynb` Cell 29 中的 `find_node_by_name`

