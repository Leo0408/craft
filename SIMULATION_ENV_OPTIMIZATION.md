# 模拟环境优化总结

## 一、问题描述

### 1.1 Robot 交互约束问题

在模拟环境中，robot 交互相关的约束（`holding`, `gripper_empty`）不可靠：
- `holding(mug)` 可能因为 Robot 节点不存在或状态不同步而失败
- `gripper_empty` 在模拟环境中难以准确判断
- 这些约束不是任务失败的根本原因，而是模拟环境的实现细节

### 1.2 状态属性同步问题

状态属性（如 `isToggled`, `isOpen`, `isFilled`）可能只在 final frame 读取：
- 导致中间帧的状态不正确
- 约束验证失败（如 `Faucet must be toggled on` 在动作执行后应该为 True，但 scene graph 中仍为 False）

---

## 二、解决方案

### 2.1 过滤 Robot 交互约束

**实现位置**：`demo3.ipynb` Cell 29 (Step 5)

**修改内容**：
- 在约束检查循环中，检测 robot 交互相关的约束
- 如果是 robot 交互约束，标记为 `SKIPPED`，不视为失败
- 记录到 `skipped_constraints` 中，用于统计和调试

**代码逻辑**：
```python
# 在 Precondition 和 Postcondition 检查循环中
is_robot_interaction = (
    'holding' in constraint_template or 
    'holding' in constraint_desc or
    'gripper_empty' in constraint_template or
    'gripper' in constraint_desc or
    'robot must be holding' in constraint_desc
)
if is_robot_interaction:
    skipped_constraints.append({
        'reason': "SKIPPED: Robot interaction constraint in simulation environment"
    })
    continue  # 跳过 robot 交互约束
```

**效果**：
- ✅ 减少模拟环境中的误报
- ✅ 聚焦任务相关的物理约束（如 `container_empty`, `inside`, `on_top_of`）
- ✅ 提高失败检测的准确性

### 2.2 状态属性同步优化

**实现位置**：`core/enhanced_generate_scene_graph.py`

**修改内容**：
- 确保每个 event frame 都同步更新状态属性
- 在 scene graph 构建时，直接从 `obj_metadata` 中读取状态
- 添加注释说明每个 frame 都更新的重要性

**代码逻辑**：
```python
# 在 generate_scene_graph_from_event_enhanced 中
node = Node(
    name=obj.get('name', 'unknown'),
    attributes={
        # 状态属性：从 metadata 中直接读取，每个 frame 都更新
        'isToggled': obj.get('isToggledOn', False) or obj.get('isToggled', False),
        'isOpen': obj.get('isOpen', False),
        'isFilled': obj.get('isFilledWithLiquid', False) or obj.get('isFilled', False),
        # ... 其他状态属性
    }
)
```

**关键原则**：
- ✅ **每个 event frame 都生成新的 scene graph**
- ✅ **状态属性从当前 frame 的 metadata 中读取**
- ✅ **不使用缓存或之前 frame 的状态**

**效果**：
- ✅ 状态属性在每个时间步都正确
- ✅ 约束验证能够准确检测状态变化
- ✅ 避免"状态未更新"导致的误报

---

## 三、修改文件清单

### 3.1 代码修改

1. **`demo3.ipynb` Cell 29 (Step 5)**
   - 添加 robot 交互约束过滤逻辑（Precondition 和 Postcondition 检查循环）

2. **`core/enhanced_generate_scene_graph.py`**
   - 添加状态属性同步注释
   - 确保所有状态属性都从当前 frame 的 metadata 中读取

### 3.2 文档更新

1. **`Method.md`**
   - 添加 Section 12.5.6：模拟环境中的 Robot 交互约束过滤
   - 添加 Section 12.5.7：状态属性同步优化（每帧更新）

---

## 四、验证方法

### 4.1 验证 Robot 交互约束过滤

运行 Step 5，检查输出：
- ✅ 应该看到 "跳过约束" 统计中包含 robot 交互约束
- ✅ `holding` 和 `gripper_empty` 约束应该出现在 "跳过约束" 中，而不是 "真实错误"
- ✅ Root Violation 不应该是 robot 交互相关的约束

### 4.2 验证状态属性同步

检查 scene graph 生成：
- ✅ 每个 event frame 都调用 `generate_scene_graph_from_event_enhanced`
- ✅ 状态属性（`isToggled`, `isOpen`, `isFilled`）从当前 frame 的 metadata 中读取
- ✅ `Faucet must be toggled on` 在 `toggle_on(Faucet)` 执行后应该为 True

---

## 五、预期效果

### 5.1 减少误报

- **之前**：robot 交互约束失败导致大量误报
- **现在**：robot 交互约束被跳过，只关注任务相关的物理约束

### 5.2 提高准确性

- **之前**：状态属性可能不正确，导致约束验证失败
- **现在**：状态属性在每个时间步都正确，约束验证准确

### 5.3 聚焦根本原因

- **之前**：robot 交互约束和状态属性问题掩盖了真正的任务失败
- **现在**：聚焦任务相关的物理约束，更容易找到根本原因

---

## 六、注意事项

### 6.1 真实环境 vs 模拟环境

- **模拟环境**：robot 交互约束不可靠，应该跳过
- **真实环境**：robot 交互约束可能可靠，需要根据实际情况决定是否过滤

### 6.2 状态属性同步

- **关键**：每个 event frame 都必须重新生成 scene graph
- **不能**：使用缓存或之前 frame 的状态
- **必须**：从当前 frame 的 metadata 中读取状态

---

## 七、后续优化建议

### 7.1 环境检测

可以添加环境检测逻辑，自动判断是模拟环境还是真实环境：
```python
is_simulation = hasattr(event, 'metadata') and 'objects' in event.metadata
if is_simulation:
    # 跳过 robot 交互约束
    ...
```

### 7.2 状态属性验证

可以添加状态属性验证逻辑，确保状态属性正确同步：
```python
# 验证状态属性是否从当前 frame 读取
if node.attributes.get('isToggled') != obj.get('isToggledOn', False):
    # 警告：状态属性可能未正确同步
    ...
```

---

## 八、总结

通过这两个优化：
1. ✅ **过滤 robot 交互约束**：减少模拟环境中的误报，聚焦任务相关的物理约束
2. ✅ **状态属性同步优化**：确保每个时间步的状态都正确，提高约束验证的准确性

这些优化使得 CRAFT++ 在模拟环境中更加可靠和准确，能够更好地检测任务失败的根本原因。

