# Step 5.5: Postcondition 违反详细排查 Cell

## 一、功能说明

在 Step 5 失败检测之后，新增了 **Step 5.5 详细排查 cell**，用于详细检查那些显示为违反但实际上可能没有违反的 postcondition。

## 二、主要功能

### 2.1 自动识别约束类型

排查 cell 会自动识别以下约束类型：
1. **toggled_on / toggled_off**：检查设备的开关状态
2. **on_top_of**：检查物体的空间关系（是否在表面上）
3. **inside**：检查物体是否在容器内
4. **container_empty**：检查容器是否为空

### 2.2 详细检查信息

对于每个 violation，会输出：
- **动作信息**：Step、Action、Failure Type
- **约束信息**：Constraint Description、Reason
- **帧信息**：检查的帧号（动作执行前/后）、action_idx
- **场景图状态**：
  - 节点信息（名称、类型、属性）
  - 边信息（空间关系）
  - Metadata 原始状态（isToggledOn, isToggled 等）

## 三、使用示例

### 3.1 示例输出

```
================================================================================
Violation 1/5: Step 4
================================================================================
  Action: (put_on, Mug, SinkBasin)
  Failure Type: POSTCONDITION VIOLATION
  Constraint: Mug must be on top of SinkBasin
  Reason: Evaluated: False

  📍 检查帧: Frame 5 (动作执行后) (action_idx=3, step=4)

  ✅ 场景图已生成: 15 个节点, 22 条边

  🔍 详细检查: on_top_of(Mug, SinkBasin)
     on_top_of(Mug, SinkBasin) = False
     Mug 节点: Mug_0b3dbbd3 (位置: (2.3, 1.0, 0.5))
     SinkBasin 节点: SinkBasin_3138b92f (位置: (2.5, 0.9, 0.0))
     相关边:
       Mug_0b3dbbd3 --[on_top_of]--> SinkBasin_3138b92f
```

### 3.2 示例输出（toggled_on）

```
================================================================================
Violation 2/5: Step 5
================================================================================
  Action: (toggle_on, Faucet)
  Failure Type: POSTCONDITION VIOLATION
  Constraint: Faucet must be toggled on
  Reason: Evaluated: False

  📍 检查帧: Frame 6 (动作执行后) (action_idx=4, step=5)

  ✅ 场景图已生成: 15 个节点, 22 条边

  🔍 详细检查: Faucet.isToggled
     Node 'Faucet' found, isToggled=True
     节点名称: Faucet_4105d586
     节点类型: Faucet
     所有属性: {'isFilled': False, 'isOpen': False, 'isToggled': True, ...}
     Metadata isToggledOn: True
     Metadata isToggled: None
```

## 四、关键功能

### 4.1 帧号计算

- **Precondition**：使用 `events[action_idx]`（动作执行前）
- **Postcondition**：使用 `events[action_idx + 1]`（动作执行后）

### 4.2 场景图重新生成

- 为每个 violation 重新生成对应的场景图
- 使用正确的 timestep 和 action
- 确保状态属性（isToggled, isOpen 等）正确同步

### 4.3 详细状态检查

#### toggled_on 检查
- 检查节点的 `isToggled` 属性
- 检查原始 metadata 的 `isToggledOn` 和 `isToggled`
- 输出节点的所有属性

#### on_top_of 检查
- 检查是否存在 `on_top_of` 类型的边
- 检查两个节点的位置信息
- 列出所有相关边

#### inside 检查
- 检查是否存在 `inside` 类型的边
- 检查两个节点的存在
- 列出所有相关边

#### container_empty 检查
- 检查容器内是否有对象（通过 inside 边）
- 列出容器内的所有对象
- 输出是否为空

## 五、使用步骤

1. **运行 Step 5**：确保 Step 5 已运行并生成了 violations
2. **运行 Step 5.5**：直接运行新的排查 cell
3. **查看输出**：检查每个 violation 的详细状态信息

## 六、故障排除

### 6.1 数据未找到

如果出现 "⚠️ 未找到 Step 5 的数据"，请确保：
- Step 5 已经完整运行
- Step 5 结尾的数据保存代码已执行

### 6.2 帧号超出范围

如果出现 "⚠️ 帧 X 超出范围"，可能是：
- events 数组长度不足
- action_idx 计算错误

### 6.3 节点未找到

如果出现 "Node 'X' not found"，可能是：
- 节点名称不匹配（使用了部分匹配）
- 场景图生成时该节点未包含

## 七、预期效果

运行 Step 5.5 后，你应该能够：
- ✅ 看到每个 violation 的详细检查信息
- ✅ 了解场景图中的实际状态（isToggled, on_top_of 等）
- ✅ 发现哪些 violation 可能是误报（场景图状态实际满足约束）
- ✅ 找到场景图生成或状态同步的问题

---

## 八、代码位置

- **Step 5.5 Cell**：`demo3.ipynb` Cell 30
- **数据保存**：`demo3.ipynb` Cell 29（Step 5 结尾）

