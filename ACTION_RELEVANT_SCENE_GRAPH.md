# Action-Relevant Scene Graph 生成说明

## 概述

已修改 Step 2 的场景图生成逻辑，现在**只包含与当前 action 相关的节点**，使用闭包方法扩展相关节点，裁剪掉不相关的节点。

## 主要改进

### 1. ✅ Action-Relevant 场景图生成

**之前**：
- 生成完整场景图（包含所有对象）
- 然后裁剪任务相关子图（包含所有 actions 的对象）

**现在**：
- 为每一帧生成完整场景图
- **只保留与当前帧 action 相关的节点**
- 使用闭包方法扩展相关节点（容器、支撑结构等）
- 裁剪掉不相关的节点

### 2. ✅ 闭包扩展方法

使用 `extract_action_relevant_subgraph_with_closure()` 函数：
- 从当前 action 中提取相关对象（例如：`(pick_up, Mug)` → `{Mug}`）
- 使用 BFS 从相关对象开始，沿着 `inside`/`on_top_of`/`holding` 边扩展
- 确保包含所有相关的容器和支撑结构
- 例如：Mug → CounterTop（如果 Mug 在 CounterTop 上）

### 3. ✅ 按帧输出

在 Step 2 的 cell 中，为每一帧输出：
- 当前帧的 action
- 完整场景图的节点和边数量
- Action-relevant 场景图的节点和边数量
- 场景图的文字描述
- 节点列表（最多显示5个）
- 关系列表（最多显示5个）

## 代码结构

### 新增文件

**`core/action_relevant_scene_graph.py`**：
- `extract_action_relevant_objects()` - 从单个 action 提取相关对象
- `extract_action_relevant_subgraph_with_closure()` - 使用闭包方法裁剪 action-relevant 子图

### 修改的文件

**`demo3.ipynb` - Cell 19 (Step 2)**：
- 修改为按帧生成 action-relevant 场景图
- 使用 `extract_action_relevant_subgraph_with_closure()` 裁剪
- 输出每一帧的 action-relevant 场景图

## 使用示例

### 输入

```python
# 任务：makeCoffee
actions = [
    "(pick_up, Mug)",
    "(put_in, Mug, CoffeeMachine)",
    "(toggle_on, CoffeeMachine)"
]

# 事件序列
events = [event_0, event_1, event_2, ...]
```

### 输出

对于每一帧：

**Frame 1 - Action: (pick_up, Mug)**
```
完整场景图: 50 个节点, 120 条边
Action-relevant: 3 个节点, 2 条边
场景描述: Mug (empty), CounterTop, Robot. Mug is on_top_of CounterTop. Robot is holding Mug.
📦 节点 (3 个): Mug (empty), CounterTop, Robot
🔗 关系 (2 条): Robot --[holding]--> Mug; Mug --[on_top_of]--> CounterTop
```

**Frame 2 - Action: (put_in, Mug, CoffeeMachine)**
```
完整场景图: 50 个节点, 120 条边
Action-relevant: 4 个节点, 3 条边
场景描述: Mug (empty), CoffeeMachine, CounterTop, Robot. Mug is inside CoffeeMachine. CoffeeMachine is on_top_of CounterTop.
📦 节点 (4 个): Mug (empty), CoffeeMachine, CounterTop, Robot
🔗 关系 (3 条): Mug --[inside]--> CoffeeMachine; CoffeeMachine --[on_top_of]--> CounterTop; Robot --[holding]--> Mug
```

## 优势

1. **节点数量大幅减少**：只包含与当前 action 相关的节点
2. **更精确的场景表示**：每个场景图只关注当前动作涉及的对象
3. **自动扩展相关节点**：使用闭包方法自动包含容器和支撑结构
4. **便于约束验证**：每个 action 的场景图更小，验证更快

## 技术细节

### 闭包扩展规则

1. **初始节点**：从 action 中提取的对象（例如：`Mug`, `CoffeeMachine`）
2. **扩展关系**：`inside`, `on_top_of`, `holding`
3. **扩展逻辑**：
   - 如果对象在容器内（`inside`），包含容器
   - 如果对象在表面上（`on_top_of`），包含表面
   - 如果对象被持有（`holding`），包含 Robot
4. **防止过度扩展**：
   - 表面上的非 action 相关对象不会被包含
   - 只扩展功能性因果关系

### 对象匹配规则

支持精确匹配和部分匹配：
- 精确匹配：`Mug` == `Mug`
- 部分匹配：`Mug_0b3dbbd3` 包含 `Mug`
- 类型匹配：对象类型匹配（如 `CoffeeMachine` 类型）

## 更新日期

2024-12-25

