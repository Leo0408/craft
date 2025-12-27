# Step 4: 约束编译说明

## 一、约束编译的作用

约束编译是将 Step 3 生成的**结构化约束模板**转换为**可执行的 Python 代码表达式**的过程。这样可以在后续的失败检测阶段直接执行这些代码来验证约束是否满足。

## 二、当前实现方式

### 2.1 编译流程

**输入**: Step 3 生成的约束列表（每个约束包含 `template` 字段）

**处理**: 通过字符串匹配识别模板类型，映射为可执行的 Python 代码

**输出**: 编译后的约束列表（每个约束包含 `condition_expr` 字段）

### 2.2 模板到代码的映射规则

当前实现使用简单的字符串匹配来识别模板类型，并映射为对应的 Python 代码：

| 模板类型 | 识别关键词 | 生成的 Python 代码 |
|---------|----------|-------------------|
| `holding(X)` | `holding` | `node.attributes.get('isPickedUp', False)` |
| `container_empty(Y)` | `empty` | `len([e for e in scene_graph.edges.values() if e.end.name == node.name and e.edge_type == 'inside']) == 0` |
| `filled(Y)` | `filled` | `node.attributes.get('isFilled', False)` |
| `toggled_on(Y)` | `toggled` | `node.attributes.get('isToggled', False)` |
| `toggled_off(Y)` | `toggled` + `off` | `not node.attributes.get('isToggled', False)` |
| `on_top_of(X, Y)` | `on_top_of` | `has_edge(node.name, target_name, 'on_top_of')` |
| `inside(X, Y)` | `inside` | `has_edge(node.name, target_name, 'inside')` |
| `reachable(X)` | `reachable` | `node is not None` |
| `gripper_empty()` | `gripper_empty` | `not any(n.attributes.get('isPickedUp', False) for n in scene_graph.nodes)` |

### 2.3 代码位置

- **Notebook Cell**: Cell 27 (Step 4)
- **核心逻辑**: 直接在 notebook cell 中实现，使用字符串匹配和正则表达式

### 2.4 编译逻辑详解

```python
# 1. 遍历每个约束
for constraint in constraints:
    raw_template = str(constraint.get('template', '')).lower()
    
    # 2. 通过字符串匹配识别模板类型
    if 'holding' in raw_template:
        # 提取对象名，生成代码片段
        code_snippet = f"holding({obj})"
        condition_expr = "node.attributes.get('isPickedUp', False)"
    elif 'empty' in raw_template:
        code_snippet = f"empty({obj})"
        condition_expr = "len([e for e in scene_graph.edges.values() if e.end.name == node.name and e.edge_type == 'inside']) == 0"
    # ... 其他模板类型
    
    # 3. 保存编译结果
    compiled.append({
        'original': constraint,
        'type': constraint.get('type', 'pre'),
        'condition_expr': condition_expr,  # 可执行的 Python 代码
        'code_snippet': code_snippet,       # 简短的代码形式（用于显示）
        'description': description,
        'template': raw_template
    })
```

## 三、改进后的输出格式

### 3.1 改进点

1. **按动作分组显示**: 将约束按 `action_index` 分组，每个动作的约束集中显示
2. **简短的代码形式**: 使用 `code_snippet` 显示简短的代码形式（如 `holding(Mug)`），而不是完整的 Python 表达式
3. **PRE/POST 分离**: 明确区分 Precondition 和 Postcondition
4. **清晰的层次结构**: 使用缩进和符号（•）使输出更易读

### 3.2 输出示例

**改进前**:
```
1. [PRECONDITION] gripper_empty(robot) -> Step 2: (pick_up, Mug) | Logic: len([e for e in scene_graph.edges.values() if e.end.name == node.name and e.edge_type == 'inside']) == 0
```

**改进后**:
```
Step 2: (pick_up, Mug)
  PRE:
    • gripper_empty()
    • reachable(Mug)
  POST:
    • holding(Mug)
```

### 3.3 完整输出格式

```
🔧 编译 makeCoffee 的动作约束...
   ✅ 编译了 19 个约束

   📋 约束列表（按动作分组）:

   Step 1: (navigate_to_obj, Mug)
     (如果没有约束，不显示)

   Step 2: (pick_up, Mug)
     PRE:
       • gripper_empty()
       • reachable(Mug)
     POST:
       • holding(Mug)

   Step 3: (put_in, Mug, CoffeeMachine)
     PRE:
       • holding(Mug)
       • container_open(CoffeeMachine)
       • container_empty(CoffeeMachine)
     POST:
       • inside(Mug, CoffeeMachine)

   ...
```

## 四、约束编译的关键点

### 4.1 模板识别

- 使用字符串匹配（`in` 操作符）识别模板类型
- 使用正则表达式提取对象名称和参数
- 支持多种变体（如 `container_empty` 和 `empty`）

### 4.2 代码生成

- 生成可执行的 Python 代码表达式（`condition_expr`）
- 这些表达式在后续验证阶段会被执行
- 代码需要访问 `node`、`scene_graph`、`target_name` 等变量

### 4.3 代码片段提取

- 从模板中提取简短的代码形式（`code_snippet`）
- 用于直观显示，而不是执行
- 格式类似函数调用：`holding(Mug)`, `inside(Mug, CoffeeMachine)`

## 五、约束编译的局限性

### 5.1 当前实现的限制

1. **字符串匹配不够精确**: 可能误匹配相似的模板
2. **不支持复杂表达式**: 只支持简单的单谓词约束
3. **硬编码映射**: 映射规则是硬编码的，不易扩展

### 5.2 改进方向（可选）

1. **使用 AST 解析**: 将模板解析为 AST，然后生成代码
2. **使用 Predicate 函数映射**: 直接使用 `PREDICATE_IMPL` 中的函数
3. **支持组合约束**: 支持 `AND`、`OR`、`NOT` 等逻辑组合

## 六、与 Step 3 的关系

- **Step 3**: 生成约束模板（如 `holding(Mug)`）
- **Step 4**: 将模板编译为可执行代码（如 `node.attributes.get('isPickedUp', False)`）
- **Step 5**: 使用编译后的代码验证约束是否满足

## 七、总结

Step 4 的约束编译是一个**模板到代码的转换过程**：

1. **输入**: 结构化约束模板（来自 Step 3）
2. **处理**: 字符串匹配 + 正则提取 + 代码生成
3. **输出**: 可执行的 Python 代码表达式 + 简短的代码片段（用于显示）

改进后的输出格式更加直观，能够清晰地看出：
- 每个动作有哪些约束
- 哪些是 Precondition，哪些是 Postcondition
- 约束的简短代码形式

