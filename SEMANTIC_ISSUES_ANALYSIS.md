# 语义问题分析和解决方案

## 一、问题总结

### 问题 1: Sink/SinkBasin 语义不准确

**问题描述**：
- 实际情况：`Mug inside the Sink`（Mug 在 Sink 内部）
- 约束生成：`Mug must be on top of SinkBasin`（Mug 在 SinkBasin 上面）
- 场景图显示：`Mug --[inside]--> SinkBasin`（Mug 在 SinkBasin 内部）

**根本原因**：
1. **动作语义混淆**：`put_on(Mug, SinkBasin)` 生成了 `on_top_of` 约束，但实际执行时 AI2THOR 将其识别为 `inside` 关系
2. **对象名称映射**：在 `utils/action_primitives.py` 中，`put_in` 函数会将 `Sink` 映射为 `SinkBasin`：
   ```python
   if target_obj_type == 'Sink':
       target_obj_type = 'SinkBasin'
   ```
   但约束生成时可能使用了原始的 `SinkBasin` 名称
3. **约束生成逻辑**：`put_on` 动作模板默认生成 `on_top_of` 约束，但实际上 `put_on(Mug, SinkBasin)` 应该生成 `inside` 约束

### 问题 2: Faucet toggle 状态检查失败

**问题描述**：
- `toggle_on(Faucet)` 执行后，`isToggled` 属性仍然是 `False`
- metadata 中 `isToggledOn` 是 `None`，`isToggled` 是 `False`

**根本原因**：
1. **字段名不匹配**：AI2THOR 中 Faucet 的 toggle 状态可能使用不同的字段名
2. **修复未生效**：虽然在 `core/enhanced_generate_scene_graph.py` 中修复了，但 `demo3.ipynb` 中可能有其他地方还在使用旧的逻辑

---

## 二、解决方案

### 解决方案 1: Sink/SinkBasin 语义问题

#### 方案 A: 改进约束生成逻辑（推荐）

**问题**：`put_on` 动作模板默认生成 `on_top_of` 约束，但对于 SinkBasin 等容器，应该生成 `inside` 约束

**解决方案**：
- 在约束生成时，根据目标对象的类型判断应该生成 `inside` 还是 `on_top_of` 约束
- 如果目标对象是容器类型（Sink, SinkBasin, Bowl 等），`put_on` 应该生成 `inside` 约束

**需要修改**：
- `reasoning/constraint_generator.py` 中的 `put_on` 约束生成逻辑
- 或者在约束生成后，对特定对象类型进行语义校正

#### 方案 B: 使用 LLM 进行语义对齐（备选）

**问题**：模板方法难以覆盖所有语义变化

**解决方案**：
- 在约束生成后，使用 LLM 进行语义对齐
- LLM 可以根据场景图的实际状态，修正约束的语义

**优缺点**：
- ✅ 可以处理复杂的语义变化
- ❌ 需要额外的 LLM 调用，增加成本

#### 方案 C: 语义模板映射（折中方案）

**问题**：需要手动维护语义映射

**解决方案**：
- 创建一个语义映射表，定义特定对象组合的关系类型
- 例如：`put_on(Mug, SinkBasin)` → `inside(Mug, SinkBasin)`

**需要修改**：
- 添加语义映射表
- 在约束生成后，应用语义映射

### 解决方案 2: Faucet toggle 状态问题

**问题**：`demo3.ipynb` 中可能使用了旧的 `isToggled` 提取逻辑

**解决方案**：
- 检查 `demo3.ipynb` 中所有生成 scene graph 的地方
- 确保都使用修复后的 `isToggled` 提取逻辑
- 或者在 `core/enhanced_generate_scene_graph.py` 中统一处理

---

## 三、推荐实施步骤

### 步骤 1: 修复 Faucet toggle 状态（高优先级）

1. 检查 `demo3.ipynb` 中所有 scene graph 生成的地方
2. 确保都使用 `core/enhanced_generate_scene_graph.py` 中的修复逻辑
3. 或者统一使用修复后的属性提取逻辑

### 步骤 2: 修复 Sink/SinkBasin 语义问题（中优先级）

**推荐方案 A（改进约束生成逻辑）**：

1. 在 `reasoning/constraint_generator.py` 中，改进 `put_on` 约束生成逻辑
2. 根据目标对象类型判断应该生成 `inside` 还是 `on_top_of` 约束
3. 如果目标对象是容器类型（Sink, SinkBasin, Bowl 等），`put_on` 应该生成 `inside` 约束

**或者方案 C（语义模板映射）**：

1. 创建语义映射表
2. 在约束生成后，应用语义映射
3. 例如：`put_on(Mug, SinkBasin)` → `inside(Mug, SinkBasin)`

### 步骤 3: 测试和验证

1. 重新运行 Step 5，检查修复效果
2. 使用 Step 5.5 排查 cell 验证
3. 确认语义对齐正确

---

## 四、具体修改建议

### 修改 1: 改进 put_on 约束生成逻辑

**文件**：`reasoning/constraint_generator.py`

**修改位置**：`put_on` 动作模板的约束生成逻辑

**修改内容**：
```python
# 在生成 put_on 的 postcondition 时，根据目标对象类型判断
# 如果目标对象是容器类型，应该生成 inside 而不是 on_top_of

CONTAINER_TYPES = {'sink', 'sinkbasin', 'bowl', 'pot', 'pan', 'mug', 'cup', 'coffeemachine'}

def _generate_constraints_for_action(self, action_str: str, ...):
    # ...
    if action_type == 'put_on':
        target_obj = action_args[1] if len(action_args) > 1 else None
        if target_obj and target_obj.lower() in CONTAINER_TYPES:
            # 对于容器类型，生成 inside 约束
            post_constraints.append({
                'template': 'inside',
                'args': action_args,
                'description': f"{action_args[0]} must be inside {target_obj}"
            })
        else:
            # 对于非容器类型，生成 on_top_of 约束
            post_constraints.append({
                'template': 'on_top_of',
                'args': action_args,
                'description': f"{action_args[0]} must be on top of {target_obj}"
            })
```

### 修改 2: 统一 isToggled 提取逻辑

**文件**：`demo3.ipynb` Cell 29（或其他生成 scene graph 的地方）

**修改内容**：
- 确保所有 scene graph 生成都使用 `core/enhanced_generate_scene_graph.py` 中的 `generate_scene_graph_from_event_enhanced` 函数
- 或者确保所有地方的 `isToggled` 提取逻辑都一致

---

## 五、总结

### 问题优先级

1. **高优先级**：Faucet toggle 状态问题（影响 toggle 约束检查）
2. **中优先级**：Sink/SinkBasin 语义问题（影响空间关系约束）

### 推荐方案

1. **Faucet toggle**：统一使用修复后的 `isToggled` 提取逻辑
2. **Sink/SinkBasin**：改进约束生成逻辑，根据目标对象类型判断关系类型

### 实施建议

1. 先修复 Faucet toggle 状态问题
2. 然后修复 Sink/SinkBasin 语义问题
3. 最后测试验证修复效果

