# AI2THOR 生成方法对比：REFLECT vs CRAFT

## 概述

本文档对比了 REFLECT 框架和 CRAFT 框架中生成 AI2THOR 执行数据的方法。

## 1. Controller 初始化

### REFLECT (gen_data.py)
```python
controller = Controller(
    agentMode="default",
    massThreshold=None,
    scene=task['scene'],
    visibilityDistance=1.5,
    gridSize=0.25,
    renderDepthImage=True,
    renderInstanceSegmentation=True,
    width=960,
    height=960,
    fieldOfView=60,
    platform=CloudRendering
)
```

### CRAFT (demo1.ipynb)
```python
controller = Controller(
    agentMode="default",
    massThreshold=None,
    scene=task_info_craft['scene'],
    visibilityDistance=1.5,
    gridSize=0.25,
    renderDepthImage=True,
    renderInstanceSegmentation=True,
    width=960,
    height=960,
    fieldOfView=60,
    platform=CloudRendering  # CloudRendering doesn't require X display
)
```

**✅ 结论**: Controller 初始化参数**完全相同**，两者使用相同的配置。

---

## 2. 动作执行方式

### REFLECT 方法

REFLECT 使用**模块化的动作原语（action primitives）**：

1. **动作定义**: 在 `action_primitives.py` 中定义函数（如 `navigate_to_obj`, `pick_up`, `put_in` 等）
2. **动态调用**: 通过 `globals()[action]` 动态调用函数
3. **任务工具类**: 使用 `TaskUtil` 类管理状态和失败注入

```python
# gen_data.py 中的执行方式
for i, instr in enumerate(instrs):
    lis = instr.split(',')
    lis = [item.strip("() ") for item in lis]
    action = lis[0]
    params = lis[1:]
    func = globals()[action]  # 动态获取函数
    retval = func(taskUtil, *params)  # 调用动作原语
```

**特点**:
- ✅ 模块化设计，动作原语可复用
- ✅ 支持失败注入（drop, missing_step, failed_action 等）
- ✅ 使用 TaskUtil 统一管理状态
- ✅ 支持 preactions（前置动作）

### CRAFT 方法

CRAFT 在 notebook 中**直接实现动作执行逻辑**：

1. **内联实现**: 在 demo1.ipynb 中直接编写动作执行代码
2. **条件判断**: 使用 if-elif 判断动作类型
3. **简化实现**: 专注于演示工作流，不包含失败注入

```python
# demo1.ipynb 中的执行方式
for action_idx, action_instr in enumerate(action_instructions, 1):
    parts = [p.strip() for p in action_instr.split(',')]
    action_name = parts[0]
    params = parts[1:] if len(parts) > 1 else []
    
    if action_name == "navigate_to_obj":
        # 直接实现导航逻辑
        ...
    elif action_name == "pick_up":
        # 直接实现抓取逻辑
        ...
    elif action_name == "put_in":
        # 直接实现放置逻辑
        ...
```

**特点**:
- ✅ 代码集中，易于理解
- ✅ 适合演示和快速原型
- ⚠️ 不支持失败注入
- ⚠️ 代码重复，不易复用

---

## 3. 关键功能对比

| 功能 | REFLECT | CRAFT |
|------|---------|-------|
| **Controller 初始化** | ✅ 相同 | ✅ 相同 |
| **动作原语** | ✅ 模块化（action_primitives.py） | ⚠️ 内联实现 |
| **失败注入** | ✅ 支持（drop, missing_step, failed_action, blocking 等） | ❌ 不支持 |
| **look_at 功能** | ✅ 在动作原语中实现 | ✅ 独立函数实现 |
| **对象映射** | ✅ TaskUtil 管理 | ✅ 临时变量管理 |
| **状态管理** | ✅ TaskUtil 类 | ⚠️ 局部变量 |
| **数据保存** | ✅ 保存到 pickle 和 JSON | ⚠️ 仅保存到内存 |

---

## 4. 具体动作实现对比

### navigate_to_obj

#### REFLECT (action_primitives.py)
```python
def navigate_to_obj(taskUtil, obj, to_drop=False, failure_injection_idx=None):
    # 使用 TaskUtil 的方法
    # 支持失败注入（drop）
    # 返回导航结果
```

#### CRAFT (demo1.ipynb)
```python
if action_name == "navigate_to_obj":
    # 查找对象位置
    # 计算最近可达位置
    # 使用 Teleport 导航
    # 调用 look_at_object
```

### pick_up

#### REFLECT
```python
def pick_up(taskUtil, obj, fail_execution=False):
    # 使用 TaskUtil.controller
    # 支持失败注入（fail_execution）
    # 记录交互动作
```

#### CRAFT
```python
elif action_name == "pick_up":
    # 查找对象 ID
    # 调用 look_at_object
    # 执行 PickupObject
    # 检查成功状态
```

---

## 5. 主要差异总结

### 相同点 ✅
1. **Controller 初始化参数完全相同**
2. **使用相同的 AI2THOR API**（Teleport, PickupObject, PutObject, ToggleObjectOn/Off 等）
3. **都使用 CloudRendering 平台**
4. **都实现了 look_at 功能**（确保对象可见）

### 不同点 ⚠️

#### 架构设计
- **REFLECT**: 模块化设计，动作原语独立文件，使用 TaskUtil 类管理状态
- **CRAFT**: 内联实现，代码集中在 notebook 中，使用局部变量

#### 功能完整性
- **REFLECT**: 支持多种失败注入类型，完整的任务管理
- **CRAFT**: 专注于演示工作流，不包含失败注入

#### 代码复用性
- **REFLECT**: 动作原语可复用，TaskUtil 可扩展
- **CRAFT**: 代码内联，不易复用

---

## 6. 建议

### 对于 CRAFT 框架

如果要与 REFLECT 保持一致，可以考虑：

1. **提取动作原语**: 将动作执行逻辑提取到独立的 `action_primitives.py` 文件
2. **添加失败注入支持**: 实现 REFLECT 的失败注入机制
3. **使用状态管理类**: 创建类似 TaskUtil 的类来管理执行状态
4. **保持 Controller 初始化**: 当前实现已经与 REFLECT 一致 ✅

### 当前 CRAFT 实现的优势

1. **简单直接**: 代码集中，易于理解和修改
2. **适合演示**: 专注于展示 CRAFT++ 工作流
3. **快速原型**: 不需要复杂的类结构

---

## 7. 结论

**Controller 初始化方法完全相同** ✅

**动作执行方式不同**:
- REFLECT 使用模块化的动作原语系统
- CRAFT 使用内联实现，更适合演示

**建议**: 
- 如果只是演示 CRAFT++ 工作流，当前实现已经足够
- 如果需要与 REFLECT 完全兼容或需要失败注入功能，可以考虑参考 REFLECT 的模块化设计

