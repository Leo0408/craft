# Failure Injection 实现指南

本文档详细说明如何实现和使用 Step 3 的失败注入机制。

---

## 📋 目录

1. [概述](#概述)
2. [失败类型定义](#失败类型定义)
3. [实现详解](#实现详解)
4. [使用示例](#使用示例)
5. [扩展实现](#扩展实现)
6. [集成到任务执行](#集成到任务执行)

---

## 概述

失败注入（Failure Injection）是实验的核心资产，用于可控地生成失败场景，测试检测器的能力。

### 核心思想

1. **可控性**：精确控制失败的类型、位置和参数
2. **可复现性**：相同的配置总是产生相同的失败
3. **可扩展性**：易于添加新的失败类型

### 当前实现的 2 种失败类型

1. **MISSING_PRECONDITION**：前置条件缺失
2. **PHYSICAL_IMPOSSIBLE**：物理不可能

---

## 失败类型定义

### 文件位置

`craft-experiments/failure_injection/failure_types.py`

### 代码

```python
from enum import Enum

class FailureType(Enum):
    """失败类型枚举"""
    MISSING_PRECONDITION = "MISSING_PRECONDITION"  # 前置条件缺失
    PHYSICAL_IMPOSSIBLE = "PHYSICAL_IMPOSSIBLE"    # 物理不可能
    CAUSAL_BREAK = "CAUSAL_BREAK"                  # 因果链断裂（待实现）
    PERCEPTION_NOISE = "PERCEPTION_NOISE"          # 感知噪声（待实现）
```

### 说明

- 使用 `Enum` 确保类型安全
- 值使用字符串，便于 JSON 序列化
- 已定义 4 种类型，当前实现 2 种

---

## 实现详解

### 文件位置

`craft-experiments/failure_injection/injector.py`

### 1. MISSING_PRECONDITION 注入

#### 函数签名

```python
def inject_missing_precondition(task: Dict[str, Any], step_index: int) -> Dict[str, Any]:
```

#### 实现逻辑

```python
def inject_missing_precondition(task: Dict[str, Any], step_index: int) -> Dict[str, Any]:
    """
    注入前置条件缺失失败
    
    原理：移除任务中的关键步骤，导致后续步骤失败
    
    例如：
        原始任务：["pick_up, Mug", "put_in, CoffeeMachine, Mug", "toggle_on, CoffeeMachine"]
        移除步骤 1 (put_in)：
        失败任务：["pick_up, Mug", "toggle_on, CoffeeMachine"]
        结果：杯子没有被放入咖啡机，但后续步骤仍会执行 → 失败场景
    """
    # 1. 复制任务（避免修改原始任务）
    modified_task = task.copy()
    actions = modified_task.get("actions", []).copy()
    
    # 2. 检查步骤索引是否有效
    if 0 <= step_index < len(actions):
        # 3. 移除指定步骤
        removed_action = actions.pop(step_index)
        modified_task["actions"] = actions
        
        # 4. 记录注入的失败信息（用于后续分析）
        modified_task["injected_failure"] = {
            "type": FailureType.MISSING_PRECONDITION.value,
            "step": step_index,
            "removed_action": removed_action
        }
    
    return modified_task
```

#### 关键点

- ✅ **不修改原任务**：使用 `copy()` 创建副本
- ✅ **记录失败信息**：保存到 `injected_failure` 字段
- ✅ **索引验证**：检查步骤索引是否有效

#### 示例

```python
# 原始任务
task = {
    "name": "make_coffee",
    "actions": [
        "navigate_to_obj, Mug",
        "pick_up, Mug",
        "navigate_to_obj, CoffeeMachine",
        "put_in, CoffeeMachine, Mug",  # 步骤 3（索引从 0 开始）
        "toggle_on, CoffeeMachine"
    ]
}

# 注入失败（移除步骤 3）
failed_task = inject_missing_precondition(task, step_index=3)

# 结果
print(failed_task["actions"])
# ['navigate_to_obj, Mug', 'pick_up, Mug', 'navigate_to_obj, CoffeeMachine', 'toggle_on, CoffeeMachine']

print(failed_task["injected_failure"])
# {
#     'type': 'MISSING_PRECONDITION',
#     'step': 3,
#     'removed_action': 'put_in, CoffeeMachine, Mug'
# }
```

---

### 2. PHYSICAL_IMPOSSIBLE 注入

#### 函数签名

```python
def inject_capacity_violation(state: Dict[str, Any], object_name: str) -> Dict[str, Any]:
```

#### 实现逻辑

```python
def inject_capacity_violation(state: Dict[str, Any], object_name: str) -> Dict[str, Any]:
    """
    注入容量违反失败（物理不可能）
    
    原理：设置违反物理约束的状态
    
    例如：
        设置容器容量为 1000，但最大容量为 100
        这违反了物理约束 → 失败场景
    """
    # 1. 复制状态（避免修改原始状态）
    modified_state = state.copy()
    
    # 2. 检查对象是否存在
    if object_name in modified_state:
        obj_state = modified_state[object_name].copy()
        
        # 3. 设置违反物理约束的值
        obj_state["volume"] = 1000      # 超大的容量
        obj_state["max_capacity"] = 100 # 但最大容量很小
        obj_state["contains"] = []      # 但实际上为空
        
        modified_state[object_name] = obj_state
        
        # 4. 记录注入的失败信息
        modified_state["injected_failure"] = {
            "type": FailureType.PHYSICAL_IMPOSSIBLE.value,
            "object": object_name,
            "reason": "Capacity violation"
        }
    
    return modified_state
```

#### 关键点

- ✅ **状态级注入**：修改对象状态，而不是任务定义
- ✅ **物理约束违反**：设置不可能的状态值
- ✅ **可扩展**：可以添加其他物理约束违反类型

#### 示例

```python
# 原始状态
state = {
    "coffee_machine": {
        "volume": 0,
        "max_capacity": 100,
        "contains": []
    }
}

# 注入失败
failed_state = inject_capacity_violation(state, "coffee_machine")

# 结果
print(failed_state["coffee_machine"])
# {
#     'volume': 1000,        # 违反物理约束
#     'max_capacity': 100,   # 但最大容量只有 100
#     'contains': []
# }

print(failed_state["injected_failure"])
# {
#     'type': 'PHYSICAL_IMPOSSIBLE',
#     'object': 'coffee_machine',
#     'reason': 'Capacity violation'
# }
```

---

### 3. 统一入口函数

#### 函数签名

```python
def inject_failure(task: Dict[str, Any], failure_config: Dict[str, Any]) -> Dict[str, Any]:
```

#### 实现逻辑

```python
def inject_failure(task: Dict[str, Any], failure_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据配置注入失败（统一入口）
    
    Args:
        task: 任务定义
        failure_config: 失败配置，格式：
            {
                "type": "MISSING_PRECONDITION",
                "step": 1
            }
            或
            {
                "type": "PHYSICAL_IMPOSSIBLE",
                "object": "coffee_machine"
            }
    
    Returns:
        修改后的任务定义（包含 injected_failure 字段）
    """
    failure_type = failure_config.get("type")
    
    if failure_type == FailureType.MISSING_PRECONDITION.value:
        step = failure_config.get("step", 0)
        return inject_missing_precondition(task, step)
    
    elif failure_type == FailureType.PHYSICAL_IMPOSSIBLE.value:
        # 对于物理不可能，需要状态对象，这里返回标记
        task = task.copy()
        task["injected_failure"] = failure_config
        return task
    
    else:
        raise ValueError(f"Unknown failure type: {failure_type}")
```

#### 关键点

- ✅ **统一接口**：一个函数处理所有失败类型
- ✅ **配置驱动**：通过 JSON 配置控制注入
- ✅ **类型检查**：验证失败类型是否有效

---

## 使用示例

### 示例 1：基本使用

```python
import json
from failure_injection.injector import inject_failure
from failure_injection.failure_types import FailureType

# 1. 加载任务定义
with open("tasks/task_defs.json") as f:
    tasks = json.load(f)

task = tasks["make_coffee"]

# 2. 定义失败配置
failure_config = {
    "type": FailureType.MISSING_PRECONDITION.value,
    "step": 3  # 移除步骤 3（put_in, CoffeeMachine, Mug）
}

# 3. 注入失败
failed_task = inject_failure(task, failure_config)

# 4. 查看结果
print("原始任务动作数:", len(task["actions"]))
print("失败任务动作数:", len(failed_task["actions"]))
print("注入的失败:", failed_task.get("injected_failure"))
```

### 示例 2：从配置文件加载

```python
import json
from failure_injection.injector import inject_failure

# 1. 加载任务定义
with open("tasks/task_defs.json") as f:
    tasks = json.load(f)

# 2. 加载失败配置
with open("failure_injection/injection_config.json") as f:
    failure_configs = json.load(f)

# 3. 对每个任务的每个失败配置注入失败
for task_name, configs in failure_configs.items():
    task = tasks[task_name]
    print(f"\n任务: {task_name}")
    
    for i, failure_config in enumerate(configs):
        failed_task = inject_failure(task, failure_config)
        print(f"  失败配置 {i+1}: {failure_config['type']}")
        print(f"    原始动作数: {len(task['actions'])}")
        print(f"    失败动作数: {len(failed_task['actions'])}")
```

### 示例 3：验证失败注入

```python
def verify_failure_injection(task, failed_task):
    """验证失败注入是否正确"""
    if "injected_failure" not in failed_task:
        return False, "未找到 injected_failure 字段"
    
    failure_info = failed_task["injected_failure"]
    failure_type = failure_info["type"]
    
    if failure_type == "MISSING_PRECONDITION":
        # 检查动作数是否减少
        if len(failed_task["actions"]) >= len(task["actions"]):
            return False, "动作数未减少"
        
        # 检查移除的动作是否存在
        removed_step = failure_info["step"]
        if removed_step >= len(task["actions"]):
            return False, f"移除的步骤索引 {removed_step} 超出范围"
        
        return True, "验证通过"
    
    elif failure_type == "PHYSICAL_IMPOSSIBLE":
        # 对于物理不可能，只需要检查标记是否存在
        return True, "验证通过"
    
    else:
        return False, f"未知的失败类型: {failure_type}"

# 使用示例
task = tasks["make_coffee"]
failure_config = {"type": "MISSING_PRECONDITION", "step": 3}
failed_task = inject_failure(task, failure_config)

is_valid, message = verify_failure_injection(task, failed_task)
print(f"验证结果: {is_valid}, 消息: {message}")
```

---

## 扩展实现

### 添加新的失败类型

#### 步骤 1：在枚举中添加类型

```python
# failure_injection/failure_types.py
class FailureType(Enum):
    # ... 现有类型 ...
    NEW_FAILURE_TYPE = "NEW_FAILURE_TYPE"  # 添加新类型
```

#### 步骤 2：实现注入函数

```python
# failure_injection/injector.py
def inject_new_failure_type(task: Dict[str, Any], param: Any) -> Dict[str, Any]:
    """注入新的失败类型"""
    modified_task = task.copy()
    # 实现注入逻辑
    modified_task["injected_failure"] = {
        "type": FailureType.NEW_FAILURE_TYPE.value,
        "param": param
    }
    return modified_task
```

#### 步骤 3：在统一入口函数中添加处理

```python
# failure_injection/injector.py
def inject_failure(task: Dict[str, Any], failure_config: Dict[str, Any]) -> Dict[str, Any]:
    failure_type = failure_config.get("type")
    
    if failure_type == FailureType.NEW_FAILURE_TYPE.value:
        param = failure_config.get("param")
        return inject_new_failure_type(task, param)
    # ... 其他类型 ...
```

### 示例：添加 CAUSAL_BREAK 类型

```python
def inject_causal_break(task: Dict[str, Any], break_step: int) -> Dict[str, Any]:
    """
    注入因果链断裂失败
    
    原理：在指定步骤后插入一个错误动作，破坏因果链
    
    例如：
        原始任务：["fill, Kettle", "heat, Kettle", "pour, Mug"]
        在步骤 1 后插入错误动作：
        失败任务：["fill, Kettle", "heat, Kettle", "wrong_action", "pour, Mug"]
        结果：因果链被破坏 → 失败场景
    """
    modified_task = task.copy()
    actions = modified_task.get("actions", []).copy()
    
    if 0 <= break_step < len(actions):
        # 在指定步骤后插入错误动作
        wrong_action = "wrong_action, Object"
        actions.insert(break_step + 1, wrong_action)
        modified_task["actions"] = actions
        modified_task["injected_failure"] = {
            "type": FailureType.CAUSAL_BREAK.value,
            "break_step": break_step,
            "inserted_action": wrong_action
        }
    
    return modified_task
```

---

## 集成到任务执行

### 在任务执行前注入失败

```python
def execute_task_with_failure_injection(task_def, failure_config):
    """执行带失败注入的任务"""
    
    # 1. 注入失败
    failed_task = inject_failure(task_def, failure_config)
    
    # 2. 执行任务（AI2THOR 或其他执行器）
    # 这里使用失败的任务定义
    events = execute_task(failed_task)
    
    # 3. 记录失败信息
    result = {
        "task": task_def["name"],
        "failure_injected": True,
        "failure_info": failed_task.get("injected_failure"),
        "events": events
    }
    
    return result
```

### 在状态级别注入失败

```python
def execute_task_with_state_failure(task_def, failure_config):
    """执行带状态级别失败注入的任务"""
    
    # 1. 正常执行任务
    events = execute_task(task_def)
    
    # 2. 从事件中提取状态
    state = extract_state_from_events(events)
    
    # 3. 在状态级别注入失败（如 PHYSICAL_IMPOSSIBLE）
    if failure_config["type"] == "PHYSICAL_IMPOSSIBLE":
        failed_state = inject_capacity_violation(state, failure_config["object"])
        # 使用失败状态进行后续分析
        result = analyze_with_failed_state(failed_state)
    
    return result
```

### 与 REFLECT 框架对比

REFLECT 框架的失败注入方式（参考 `utils/gen_data.py`）：

```python
# REFLECT 风格的失败注入
# 在任务执行过程中动态注入

if taskUtil.chosen_failure == 'missing_step':
    if idx in failure_injection_idx:
        # 跳过该步骤
        continue

elif taskUtil.chosen_failure == 'drop':
    if idx == failure_injection_idx:
        # 执行 drop 动作
        drop(taskUtil, failure_injection_idx)

elif taskUtil.chosen_failure == 'failed_action':
    if idx == failure_injection_idx:
        # 标记动作失败
        fail_execution = True
```

**我们的实现优势**：
- ✅ **配置驱动**：通过 JSON 配置文件，更易管理
- ✅ **提前注入**：在任务执行前注入，更清晰
- ✅ **可复现**：相同配置总是产生相同失败

---

## 配置文件格式

### injection_config.json

```json
{
  "make_coffee": [
    {
      "type": "MISSING_PRECONDITION",
      "step": 3,
      "description": "移除 put_in 步骤，导致杯子未被放入咖啡机"
    },
    {
      "type": "PHYSICAL_IMPOSSIBLE",
      "object": "coffee_machine",
      "description": "设置咖啡机容量违反物理约束"
    }
  ],
  "make_tea": [
    {
      "type": "MISSING_PRECONDITION",
      "step": 4,
      "description": "移除 toggle_on Faucet 步骤"
    }
  ]
}
```

### 配置字段说明

- `type`: 失败类型（必须）
- `step`: 步骤索引（MISSING_PRECONDITION 需要）
- `object`: 对象名称（PHYSICAL_IMPOSSIBLE 需要）
- `description`: 描述（可选，用于文档）

---

## 测试

### 单元测试示例

```python
import unittest
from failure_injection.injector import inject_missing_precondition, inject_failure
from failure_injection.failure_types import FailureType

class TestFailureInjection(unittest.TestCase):
    
    def test_missing_precondition(self):
        task = {
            "name": "test_task",
            "actions": ["action1", "action2", "action3"]
        }
        
        failed_task = inject_missing_precondition(task, step_index=1)
        
        self.assertEqual(len(failed_task["actions"]), 2)
        self.assertEqual(failed_task["actions"], ["action1", "action3"])
        self.assertIn("injected_failure", failed_task)
        self.assertEqual(failed_task["injected_failure"]["type"], "MISSING_PRECONDITION")
    
    def test_inject_failure_unified(self):
        task = {
            "name": "test_task",
            "actions": ["action1", "action2", "action3"]
        }
        
        failure_config = {
            "type": FailureType.MISSING_PRECONDITION.value,
            "step": 1
        }
        
        failed_task = inject_failure(task, failure_config)
        
        self.assertEqual(len(failed_task["actions"]), 2)
        self.assertIn("injected_failure", failed_task)

if __name__ == "__main__":
    unittest.main()
```

运行测试：

```bash
cd craft-experiments
python -m pytest failure_injection/ -v
```

---

## 总结

失败注入实现的关键点：

1. **类型安全**：使用 Enum 定义失败类型
2. **配置驱动**：通过 JSON 配置文件控制注入
3. **可扩展性**：易于添加新的失败类型
4. **可复现性**：相同配置总是产生相同失败
5. **记录信息**：保存注入的失败信息，便于后续分析

下一步：
- 集成到实际的任务执行流程
- 添加更多失败类型（CAUSAL_BREAK, PERCEPTION_NOISE）
- 实现状态级别的失败注入
- 添加验证和测试

