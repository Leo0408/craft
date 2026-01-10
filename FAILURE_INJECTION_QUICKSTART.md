# Failure Injection 快速开始指南

本文档快速说明如何实现和使用 Step 3 的失败注入功能。

---

## 📁 文件结构

```
craft-experiments/failure_injection/
├── failure_types.py          # 失败类型枚举
├── injector.py               # 注入逻辑实现
├── injection_config.json     # 失败配置（JSON格式）
├── IMPLEMENTATION_GUIDE.md   # 详细实现指南
└── test_injection.py         # 简单测试脚本
```

---

## 🚀 快速使用

### 1. 基本使用 - MISSING_PRECONDITION

```python
from failure_injection.injector import inject_missing_precondition

# 定义任务
task = {
    "name": "make_coffee",
    "actions": [
        "navigate_to_obj, Mug",
        "pick_up, Mug",
        "navigate_to_obj, CoffeeMachine",
        "put_in, CoffeeMachine, Mug",  # 步骤 3
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

### 2. 使用统一接口

```python
from failure_injection.injector import inject_failure
from failure_injection.failure_types import FailureType

# 定义失败配置
failure_config = {
    "type": FailureType.MISSING_PRECONDITION.value,
    "step": 3
}

# 注入失败
failed_task = inject_failure(task, failure_config)
```

### 3. 从配置文件加载

```python
import json
from failure_injection.injector import inject_failure

# 加载任务定义
with open("tasks/task_defs.json") as f:
    tasks = json.load(f)

# 加载失败配置
with open("failure_injection/injection_config.json") as f:
    failure_configs = json.load(f)

# 对每个任务的每个失败配置注入失败
for task_name, configs in failure_configs.items():
    task = tasks[task_name]
    for failure_config in configs:
        failed_task = inject_failure(task, failure_config)
        print(f"任务: {task_name}, 失败类型: {failure_config['type']}")
```

---

## 📋 失败类型说明

### 1. MISSING_PRECONDITION（前置条件缺失）

**原理**：移除任务中的关键步骤

**配置格式**：
```json
{
    "type": "MISSING_PRECONDITION",
    "step": 3
}
```

**效果**：
- 移除指定步骤的动作
- 后续步骤仍会执行
- 导致前置条件缺失，任务失败

**示例**：
- 原始任务：`["pick_up, Mug", "put_in, CoffeeMachine, Mug", "toggle_on, CoffeeMachine"]`
- 移除步骤 1：`["pick_up, Mug", "toggle_on, CoffeeMachine"]`
- 结果：杯子没有被放入咖啡机，但开启咖啡机的步骤仍会执行 → **失败场景**

### 2. PHYSICAL_IMPOSSIBLE（物理不可能）

**原理**：设置违反物理约束的状态

**配置格式**：
```json
{
    "type": "PHYSICAL_IMPOSSIBLE",
    "object": "coffee_machine"
}
```

**效果**：
- 设置对象的物理属性为不可能的值
- 例如：容量为 1000，但最大容量为 100
- 违反物理约束 → **失败场景**

---

## 🔧 实现细节

### 核心函数

#### `inject_missing_precondition(task, step_index)`

```python
def inject_missing_precondition(task: Dict[str, Any], step_index: int) -> Dict[str, Any]:
    """注入前置条件缺失失败"""
    modified_task = task.copy()
    actions = modified_task.get("actions", []).copy()
    
    if 0 <= step_index < len(actions):
        removed_action = actions.pop(step_index)
        modified_task["actions"] = actions
        modified_task["injected_failure"] = {
            "type": FailureType.MISSING_PRECONDITION.value,
            "step": step_index,
            "removed_action": removed_action
        }
    
    return modified_task
```

#### `inject_capacity_violation(state, object_name)`

```python
def inject_capacity_violation(state: Dict[str, Any], object_name: str) -> Dict[str, Any]:
    """注入容量违反失败"""
    modified_state = state.copy()
    if object_name in modified_state:
        obj_state = modified_state[object_name].copy()
        obj_state["volume"] = 1000      # 超大的容量
        obj_state["max_capacity"] = 100 # 但最大容量很小
        modified_state[object_name] = obj_state
        modified_state["injected_failure"] = {
            "type": FailureType.PHYSICAL_IMPOSSIBLE.value,
            "object": object_name,
            "reason": "Capacity violation"
        }
    return modified_state
```

#### `inject_failure(task, failure_config)` - 统一接口

```python
def inject_failure(task: Dict[str, Any], failure_config: Dict[str, Any]) -> Dict[str, Any]:
    """根据配置注入失败（统一入口）"""
    failure_type = failure_config.get("type")
    
    if failure_type == FailureType.MISSING_PRECONDITION.value:
        step = failure_config.get("step", 0)
        return inject_missing_precondition(task, step)
    elif failure_type == FailureType.PHYSICAL_IMPOSSIBLE.value:
        task = task.copy()
        task["injected_failure"] = failure_config
        return task
    else:
        raise ValueError(f"Unknown failure type: {failure_type}")
```

---

## 📝 配置文件格式

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

---

## 🧪 测试

运行简单测试：

```bash
cd craft-experiments/failure_injection
python3 test_injection.py
```

或查看详细实现指南：

```bash
cat craft-experiments/failure_injection/IMPLEMENTATION_GUIDE.md
```

---

## 🔄 集成到任务执行

### 在任务执行前注入失败

```python
from failure_injection.injector import inject_failure

def execute_task_with_failure(task_def, failure_config):
    """执行带失败注入的任务"""
    
    # 1. 注入失败
    failed_task = inject_failure(task_def, failure_config)
    
    # 2. 执行任务（使用失败的任务定义）
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

---

## 📚 相关文档

- [IMPLEMENTATION_GUIDE.md](craft-experiments/failure_injection/IMPLEMENTATION_GUIDE.md): 详细实现指南
- [Experiment.md](Experiment.md): 实验设计文档
- [EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md): 实验详细指南

---

## 💡 关键点总结

1. **类型安全**：使用 Enum 定义失败类型
2. **配置驱动**：通过 JSON 配置文件控制注入
3. **可扩展性**：易于添加新的失败类型
4. **可复现性**：相同配置总是产生相同失败
5. **记录信息**：保存注入的失败信息，便于后续分析

---

## ❓ 常见问题

### Q: 如何添加新的失败类型？

A: 
1. 在 `failure_types.py` 中添加新类型
2. 在 `injector.py` 中实现注入函数
3. 在 `inject_failure()` 中添加处理逻辑

### Q: 失败注入会影响原始任务吗？

A: 不会。所有注入函数都使用 `copy()` 创建副本，不会修改原始任务。

### Q: 如何验证失败注入是否正确？

A: 检查返回的任务是否包含 `injected_failure` 字段，并验证字段内容是否符合预期。



