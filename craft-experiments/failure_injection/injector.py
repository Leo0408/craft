"""
Failure Injection Logic

实现 2 种 failure injection：
1. MISSING_PRECONDITION: 移除前置条件步骤
2. PHYSICAL_IMPOSSIBLE: 注入物理不可能的状态
"""
from typing import Dict, List, Any
from .failure_types import FailureType


def inject_missing_precondition(task: Dict[str, Any], step_index: int) -> Dict[str, Any]:
    """
    注入前置条件缺失失败
    
    例如：移除 put_in 步骤，导致后续步骤失败
    """
    modified_task = task.copy()
    actions = modified_task.get("actions", []).copy()
    
    if 0 <= step_index < len(actions):
        # 移除指定步骤
        removed_action = actions.pop(step_index)
        modified_task["actions"] = actions
        modified_task["injected_failure"] = {
            "type": FailureType.MISSING_PRECONDITION.value,
            "step": step_index,
            "removed_action": removed_action
        }
    
    return modified_task


def inject_capacity_violation(state: Dict[str, Any], object_name: str) -> Dict[str, Any]:
    """
    注入容量违反失败（物理不可能）
    
    例如：设置容器容量为负值或超出限制
    """
    modified_state = state.copy()
    if object_name in modified_state:
        obj_state = modified_state[object_name].copy()
        # 设置容器容量为不可能的值
        obj_state["volume"] = 1000  # 超大的容量
        obj_state["max_capacity"] = 100  # 但最大容量很小
        obj_state["contains"] = []  # 但实际上为空
        modified_state[object_name] = obj_state
        modified_state["injected_failure"] = {
            "type": FailureType.PHYSICAL_IMPOSSIBLE.value,
            "object": object_name,
            "reason": "Capacity violation"
        }
    
    return modified_state


def inject_failure(task: Dict[str, Any], failure_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据配置注入失败
    
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
