#!/usr/bin/env python3
"""
Failure Injection 简单测试脚本

演示失败注入的基本用法
"""

import json
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

# 直接导入（不使用相对导入）
try:
    from failure_types import FailureType
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from craft_experiments.failure_injection.failure_types import FailureType

# 直接实现简单的注入函数（避免导入问题）
def inject_missing_precondition(task, step_index):
    """注入前置条件缺失失败"""
    modified_task = task.copy()
    actions = modified_task.get("actions", []).copy()
    
    if 0 <= step_index < len(actions):
        removed_action = actions.pop(step_index)
        modified_task["actions"] = actions
        modified_task["injected_failure"] = {
            "type": "MISSING_PRECONDITION",
            "step": step_index,
            "removed_action": removed_action
        }
    
    return modified_task


def inject_capacity_violation(state, object_name):
    """注入容量违反失败"""
    modified_state = state.copy()
    if object_name in modified_state:
        obj_state = modified_state[object_name].copy()
        obj_state["volume"] = 1000
        obj_state["max_capacity"] = 100
        obj_state["contains"] = []
        modified_state[object_name] = obj_state
        modified_state["injected_failure"] = {
            "type": "PHYSICAL_IMPOSSIBLE",
            "object": object_name,
            "reason": "Capacity violation"
        }
    return modified_state


def main():
    print("=" * 60)
    print("Failure Injection 测试")
    print("=" * 60)
    
    # 示例 1: MISSING_PRECONDITION
    print("\n1. MISSING_PRECONDITION 示例")
    print("-" * 60)
    
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
    
    print("\n原始任务:")
    for i, action in enumerate(task['actions']):
        print(f"  {i}. {action}")
    
    failed_task = inject_missing_precondition(task, step_index=3)
    
    print("\n失败任务（移除步骤 3）:")
    for i, action in enumerate(failed_task['actions']):
        print(f"  {i}. {action}")
    
    print("\n注入的失败信息:")
    print(f"  {json.dumps(failed_task.get('injected_failure'), indent=2)}")
    
    # 示例 2: PHYSICAL_IMPOSSIBLE
    print("\n\n2. PHYSICAL_IMPOSSIBLE 示例")
    print("-" * 60)
    
    state = {
        "coffee_machine": {
            "volume": 0,
            "max_capacity": 100,
            "contains": []
        }
    }
    
    print("\n原始状态:")
    print(f"  {state['coffee_machine']}")
    
    failed_state = inject_capacity_violation(state, "coffee_machine")
    
    print("\n失败状态:")
    print(f"  {failed_state['coffee_machine']}")
    print(f"\n注入的失败信息:")
    print(f"  {json.dumps(failed_state.get('injected_failure'), indent=2)}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()



