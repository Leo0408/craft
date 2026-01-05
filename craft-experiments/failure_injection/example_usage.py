#!/usr/bin/env python3
"""
Failure Injection 使用示例

演示如何使用失败注入功能
"""

import json
import sys
from pathlib import Path

# 添加当前目录到路径（以便导入同目录的模块）
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# 添加项目根目录到路径
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))

# 导入失败注入模块
from injector import (
    inject_failure,
    inject_missing_precondition,
    inject_capacity_violation
)
from failure_types import FailureType


def example_1_basic_usage():
    """示例 1: 基本使用"""
    print("=" * 60)
    print("示例 1: 基本使用 - MISSING_PRECONDITION")
    print("=" * 60)
    
    # 定义任务
    task = {
        "name": "make_coffee",
        "scene": "FloorPlan16",
        "actions": [
            "navigate_to_obj, Mug",
            "pick_up, Mug",
            "navigate_to_obj, CoffeeMachine",
            "put_in, CoffeeMachine, Mug",  # 步骤 3（索引从 0 开始）
            "toggle_on, CoffeeMachine"
        ]
    }
    
    print("\n原始任务:")
    print(f"  动作数: {len(task['actions'])}")
    for i, action in enumerate(task['actions']):
        print(f"    {i}. {action}")
    
    # 注入失败（移除步骤 3）
    failed_task = inject_missing_precondition(task, step_index=3)
    
    print("\n失败任务（移除步骤 3）:")
    print(f"  动作数: {len(failed_task['actions'])}")
    for i, action in enumerate(failed_task['actions']):
        print(f"    {i}. {action}")
    
    print("\n注入的失败信息:")
    print(f"  {json.dumps(failed_task.get('injected_failure'), indent=2)}")
    
    print("\n✅ 示例 1 完成\n")


def example_2_unified_interface():
    """示例 2: 使用统一接口"""
    print("=" * 60)
    print("示例 2: 使用统一接口 inject_failure()")
    print("=" * 60)
    
    # 加载任务定义
    task_defs_path = project_root / "craft-experiments" / "tasks" / "task_defs.json"
    if not task_defs_path.exists():
        print(f"⚠️  任务定义文件不存在: {task_defs_path}")
        print("   请先运行 day_one_checklist.py 生成文件")
        return
    
    with open(task_defs_path) as f:
        tasks = json.load(f)
    
    task = tasks["make_coffee"]
    
    print("\n任务:", task.get("name", "unknown"))
    print(f"原始动作数: {len(task['actions'])}")
    
    # 使用统一接口注入失败
    failure_config = {
        "type": FailureType.MISSING_PRECONDITION.value,
        "step": 3
    }
    
    failed_task = inject_failure(task, failure_config)
    
    print(f"失败动作数: {len(failed_task['actions'])}")
    print(f"移除的动作: {failed_task['injected_failure'].get('removed_action')}")
    
    print("\n✅ 示例 2 完成\n")


def example_3_from_config_file():
    """示例 3: 从配置文件加载"""
    print("=" * 60)
    print("示例 3: 从配置文件加载失败配置")
    print("=" * 60)
    
    # 加载任务定义
    task_defs_path = project_root / "craft-experiments" / "tasks" / "task_defs.json"
    config_path = current_dir / "injection_config.json"
    
    if not task_defs_path.exists() or not config_path.exists():
        print(f"⚠️  文件不存在")
        print(f"   task_defs: {task_defs_path.exists()}")
        print(f"   config: {config_path.exists()}")
        print("   请先运行 day_one_checklist.py 生成文件")
        return
    
    with open(task_defs_path) as f:
        tasks = json.load(f)
    
    with open(config_path) as f:
        failure_configs = json.load(f)
    
    print("\n从配置文件加载失败配置:")
    
    for task_name, configs in failure_configs.items():
        if task_name not in tasks:
            print(f"⚠️  任务 '{task_name}' 不存在于任务定义中")
            continue
        
        task = tasks[task_name]
        print(f"\n任务: {task_name}")
        print(f"  原始动作数: {len(task['actions'])}")
        
        for i, failure_config in enumerate(configs, 1):
            failed_task = inject_failure(task, failure_config)
            failure_type = failure_config.get("type")
            
            print(f"\n  失败配置 {i}: {failure_type}")
            
            if failure_type == FailureType.MISSING_PRECONDITION.value:
                print(f"    步骤索引: {failure_config.get('step')}")
                print(f"    失败动作数: {len(failed_task['actions'])}")
                print(f"    移除的动作: {failed_task['injected_failure'].get('removed_action')}")
            elif failure_type == FailureType.PHYSICAL_IMPOSSIBLE.value:
                print(f"    对象: {failure_config.get('object')}")
                print(f"    失败信息: {failed_task['injected_failure']}")
    
    print("\n✅ 示例 3 完成\n")


def example_4_physical_impossible():
    """示例 4: PHYSICAL_IMPOSSIBLE 类型"""
    print("=" * 60)
    print("示例 4: PHYSICAL_IMPOSSIBLE 类型 - 容量违反")
    print("=" * 60)
    
    # 定义状态
    state = {
        "coffee_machine": {
            "volume": 0,
            "max_capacity": 100,
            "contains": []
        },
        "mug": {
            "volume": 50,
            "max_capacity": 200,
            "contains": []
        }
    }
    
    print("\n原始状态:")
    print(f"  coffee_machine: {state['coffee_machine']}")
    
    # 注入容量违反失败
    failed_state = inject_capacity_violation(state, "coffee_machine")
    
    print("\n失败状态（容量违反）:")
    print(f"  coffee_machine: {failed_state['coffee_machine']}")
    print(f"  注入的失败信息: {failed_state.get('injected_failure')}")
    
    print("\n✅ 示例 4 完成\n")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("Failure Injection 使用示例")
    print("=" * 60)
    print()
    
    try:
        example_1_basic_usage()
        example_2_unified_interface()
        example_3_from_config_file()
        example_4_physical_impossible()
        
        print("=" * 60)
        print("✅ 所有示例运行完成!")
        print("=" * 60)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: 文件未找到 - {e}")
        print("请确保在正确的目录运行此脚本，或先运行 day_one_checklist.py")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
