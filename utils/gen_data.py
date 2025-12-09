"""
Data generation for AI2THOR simulation with failure injection
Adapted from REFLECT framework
"""

import os
import json
import numpy as np
import random
import pickle
from typing import Dict, List, Optional
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

from .task_utils import TaskUtil, closest_position
from .constants import TASK_DICT
from . import action_primitives as ap


def flatten_list(lis):
    """Flatten a nested list"""
    output = []
    for item in lis:
        if isinstance(item, list):
            output.extend(item)
        else:
            output.append(item)
    return output


def get_failure_injection_idx(taskUtil, actions, task, action_idxs, nav_idxs, 
                               interact_cnt=0, nav_cnt=0):
    """
    Get the index where to inject a failure
    
    Args:
        taskUtil: TaskUtil instance
        actions: List of action instructions
        task: Task configuration
        action_idxs: List of interaction action indices
        nav_idxs: List of navigation action indices
        interact_cnt: Current interaction counter
        nav_cnt: Current navigation counter
        
    Returns:
        Failure injection index, or -1 if unable to inject
    """
    counter = 0
    print(f"[INFO] Injected failures: {taskUtil.failures_already_injected}")
    
    try:
        while True:
            if taskUtil.chosen_failure == 'missing_step':
                if "specified_missing_steps" in task:
                    cnt = 0
                    for f in taskUtil.failures_already_injected:
                        if f[0] == 'missing_step':
                            cnt += 1
                    if cnt < len(task['specified_missing_steps']):
                        failure_injection_idx = task['specified_missing_steps'][cnt]
                        return failure_injection_idx
                
                failure_injection_idx = np.random.choice(action_idxs[interact_cnt:])
                if "toggle_off" in actions[failure_injection_idx] or "close_obj" in actions[failure_injection_idx]:
                    continue
                if len(taskUtil.failures_already_injected) == 0 or \
                    failure_injection_idx not in flatten_list([f[1] for f in taskUtil.failures_already_injected]):
                    return failure_injection_idx
                    
            elif taskUtil.chosen_failure == 'failed_action':
                failure_injection_idx = np.random.choice(action_idxs[interact_cnt:])
                if "toggle_off" in actions[failure_injection_idx] or "close_obj" in actions[failure_injection_idx]:
                    continue
                if len(taskUtil.failures_already_injected) == 0 or \
                    failure_injection_idx not in flatten_list([f[1] for f in taskUtil.failures_already_injected]):
                    return failure_injection_idx
                    
            elif taskUtil.chosen_failure == 'drop':
                failure_injection_idx = np.random.choice(nav_idxs[nav_cnt:])
                return failure_injection_idx
            
            if counter > 20:
                print(f"[INFO] Unable to inject a novel failure for failure type: {taskUtil.chosen_failure}. Choosing a new failure type")
                taskUtil.chosen_failure = np.random.choice(taskUtil.failures)
            if counter > 60:
                print("[INFO] Unable to inject a novel failure. Skipping this round.")
                return -1
            counter += 1
    except Exception as e:
        print(f"[INFO] Unable to inject a novel failure: {e}")
        return -1


def run_data_gen(data_path: str, task: Dict):
    """
    Run data generation for a task
    
    Args:
        data_path: Path to save data
        task: Task configuration dictionary
    """
    np.random.seed(91)
    random.seed(91)
    
    # Create directory for task
    os.makedirs(f'thor_tasks/{TASK_DICT[task["task_idx"]]}', exist_ok=True)
    pickle_path = f'thor_tasks/{TASK_DICT[task["task_idx"]]}/{task["folder_name"]}.pickle'
    with open(pickle_path, 'wb') as handle:
        pickle.dump([], handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    for i in range(int(task.get('num_samples', 1))):
        # Initialize AI2THOR Controller
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
        
        # Get reachable positions
        reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]
        
        # Get failure injection parameters
        chosen_failure = task.get('chosen_failure', None)
        failure_injection_params = task.get('failure_injection_params', None)
        
        # Initialize TaskUtil
        taskUtil = TaskUtil(
            folder_name=os.path.join(TASK_DICT[task["task_idx"]], task['folder_name']),
            controller=controller,
            reachable_positions=reachable_positions,
            failure_injection=task.get('failure_injection', False),
            index=i,
            repo_path=data_path,
            chosen_failure=chosen_failure,
            failure_injection_params=failure_injection_params
        )
        
        # Handle blocking/occupied failures (pre-place objects)
        if taskUtil.chosen_failure in ['blocking', 'occupied', 'occupied_put'] and 'failure_injection_params' in task:
            # This would call place_obj function to pre-place objects
            # For now, we skip this as it requires additional implementation
            pass
        
        # Execute preactions if specified
        if "preactions" in task:
            for preaction_instr in task['preactions']:
                lis = preaction_instr.split(',')
                lis = [item.strip("() ") for item in lis]
                preaction = lis[0]
                params = lis[1:]
                func = getattr(ap, preaction, None)
                if func:
                    retval = func(taskUtil, *params)
        
        # Parse action instructions
        instrs, new_instrs = [], []
        action_idxs, nav_idxs = [], []
        
        for idx, instr in enumerate(task['actions']):
            instrs.append(instr)
            lis = instr.split(',')
            lis = [item.strip("() ") for item in lis]
            action = lis[0]
            
            if action in taskUtil.interact_action_primitives:
                action_idxs.append(idx)
            if 'navigate_to_obj' == action:
                nav_idxs.append(idx)
        
        # Get failure injection index
        failure_injection_idx = None
        if task.get('failure_injection', False):
            failure_injection_idx = get_failure_injection_idx(
                taskUtil, instrs, task, action_idxs, nav_idxs
            )
            if failure_injection_idx == -1:
                print("[INFO] Skipping this sample due to failure injection issues")
                controller.stop()
                continue
            print(f"[INFO] Failure injection index: {failure_injection_idx}")
        
        # Execute actions
        nav_counter = 0
        interact_counter = 0
        
        for idx, instr in enumerate(instrs):
            lis = instr.split(',')
            lis = [item.strip("() ") for item in lis]
            action = lis[0]
            params = lis[1:]
            
            # Get action function
            func = getattr(ap, action, None)
            if func is None:
                print(f"[WARNING] Action '{action}' not found in action_primitives")
                continue
            
            # Update counters
            if action in taskUtil.interact_action_primitives:
                interact_counter += 1
            if 'navigate_to_obj' == action:
                nav_counter += 1
            
            # Handle drop failure injection
            to_drop = False
            if (not taskUtil.failure_added and taskUtil.chosen_failure == 'drop' 
                and idx == failure_injection_idx):
                to_drop = True
                params.append(to_drop)
                params.append(failure_injection_idx)
            
            # Handle missing step failure injection
            if (not taskUtil.failure_added and taskUtil.chosen_failure == 'missing_step' 
                and action in taskUtil.interact_action_primitives):
                if not isinstance(failure_injection_idx, list):
                    failure_injection_idx = [failure_injection_idx]
                if idx in failure_injection_idx:
                    if 'gt_failure_reason' in taskUtil.gt_failure:
                        taskUtil.gt_failure['gt_failure_reason'] += ', ' + instr
                    else:
                        taskUtil.gt_failure['gt_failure_reason'] = 'Missing ' + instr
                    taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter + 1
                    if idx == failure_injection_idx[-1]:
                        taskUtil.failure_added = True
                        taskUtil.failures_already_injected.append([taskUtil.chosen_failure, failure_injection_idx])
                    else:
                        taskUtil.failure_added = False
                    continue
            
            # Handle failed action failure injection
            fail_execution = False
            if (not taskUtil.failure_added and taskUtil.chosen_failure == 'failed_action' 
                and action in taskUtil.interact_action_primitives and idx == failure_injection_idx):
                print("[INFO] Injecting failed action...")
                fail_execution = True
                taskUtil.gt_failure['gt_failure_reason'] = 'Failed to successfully execute ' + instr
                taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter + 1
                taskUtil.failures_already_injected.append([taskUtil.chosen_failure, failure_injection_idx])
                taskUtil.failure_added = True
                params.append(fail_execution)
            
            # Execute action
            new_instrs.append(instr)
            try:
                retval = func(taskUtil, *params)
                
                # If drop failure was not successfully injected, find a different failure instance
                if retval == False and taskUtil.chosen_failure == 'drop':
                    failure_injection_idx = get_failure_injection_idx(
                        taskUtil, instrs, task, action_idxs, nav_idxs,
                        interact_cnt=interact_counter, nav_cnt=nav_counter
                    )
                    if failure_injection_idx == -1:
                        break
            except Exception as e:
                print(f"[ERROR] Error executing action {action}: {e}")
                continue
        
        # Add buffer frames at the end
        for _ in range(2):
            e = controller.step(action="Done")
            ap.save_data(taskUtil, e)
        
        print(f"[INFO] Interact actions: {taskUtil.interact_actions}")
        print(f"[INFO] Nav actions: {taskUtil.nav_actions}")
        
        # Save action data
        os.makedirs(f'thor_tasks/{taskUtil.specific_folder_name}', exist_ok=True)
        with open(f'thor_tasks/{taskUtil.specific_folder_name}/interact_actions.pickle', 'wb') as handle:
            pickle.dump(taskUtil.interact_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(f'thor_tasks/{taskUtil.specific_folder_name}/nav_actions.pickle', 'wb') as handle:
            pickle.dump(taskUtil.nav_actions, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        with open(pickle_path, 'wb') as handle:
            pickle.dump(taskUtil.failures_already_injected, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Save task configuration
        updated_task = task.copy()
        updated_task['specific_folder_name'] = taskUtil.specific_folder_name
        
        if 'gt_failure_reason' not in taskUtil.gt_failure:
            taskUtil.gt_failure['gt_failure_reason'] = 'No failure added'
            taskUtil.gt_failure['gt_failure_step'] = 0
        
        if 'gt_failure_reason' not in updated_task:
            updated_task['gt_failure_reason'] = taskUtil.gt_failure['gt_failure_reason']
            updated_task['gt_failure_step'] = taskUtil.gt_failure['gt_failure_step']
        
        updated_task['unity_name_map'] = taskUtil.unity_name_map
        updated_task['actions'] = new_instrs
        
        with open(f'thor_tasks/{taskUtil.specific_folder_name}/task.json', 'w') as f:
            json.dump(updated_task, f, indent=2)
        
        # Stop controller
        controller.stop()
        
        print(f"[INFO] Completed data generation for sample {i+1}")


if __name__ == "__main__":
    # Example usage
    task = {
        "name": "make coffee",
        "task_idx": 5,
        "num_samples": 1,
        "failure_injection": True,
        "folder_name": "makeCoffee-1",
        "scene": "FloorPlan16",
        "actions": [
            "navigate_to_obj, Mug",
            "pick_up, Mug",
            "navigate_to_obj, Sink",
            "put_on, SinkBasin, Mug",
            "toggle_on, Faucet",
            "toggle_off, Faucet",
            "pick_up, Mug",
            "navigate_to_obj, CoffeeMachine",
            "put_in, CoffeeMachine, Mug",
        ]
    }
    
    run_data_gen(data_path=".", task=task)

