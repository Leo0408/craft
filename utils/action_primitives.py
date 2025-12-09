"""
Action primitives for AI2THOR task execution
Simplified version adapted from REFLECT framework
"""

import time
import math
import numpy as np
from typing import Optional
from .task_utils import closest_position, find_path
from .constants import NAME_MAP, OBJ_UNSLICED_MAP, OBJ_SLICED_MAP


def look_at(taskUtil, target_pos: dict, robot_pos: dict, center_to_camera_disp: float = 0.6, replan: bool = False):
    """
    Make robot look at target position
    
    Args:
        taskUtil: TaskUtil instance
        target_pos: Target position dict with 'x', 'y', 'z'
        robot_pos: Robot position dict
        center_to_camera_disp: Camera displacement from robot center
        replan: Whether this is a replan action
    """
    robot_y = robot_pos["y"] + center_to_camera_disp
    yaw = np.arctan2(target_pos["x"] - robot_pos["x"], target_pos["z"] - robot_pos["z"])
    yaw = math.degrees(yaw)
    
    tilt = -np.arctan2(
        target_pos["y"] - robot_y,
        np.sqrt((target_pos["z"] - robot_pos["z"])**2 + (target_pos["x"] - robot_pos["x"])**2)
    )
    tilt = np.round(np.degrees(tilt), 1)
    org_tilt = taskUtil.controller.last_event.metadata["agent"]["cameraHorizon"]
    final_tilt = tilt - org_tilt
    
    if tilt > 60:
        final_tilt = 60
    if tilt < -30:
        final_tilt = -30
    final_tilt = np.round(final_tilt, 1)
    
    # Rotate robot to face object
    event = taskUtil.controller.step(
        action="Teleport",
        **robot_pos,
        rotation=dict(x=0, y=yaw, z=0),
        forceAction=True
    )
    
    # Adjust camera tilt
    if final_tilt > 0:
        event = taskUtil.controller.step(action="LookDown", degrees=final_tilt)
    elif final_tilt < 0:
        event = taskUtil.controller.step(action="LookUp", degrees=-final_tilt)
    
    return event


def navigate_to_obj(taskUtil, obj_type: str, to_drop: bool = False, 
                    failure_injection_idx: int = 0, obj_id: Optional[str] = None, 
                    replan: bool = False, fail_execution: bool = False):
    """
    Navigate to an object using BFS path planning
    
    Args:
        taskUtil: TaskUtil instance
        obj_type: Object type to navigate to
        to_drop: Whether to inject drop failure
        failure_injection_idx: Failure injection index
        obj_id: Specific object ID (optional)
        replan: Whether this is a replan action
        fail_execution: Whether to fail execution
        
    Returns:
        True if successful, False otherwise, or drop_failure_injected if to_drop
    """
    print(f"[INFO] Execute action: Navigate to {obj_type}")
    
    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        start_frame = taskUtil.counter + 1
        taskUtil.nav_actions[(start_frame, start_frame)] = f'Move to {obj_type.lower()}'
        return False
    
    # Find object
    if obj_id is not None:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                  if obj["objectId"] == obj_id)
    elif '-' in obj_type:
        # Handle objects with multiple instances (e.g., CounterTop-1)
        for key, val in taskUtil.unity_name_map.items():
            if val == obj_type:
                obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                          if obj["name"] == key)
                break
    else:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                  if obj["objectType"] == obj_type)
    
    # Find closest reachable position
    closest_pos = closest_position(obj["position"], taskUtil.reachable_positions)
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    target_pos_val = [closest_pos['x'], closest_pos['z']]
    
    # Calculate grid indices
    taskUtil.grid_size = taskUtil.grid.shape[0]
    robot_x, robot_y = None, None
    target_x, target_y = None, None
    
    for row in range(taskUtil.grid.shape[0]):
        for col in range(taskUtil.grid.shape[1]):
            if [round(robot_pos['x'], 2), round(robot_pos['z'], 2)] == [taskUtil.grid[row, col, 0], taskUtil.grid[row, col, 1]]:
                robot_x = row
                robot_y = col
            if [round(target_pos_val[0], 2), round(target_pos_val[1], 2)] == [taskUtil.grid[row, col, 0], taskUtil.grid[row, col, 1]]:
                target_x = row
                target_y = col
    
    if robot_x is None or robot_y is None or target_x is None or target_y is None:
        print("[ERROR] Could not find robot or target position in grid")
        save_data(taskUtil, taskUtil.controller.last_event, replan=replan)
        taskUtil.controller.step(action="Done")
        return False
    
    target_pos = [target_x, target_y]
    path = find_path(taskUtil.grid, x=robot_x, y=robot_y, target_pos=target_pos, 
                     reachable_points=taskUtil.reachable_points)
    
    if path is None:
        print("[ERROR] No valid path found from robot to target object")
        save_data(taskUtil, taskUtil.controller.last_event, replan=replan)
        taskUtil.controller.step(action="Done")
        return False
    
    # Execute path
    start_frame = taskUtil.counter + 1
    drop_failure_injected = False
    
    for p in path:
        x = taskUtil.grid[p.x, p.y][0]
        z = taskUtil.grid[p.x, p.y][1]
        y = 0.9
        
        e = taskUtil.controller.step(
            action="Teleport",
            position=dict(x=x, y=y, z=z),
            forceAction=True,
            standing=True
        )
        save_data(taskUtil, e, replan=replan)
        taskUtil.controller.step(action="Done")
        
        # Check for drop failure injection
        obj_in_hand = False
        for o in taskUtil.controller.last_event.metadata['objects']:
            if o['isPickedUp']:
                obj_in_hand = True
                break
        
        if (not taskUtil.failure_added and taskUtil.chosen_failure == 'drop' 
            and obj_in_hand and to_drop):
            add_failure = np.random.uniform()
            if add_failure > 0.5:
                print(f"[INFO] Injected drop at step {taskUtil.counter}")
                drop(taskUtil, failure_injection_idx)
                drop_failure_injected = True
    
    # Look at object after navigation
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    look_at(taskUtil, target_pos=obj["position"], robot_pos=robot_pos, replan=replan)
    end_frame = taskUtil.counter
    taskUtil.nav_actions[(start_frame, end_frame)] = f'Move to {obj_type.lower()}'
    taskUtil.controller.step(action="Done")
    
    if to_drop:
        return drop_failure_injected
    else:
        return True


def pick_up(taskUtil, obj_type: str, fail_execution: bool = False, replan: bool = False):
    """
    Pick up an object
    
    Args:
        taskUtil: TaskUtil instance
        obj_type: Object type to pick up
        fail_execution: Whether to fail execution
        replan: Whether this is a replan action
    """
    print(f"[INFO] Execute action: Picking up {obj_type}")
    
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]
    
    # Handle sliced/unsliced objects
    obj_types = sorted([obj["objectType"] for obj in taskUtil.controller.last_event.metadata["objects"]])
    if obj_type in OBJ_UNSLICED_MAP:
        obj_unsliced_type = OBJ_UNSLICED_MAP[obj_type]
        if obj_unsliced_type in obj_types and obj_type not in obj_types:
            obj_type = obj_unsliced_type
    elif obj_type in OBJ_SLICED_MAP:
        obj_sliced_type = OBJ_SLICED_MAP[obj_type]
        if obj_sliced_type in obj_types:
            obj_type = obj_sliced_type
    
    e = taskUtil.controller.last_event
    objs = [obj for obj in taskUtil.controller.last_event.metadata["objects"] 
            if obj["objectType"] == obj_type]
    
    # Handle special cases (sliced objects)
    if 'LettuceSliced' in obj_type or 'AppleSliced' in obj_type:
        objs = sorted(objs, key=lambda x: int(x['name'].split('_')[-1])*-1)
    if 'PotatoSliced' in obj_type:
        objs = objs[2:]
    if 'TomatoSliced' in obj_type:
        objs = objs[4:]
    if "EggCracked" in obj_type:
        objs = []
    
    if fail_execution or len(objs) == 0:
        if len(objs) == 0:
            print("Cannot find the target object to pick up")
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = f"Pick up {obj_type_in_sim.lower()}"
        return
    
    # Navigate if object not visible
    if not objs[0]['visible'] and objs[0]['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, objs[0]['objectType'], replan=replan)
    
    # Pick up object
    for obj in objs:
        obj_id = obj['objectId']
        obj_pos = obj['position']
        
        # Look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj_pos, robot_pos=robot_pos, replan=replan)
        
        e = taskUtil.controller.step(
            action="PickupObject",
            objectId=obj_id,
            forceAction=False,
            manualInteract=False
        )
        
        if e.metadata['lastActionSuccess']:
            break
    
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = f"Pick up {obj_type_in_sim.lower()}"
    taskUtil.controller.step(action="Done")
    time.sleep(1)


def put_in(taskUtil, src_obj_type: str, target_obj_type: str, 
           fail_execution: bool = False, replan: bool = False):
    """
    Put an object inside another object
    
    Args:
        taskUtil: TaskUtil instance
        src_obj_type: Source object type (object being held)
        target_obj_type: Target object type (receptacle)
        fail_execution: Whether to fail execution
        replan: Whether this is a replan action
    """
    print(f"[INFO] Execute action: Putting {src_obj_type} in {target_obj_type}")
    
    src_obj_type_in_sim = src_obj_type
    if src_obj_type in NAME_MAP:
        src_obj_type_in_sim = NAME_MAP[src_obj_type]
    target_obj_type_in_sim = target_obj_type
    if target_obj_type in NAME_MAP:
        target_obj_type_in_sim = NAME_MAP[target_obj_type]
    
    # Find source object (object being held)
    src_obj = None
    for obj in taskUtil.controller.last_event.metadata["objects"]:
        if obj['isPickedUp']:
            src_obj = obj
            break
    
    if src_obj is None:
        print("The robot is not holding anything")
    elif src_obj['objectType'] != src_obj_type:
        print(f"The robot is not holding {src_obj_type}")
    else:
        print(f"The robot is holding: {src_obj['objectId']}, {src_obj['objectType']}")
    
    if fail_execution or src_obj is None:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = f"Put {src_obj_type_in_sim.lower()} inside {target_obj_type_in_sim.lower()}"
        return
    
    # Handle Sink -> SinkBasin
    if target_obj_type == 'Sink':
        target_obj_type = 'SinkBasin'
    
    # Find target object
    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == target_obj_type:
            target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                            if obj["name"] == obj_unity_name)
            found_obj = True
            break
    
    if not found_obj:
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                         if obj["objectType"] == target_obj_type)
    
    target_obj_id = target_obj['objectId']
    target_obj_pos = target_obj['position']
    
    # Navigate if needed
    if not target_obj['visible'] and target_obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, target_obj['objectType'], obj_id=target_obj['objectId'], replan=replan)
    
    # Look at target object
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    look_at(taskUtil, target_pos=target_obj_pos, robot_pos=robot_pos, replan=replan)
    
    # Check if receptacle is already occupied (e.g., Microwave)
    if target_obj_type == 'Microwave' and len(target_obj['receptacleObjectIds']) > 0:
        print(f"Microwave already contains an object: {target_obj['receptacleObjectIds']}")
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = f"Put {src_obj_type_in_sim.lower()} inside {target_obj_type_in_sim.lower()}"
        return
    
    # Put object
    if src_obj:
        e = taskUtil.controller.step(
            action="PutObject",
            objectId=target_obj_id,
            forceAction=False,
            placeStationary=True
        )
        
        if e.metadata['lastActionSuccess']:
            save_data(taskUtil, e, replan=replan)
            taskUtil.controller.step(action="Done")
            time.sleep(1)
        else:
            print("PutObject did not work")
            save_data(taskUtil, e, replan=replan)
    
    taskUtil.interact_actions[taskUtil.counter] = f"Put {src_obj_type_in_sim.lower()} inside {target_obj_type_in_sim.lower()}"


def put_on(taskUtil, src_obj_type: str, target_obj_type: str, 
           fail_execution: bool = False, target_obj_id: Optional[str] = None, replan: bool = False):
    """
    Put an object on top of another object
    
    Args:
        taskUtil: TaskUtil instance
        src_obj_type: Source object type (object being held)
        target_obj_type: Target object type (surface)
        fail_execution: Whether to fail execution
        target_obj_id: Specific target object ID (optional)
        replan: Whether this is a replan action
    """
    print(f"[INFO] Execute action: Putting {src_obj_type} on {target_obj_type}")
    
    src_obj_type_in_sim = src_obj_type
    if src_obj_type in NAME_MAP:
        src_obj_type_in_sim = NAME_MAP[src_obj_type]
    target_obj_type_in_sim = target_obj_type
    if target_obj_type in NAME_MAP:
        target_obj_type_in_sim = NAME_MAP[target_obj_type]
    
    # Find source object
    src_obj = None
    for obj in taskUtil.controller.last_event.metadata["objects"]:
        if obj['isPickedUp']:
            src_obj = obj
            break
    
    if src_obj is None or src_obj['objectType'] != src_obj_type:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = f"Put {src_obj_type_in_sim.lower()} on {target_obj_type_in_sim.lower()}"
        return
    
    # Find target object
    if target_obj_id is None:
        if "-" in target_obj_type and target_obj_type.split("-")[0] in ['StoveBurner', 'CounterTop']:
            for key, val in taskUtil.unity_name_map.items():
                if val == target_obj_type:
                    target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                                     if obj["name"] == key)
                    break
        else:
            target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                             if obj["objectType"] == target_obj_type)
    else:
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                         if obj["objectId"] == target_obj_id)
    
    target_obj_id = target_obj['objectId']
    target_obj_pos = target_obj['position']
    
    # Navigate if needed
    if target_obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, target_obj['objectType'], replan=replan)
    
    # Look at target object
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    look_at(taskUtil, target_pos=target_obj_pos, robot_pos=robot_pos, replan=replan)
    
    # Put object
    e = taskUtil.controller.step(
        action="PutObject",
        objectId=target_obj_id,
        forceAction=False,
        placeStationary=True
    )
    
    # Try standing if not successful
    if not e.metadata['lastActionSuccess']:
        taskUtil.controller.step(action="Stand")
        e = taskUtil.controller.step(
            action="PutObject",
            objectId=target_obj_id,
            forceAction=False,
            placeStationary=True
        )
    
    taskUtil.controller.step(action="Done")
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = f"Put {src_obj_type_in_sim.lower()} on {target_obj_type_in_sim.lower()}"
    time.sleep(1)


def toggle_on(taskUtil, obj_type: str, fail_execution: bool = False, replan: bool = False):
    """
    Toggle an object on
    
    Args:
        taskUtil: TaskUtil instance
        obj_type: Object type to toggle
        fail_execution: Whether to fail execution
        replan: Whether this is a replan action
    """
    print(f"[INFO] Execute action: Toggling on {obj_type}")
    
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]
    
    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = f"Toggle on {obj_type_in_sim.lower()}"
        return
    
    # Find object
    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == obj_type:
            obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                      if obj["name"] == obj_unity_name)
            found_obj = True
            break
    
    if not found_obj:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                  if obj["objectType"] == obj_type)
    
    obj_id = obj['objectId']
    
    # Navigate if needed
    if not obj['visible'] and obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, obj['objectType'], obj_id=obj['objectId'], replan=replan)
    else:
        # Look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj['position'], robot_pos=robot_pos, replan=replan)
    
    # Handle StoveBurner -> StoveKnob
    if "StoveBurner" in obj_type:
        for o in taskUtil.controller.last_event.metadata["objects"]:
            if 'StoveKnob' in o['objectType'] and o['controlledObjects'] is not None \
                    and o['controlledObjects'][0] == obj_id:
                obj_id = o['objectId']
    
    # Toggle on
    e = taskUtil.controller.step(
        action="ToggleObjectOn",
        objectId=obj_id,
        forceAction=True
    )
    
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = f"Toggle on {obj_type_in_sim.lower()}"
    taskUtil.controller.step(action="Done")
    time.sleep(1)


def toggle_off(taskUtil, obj_type: str, fail_execution: bool = False, replan: bool = False):
    """
    Toggle an object off
    
    Args:
        taskUtil: TaskUtil instance
        obj_type: Object type to toggle
        fail_execution: Whether to fail execution
        replan: Whether this is a replan action
    """
    print(f"[INFO] Execute action: Toggling off {obj_type}")
    
    obj_type_in_sim = obj_type
    if obj_type in NAME_MAP:
        obj_type_in_sim = NAME_MAP[obj_type]
    
    if fail_execution:
        e = taskUtil.controller.last_event
        save_data(taskUtil, e, replan=replan)
        taskUtil.interact_actions[taskUtil.counter] = f"Toggle off {obj_type_in_sim.lower()}"
        return
    
    # Find object
    found_obj = False
    for obj_unity_name, v in taskUtil.unity_name_map.items():
        if v == obj_type:
            obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                      if obj["name"] == obj_unity_name)
            found_obj = True
            break
    
    if not found_obj:
        obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                  if obj["objectType"] == obj_type)
    
    obj_id = obj['objectId']
    
    # Navigate if needed
    if not obj['visible'] and obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, obj['objectType'], obj_id=obj['objectId'], replan=replan)
    else:
        # Look at object
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj['position'], robot_pos=robot_pos, replan=replan)
    
    # Handle StoveBurner -> StoveKnob
    if "StoveBurner" in obj_type:
        for o in taskUtil.controller.last_event.metadata["objects"]:
            if 'StoveKnob' in o['objectType'] and o['controlledObjects'] is not None \
                    and o['controlledObjects'][0] == obj_id:
                obj_id = o['objectId']
    
    # Toggle off
    e = taskUtil.controller.step(
        action="ToggleObjectOff",
        objectId=obj_id,
        forceAction=True
    )
    
    save_data(taskUtil, e, replan=replan)
    taskUtil.interact_actions[taskUtil.counter] = f"Toggle off {obj_type_in_sim.lower()}"
    taskUtil.controller.step(action="Done")
    time.sleep(1)


def drop(taskUtil, failure_injection_idx: int):
    """
    Drop the object being held (for failure injection)
    
    Args:
        taskUtil: TaskUtil instance
        failure_injection_idx: Failure injection index
    """
    print(f"[INFO] Dropping object at step {taskUtil.counter}")
    e = taskUtil.controller.step(action="DropHandObject", forceAction=True)
    save_data(taskUtil, e)
    taskUtil.failure_added = True
    taskUtil.gt_failure['gt_failure_reason'] = f'Dropped object at step {failure_injection_idx}'
    taskUtil.gt_failure['gt_failure_step'] = taskUtil.counter + 1
    taskUtil.failures_already_injected.append(['drop', failure_injection_idx])


def save_data(taskUtil, event, replan: bool = False):
    """
    Save event data (simplified version)
    
    Args:
        taskUtil: TaskUtil instance
        event: AI2THOR event
        replan: Whether this is a replan action
    """
    taskUtil.counter += 1
    # In a full implementation, this would save frames, metadata, etc.
    # For now, we just increment the counter and store the event
    if not hasattr(taskUtil, 'events'):
        taskUtil.events = []
    taskUtil.events.append(event)

