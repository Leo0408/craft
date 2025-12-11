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
    Save event data including frames (like REFLECT)
    
    Args:
        taskUtil: TaskUtil instance
        event: AI2THOR event
        replan: Whether this is a replan action
    """
    import os
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    
    print(f"[DEBUG] save_data: Called for step {taskUtil.counter + 1}, event={event is not None}")
    
    taskUtil.counter += 1
    
    # Determine folder (like REFLECT)
    if replan:
        folder = 'recovery'
    else:
        folder = 'thor_tasks'
    
    # Create directories
    base_path = getattr(taskUtil, 'repo_path', os.getcwd())
    specific_folder = getattr(taskUtil, 'specific_folder_name', taskUtil.folder_name if hasattr(taskUtil, 'folder_name') else 'default')
    
    events_dir = f'{base_path}/{folder}/{specific_folder}/events'
    ego_img_dir = f'{base_path}/{folder}/{specific_folder}/ego_img'
    
    print(f"[DEBUG] save_data: Creating directories:")
    print(f"  events_dir: {events_dir}")
    print(f"  ego_img_dir: {ego_img_dir}")
    
    try:
        os.makedirs(events_dir, exist_ok=True)
        os.makedirs(ego_img_dir, exist_ok=True)
        print(f"[DEBUG] save_data: Directories created successfully")
    except Exception as dir_error:
        print(f"[DEBUG] save_data: Failed to create directories: {dir_error}")
        import traceback
        traceback.print_exc()
        return
    
    # Save event as pickle (like REFLECT)
    event_path = f'{events_dir}/step_{taskUtil.counter}.pickle'
    try:
        with open(event_path, 'wb') as handle:
            pickle.dump(event, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[DEBUG] save_data: Saved event pickle to {event_path}")
    except Exception as pickle_error:
        print(f"[DEBUG] save_data: Failed to save pickle: {pickle_error}")
        import traceback
        traceback.print_exc()
    
    # Save frame to ego_img directory (like REFLECT)
    try:
        # Extract frame from event (like REFLECT: directly use e.frame)
        frame = None
        print(f"[DEBUG] save_data: Extracting frame from event...")
        print(f"  event has frame attr: {hasattr(event, 'frame')}")
        if hasattr(event, 'frame'):
            print(f"  event.frame is not None: {event.frame is not None}")
            if event.frame is not None:
                print(f"  event.frame type: {type(event.frame)}, shape: {event.frame.shape if hasattr(event.frame, 'shape') else 'N/A'}")
        
        if hasattr(event, 'frame') and event.frame is not None:
            # REFLECT style: directly use e.frame
            frame = event.frame
            print(f"[DEBUG] save_data: Using event.frame")
        elif hasattr(event, 'cv2image'):
            print(f"[DEBUG] save_data: Trying cv2image...")
            # Fallback: try cv2image
            if callable(event.cv2image):
                frame = event.cv2image()
                print(f"[DEBUG] save_data: Called event.cv2image()")
            else:
                frame = event.cv2image
                print(f"[DEBUG] save_data: Using event.cv2image attribute")
            # Convert BGR to RGB if needed
            if frame is not None and len(frame.shape) == 3:
                import cv2
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    print(f"[DEBUG] save_data: Converted BGR to RGB")
        
        if frame is not None:
            img_path = f'{ego_img_dir}/img_step_{taskUtil.counter}.png'
            # Use REFLECT style: np.asarray(rgb, order='C')
            try:
                plt.imsave(img_path, np.asarray(frame, order='C'))
                print(f"[DEBUG] save_data: Successfully saved frame for step {taskUtil.counter} to {img_path}")
            except Exception as save_error:
                print(f"[DEBUG] save_data: plt.imsave failed for step {taskUtil.counter}: {save_error}")
                import traceback
                traceback.print_exc()
        else:
            # Debug: print warning if frame is None
            print(f"[DEBUG] save_data: Frame is None for step {taskUtil.counter}")
            print(f"[DEBUG] save_data: event has frame attr: {hasattr(event, 'frame')}")
            if hasattr(event, 'frame'):
                print(f"[DEBUG] save_data: event.frame is None: {event.frame is None}")
            print(f"[DEBUG] save_data: event has cv2image attr: {hasattr(event, 'cv2image')}")
    except Exception as e:
        # Debug: print error instead of silently failing
        print(f"[DEBUG] save_data: Failed to save frame for step {taskUtil.counter}: {e}")
        import traceback
        traceback.print_exc()
    
    # Also store event in memory for backward compatibility
    if not hasattr(taskUtil, 'events'):
        taskUtil.events = []
    taskUtil.events.append(event)


def pour(taskUtil, src_obj_type: str, target_obj_type: str, 
         fail_execution: bool = False, replan: bool = False):
    """
    Pour liquid from one object to another
    
    Args:
        taskUtil: TaskUtil instance
        src_obj_type: Source object type (object with liquid)
        target_obj_type: Target object type (receptacle)
        fail_execution: Whether to fail execution
        replan: Whether this is a replan action
    """
    print(f"[INFO] Execute action: Pouring liquid from {src_obj_type} to {target_obj_type}")
    liquid_type = None
    src_obj_type_in_sim = src_obj_type
    if src_obj_type in NAME_MAP:
        src_obj_type_in_sim = NAME_MAP[src_obj_type]
    target_obj_type_in_sim = target_obj_type
    if target_obj_type in NAME_MAP:
        target_obj_type_in_sim = NAME_MAP[target_obj_type]

    if taskUtil.chosen_failure == "wrong_perception":
        if src_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            src_obj_type = taskUtil.failure_injection_params['wrong_obj_type']
        elif target_obj_type == taskUtil.failure_injection_params['correct_obj_type']:
            target_obj_type = taskUtil.failure_injection_params['wrong_obj_type']

    target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                     if obj["objectType"] == target_obj_type)
    target_obj_id = target_obj['objectId']
    src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                  if obj["objectType"] == src_obj_type)
    src_obj_id = src_obj['objectId']

    # Navigate if needed
    if not target_obj['visible'] and target_obj['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, target_obj['objectType'], replan=replan)

    # Check if object in hand has liquid
    obj_in_hand = None
    for obj in taskUtil.controller.last_event.metadata['objects']:
        if obj['isPickedUp'] == True:
            obj_in_hand = obj
            break
    
    if obj_in_hand is not None and obj_in_hand["isFilledWithLiquid"] and src_obj_id == obj_in_hand['objectId']:
        liquid_type = obj_in_hand['fillLiquid']

        if fail_execution:
            e = taskUtil.controller.last_event
            save_data(taskUtil, e, replan=replan)
            return
        
        e = taskUtil.controller.step(
            action="EmptyLiquidFromObject",
            objectId=src_obj_id,
            forceAction=False
        )
        e = taskUtil.controller.step(
            action="FillObjectWithLiquid",
            objectId=target_obj_id,
            fillLiquid=liquid_type.lower(),
            forceAction=False
        )
        save_data(taskUtil, e, replan=replan)
        taskUtil.controller.step(action="Done")
        taskUtil.interact_actions[taskUtil.counter] = f"Pour {liquid_type.lower() if liquid_type else 'liquid'} from {src_obj_type_in_sim.lower()} into {target_obj_type_in_sim.lower()}"

    time.sleep(1)


def dirty_obj(taskUtil, obj_type: str):
    """
    Make an object dirty (preaction)
    
    Args:
        taskUtil: TaskUtil instance
        obj_type: Object type to make dirty
    """
    src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                  if obj["objectType"] == obj_type)
    
    e = taskUtil.controller.step(
        action="DirtyObject",
        objectId=src_obj["objectId"],
        forceAction=True
    )
    print("DirtyObject: ", e)
    taskUtil.controller.step(action="Done")


def place_obj(taskUtil, failure_injection_params: dict):
    """
    Place objects for failure injection (occupied, blocking, etc.)
    Based on REFLECT's place_obj function
    
    Args:
        taskUtil: TaskUtil instance
        failure_injection_params: Parameters for failure injection
            - src_obj_type: Source object type to place
            - target_obj_type: Target object type (receptacle)
            - disp_x, disp_y, disp_z: Displacement offsets
    """
    if taskUtil.chosen_failure == "occupied_put":
        src_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                      if obj["objectType"] == failure_injection_params['src_obj_type'])
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                         if obj["objectType"] == failure_injection_params['target_obj_type'])
        
        e = taskUtil.controller.step(
            action="PickupObject",
            objectId=src_obj['objectId'],
            forceAction=True,
            manualInteract=False
        )
        taskUtil.controller.step(action='Done')
        
        if failure_injection_params['target_obj_type'] == 'Microwave':
            taskUtil.controller.step(
                action="OpenObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
            e = taskUtil.controller.step(
                action="PutObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
            taskUtil.controller.step(
                action="CloseObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
            taskUtil.controller.step(action='Done')
        else:
            e = taskUtil.controller.step(
                action="PutObject",
                objectId=target_obj['objectId'],
                forceAction=True
            )
    elif taskUtil.chosen_failure == "occupied":
        target_obj_type = failure_injection_params['target_obj_type']
        if "-" in target_obj_type and target_obj_type.split("-")[0] in ['StoveBurner', 'CounterTop']:
            for key, val in taskUtil.unity_name_map.items():
                if val == target_obj_type:
                    target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                                    if obj["name"] == key)
                    break
        else:
            target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                            if obj["objectType"] == failure_injection_params['target_obj_type'])
        
        objectPoses = []
        place_location = target_obj['position'].copy()
        objs = taskUtil.controller.last_event.metadata["objects"]
        for obj in objs:
            obj_name = obj['name']
            obj_type = obj['objectType']
            pos = obj['position']
            rot = obj['rotation']
            if not obj['pickupable'] and not obj['moveable']:
                continue
            if obj_type == failure_injection_params['src_obj_type']:
                pos = place_location.copy()
                pos['x'] += failure_injection_params.get('disp_x', 0.0)
                pos['z'] += failure_injection_params.get('disp_z', 0.0)
                pos['y'] += failure_injection_params.get('disp_y', 0.0)
            temp_dict = {'objectName': obj_name, 'position': pos, 'rotation': rot}
            objectPoses.append(temp_dict)
        
        e = taskUtil.controller.step(
            action='SetObjectPoses',
            objectPoses=objectPoses,
            placeStationary=False
        )
        print("SetObjectPoses (occupied): ", e)
        taskUtil.controller.step(
            action="AdvancePhysicsStep",
            timeStep=0.01
        )
        taskUtil.controller.step(action='Done')
    else:
        # For blocking and other failure types
        target_obj = next(obj for obj in taskUtil.controller.last_event.metadata["objects"] 
                         if obj["objectType"] == failure_injection_params['target_obj_type'])
        
        place_location = target_obj['position'].copy()
        objs = taskUtil.controller.last_event.metadata["objects"]
        objectPoses = []
        for obj in objs:
            obj_name = obj['name']
            obj_type = obj['objectType']
            pos = obj['position']
            rot = obj['rotation']
            if not obj['pickupable'] and not obj['moveable']:
                continue
            if obj_type == failure_injection_params['src_obj_type']:
                pos = place_location.copy()
                pos['x'] += failure_injection_params.get('disp_x', 0.0)
                pos['z'] += failure_injection_params.get('disp_z', 0.0)
                pos['y'] += failure_injection_params.get('disp_y', 0.0)
            temp_dict = {'objectName': obj_name, 'position': pos, 'rotation': rot}
            objectPoses.append(temp_dict)
        
        e = taskUtil.controller.step(
            action='SetObjectPoses',
            objectPoses=objectPoses,
            placeStationary=False
        )
        taskUtil.controller.step(action='Done')
        print("SetObjectPoses: ", e)

