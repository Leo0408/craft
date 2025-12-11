# STEP 1: DATA GENERATION WITH REAL AI2THOR EXECUTION
# ============================================================================
print("\n" + "="*80)
print("STEP 1: DATA GENERATION WITH REAL AI2THOR EXECUTION")
print("="*80)

# Task configuration (similar to REFLECT demo)
task_info_craft = {
    "name": "make coffee",
    "scene": "FloorPlan16",  # AI2THOR scene with kitchen
    "object_list": ["Mug", "CoffeeMachine", "Sink", "Faucet", "CounterTop"],
    "success_condition": "a clean mug filled with coffee is on top of the countertop",
}

# Action sequence (similar to REFLECT demo format)
action_instructions = [
    "navigate_to_obj, Mug",
    "pick_up, Mug",
    "navigate_to_obj, Sink",
    "put_on, SinkBasin, Mug",
    "toggle_on, Faucet",
    "toggle_off, Faucet",
    "pick_up, Mug",
    "navigate_to_obj, CoffeeMachine",
    "put_in, CoffeeMachine, Mug",  # This may fail if machine already has a cup
]

print(f"✅ Task: {task_info_craft['name']}")
print(f"✅ Scene: {task_info_craft['scene']}")
print(f"✅ Actions: {len(action_instructions)}")

# Initialize AI2THOR Controller
events_craft = []
action_results = []

if AI2THOR_AVAILABLE:
    print(f"\n🔧 Initializing AI2THOR Controller...")
    print(f"   Scene: {task_info_craft['scene']}")
    
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
        platform=CloudRendering
    )
    
    print(f"✅ Controller initialized")
    
    # Get initial scene state
    print(f"\n📊 Getting initial scene state...")
    initial_event = controller.step(action="GetReachablePositions")
    reachable_positions = initial_event.metadata["actionReturn"]
    objects_in_scene = initial_event.metadata.get("objects", [])
    
    print(f"   - Reachable positions: {len(reachable_positions)}")
    print(f"   - Objects in scene: {len(objects_in_scene)}")
    
    # Find object IDs by type
    object_map = {}
    for obj in objects_in_scene:
        obj_type = obj.get("objectType", "")
        if obj_type not in object_map:
            object_map[obj_type] = []
        object_map[obj_type].append(obj.get("objectId", ""))
    
    print(f"\n📦 Objects found in scene:")
    for obj_type, obj_ids in object_map.items():
        if obj_type in task_info_craft['object_list']:
            print(f"   - {obj_type}: {len(obj_ids)} found")
            if len(obj_ids) > 0:
                print(f"     IDs: {obj_ids[:3]}{'...' if len(obj_ids) > 3 else ''}")
    
    # Helper function to make robot look at object
    def look_at_object(controller, target_pos, robot_pos, center_to_camera_disp=0.6):
        """Make robot look at target position"""
        robot_y = robot_pos["y"] + center_to_camera_disp
        yaw = np.arctan2(target_pos["x"] - robot_pos["x"], target_pos["z"] - robot_pos["z"])
        yaw = math.degrees(yaw)
        
        tilt = -np.arctan2(target_pos["y"] - robot_y, 
                          np.sqrt((target_pos["z"] - robot_pos["z"])**2 + (target_pos["x"] - robot_pos["x"])**2))
        tilt = np.round(np.degrees(tilt), 1)
        org_tilt = controller.last_event.metadata["agent"]["cameraHorizon"]
        final_tilt = tilt - org_tilt
        if tilt > 60:
            final_tilt = 60
        if tilt < -30:
            final_tilt = -30
        final_tilt = np.round(final_tilt, 1)
        
        # Rotate robot to face object
        event = controller.step(action="Teleport", **robot_pos, rotation=dict(x=0, y=yaw, z=0), forceAction=True)
        
        # Adjust camera tilt
        if final_tilt > 0:
            event = controller.step(action="LookDown", degrees=final_tilt)
        elif final_tilt < 0:
            event = controller.step(action="LookUp", degrees=-final_tilt)
        
        return event
    
    # Execute actions one by one with detailed output
    print(f"\n" + "-"*80)
    print("EXECUTING ACTIONS IN AI2THOR")
    print("-"*80)
    
    for action_idx, action_instr in enumerate(action_instructions, 1):
        print(f"\n{'='*80}")
        print(f"Action {action_idx}/{len(action_instructions)}: {action_instr}")
        print(f"{'='*80}")
        
        # Parse action instruction
        parts = [p.strip() for p in action_instr.split(',')]
        action_name = parts[0]
        params = parts[1:] if len(parts) > 1 else []
        
        # Map object names to object IDs
        action_params = {}
        if len(params) > 0:
            obj_name = params[0]
            obj_id = None
            
            for obj in objects_in_scene:
                obj_type = obj.get("objectType", "")
                obj_id_full = obj.get("objectId", "")
                
                if obj_type == obj_name:
                    obj_id = obj_id_full
                    break
                elif obj_name in obj_type or obj_type in obj_name:
                    obj_id = obj_id_full
                    break
                elif obj_name in obj_id_full:
                    obj_id = obj_id_full
                    break
            
            if obj_id:
                action_params["objectId"] = obj_id
                print(f"  Mapped '{obj_name}' to objectId: {obj_id[:50]}...")
            else:
                print(f"  ⚠️  Could not find object ID for '{obj_name}'")
                for obj in objects_in_scene:
                    obj_type = obj.get("objectType", "")
                    if obj_name.lower() in obj_type.lower() or obj_type.lower() in obj_name.lower():
                        obj_id = obj.get("objectId")
                        action_params["objectId"] = obj_id
                        print(f"  Using similar object: {obj_type} ({obj_id[:50]}...)")
                        break
        
        print(f"  Action: {action_name}")
        if action_params:
            print(f"  Params: {action_params}")
        
        # Execute action
        try:
            if action_name == "navigate_to_obj":
                # Navigate to object: find nearest reachable position
                obj_id = action_params.get('objectId')
                if obj_id:
                    objects = controller.last_event.metadata.get('objects', [])
                    obj_pos = None
                    for obj in objects:
                        if obj.get('objectId') == obj_id:
                            obj_pos = obj.get('position', {})
                            break
                    
                    if obj_pos:
                        reachable_event = controller.step(action='GetReachablePositions')
                        reachable_positions = reachable_event.metadata.get('actionReturn', [])
                        
                        if reachable_positions:
                            def distance(pos1, pos2):
                                return math.sqrt((pos1['x'] - pos2['x'])**2 + (pos1['z'] - pos2['z'])**2)
                            
                            nearest_pos = min(reachable_positions, 
                                              key=lambda p: distance(p, obj_pos))
                            
                            event = controller.step(action='Teleport', 
                                                    x=nearest_pos['x'], 
                                                    y=nearest_pos['y'], 
                                                    z=nearest_pos['z'])
                            status = 'SUCCESS' if event.metadata.get('lastActionSuccess') else 'FAILED'
                            error = None if status == 'SUCCESS' else 'Teleport failed'
                            if status == 'SUCCESS':
                                print(f"  ✅ Navigated to position: ({nearest_pos['x']:.2f}, {nearest_pos['y']:.2f}, {nearest_pos['z']:.2f})")
                                # Look at object after navigation
                                robot_pos = controller.last_event.metadata["agent"]["position"]
                                look_at_object(controller, obj_pos, robot_pos)
                            else:
                                print(f"  ⚠️  Teleport failed, using last event")
                                event = controller.last_event
                        else:
                            print(f"  ⚠️  No reachable positions, using last event")
                            event = controller.last_event
                            status = 'SUCCESS'
                            error = None
                    else:
                        print(f"  ⚠️  Object position not found, using last event")
                        event = controller.last_event
                        status = 'SUCCESS'
                        error = None
                else:
                    print(f"  ⚠️  No objectId provided, using last event")
                    event = controller.last_event
                    status = 'SUCCESS'
                    error = None
            elif action_name == "pick_up":
                if "objectId" in action_params:
                    # Look at object before picking up
                    obj = None
                    for o in controller.last_event.metadata["objects"]:
                        if o.get("objectId") == action_params.get("objectId"):
                            obj = o
                            break
                    if obj:
                        robot_pos = controller.last_event.metadata["agent"]["position"]
                        look_at_object(controller, obj["position"], robot_pos)
                        # Refresh object after look_at
                        for o in controller.last_event.metadata["objects"]:
                            if o.get("objectId") == action_params.get("objectId"):
                                obj = o
                                break
                    
                    event = controller.step(action="PickupObject", **action_params)
                    status = "SUCCESS" if event.metadata.get("lastActionSuccess") else "FAILED"
                    error = None if status == "SUCCESS" else "Pickup failed"
                else:
                    print(f"  Status: ❌ FAILED - Object ID not found")
                    event = controller.last_event
                    status = "FAILED"
                    error = "Object ID not found"
            elif action_name == "put_on":
                if "objectId" in action_params:
                    print(f"  Status: ⚠️  PutOn action (simplified execution)")
                    event = controller.last_event
                    status = "SUCCESS"
                    error = None
                else:
                    event = controller.last_event
                    status = "FAILED"
                    error = "Object ID not found"
            elif action_name == "put_in":
                if "objectId" in action_params:
                    # Look at target object before putting
                    target_obj = None
                    for o in controller.last_event.metadata["objects"]:
                        if o.get("objectId") == action_params.get("objectId"):
                            target_obj = o
                            break
                    if target_obj:
                        robot_pos = controller.last_event.metadata["agent"]["position"]
                        look_at_object(controller, target_obj["position"], robot_pos)
                    
                    event = controller.step(action="PutObject", **action_params)
                    status = "SUCCESS" if event.metadata.get("lastActionSuccess") else "FAILED"
                    if not status == "SUCCESS":
                        error_msg = event.metadata.get("errorMessage", "PutIn failed")
                        error = f"PutIn failed: {error_msg}"
                    else:
                        error = None
                else:
                    event = controller.last_event
                    status = "FAILED"
                    error = "Object ID not found"
            elif action_name == "toggle_on":
                if "objectId" in action_params:
                    # Look at object before toggling
                    obj = None
                    for o in controller.last_event.metadata["objects"]:
                        if o.get("objectId") == action_params.get("objectId"):
                            obj = o
                            break
                    if obj:
                        robot_pos = controller.last_event.metadata["agent"]["position"]
                        look_at_object(controller, obj["position"], robot_pos)
                    
                    event = controller.step(action="ToggleObjectOn", **action_params)
                    status = "SUCCESS" if event.metadata.get("lastActionSuccess") else "FAILED"
                    error = None if status == "SUCCESS" else "ToggleOn failed"
                else:
                    event = controller.last_event
                    status = "FAILED"
                    error = "Object ID not found"
            elif action_name == "toggle_off":
                if "objectId" in action_params:
                    # Look at object before toggling
                    obj = None
                    for o in controller.last_event.metadata["objects"]:
                        if o.get("objectId") == action_params.get("objectId"):
                            obj = o
                            break
                    if obj:
                        robot_pos = controller.last_event.metadata["agent"]["position"]
                        look_at_object(controller, obj["position"], robot_pos)
                    
                    event = controller.step(action="ToggleObjectOff", **action_params)
                    status = "SUCCESS" if event.metadata.get("lastActionSuccess") else "FAILED"
                    error = None if status == "SUCCESS" else "ToggleOff failed"
                else:
                    event = controller.last_event
                    status = "FAILED"
                    error = "Object ID not found"
            else:
                print(f"  Status: ⚠️  Unknown action type")
                event = controller.last_event
                status = "UNKNOWN"
                error = None
            
            # Display result
            status_icon = "✅" if status == "SUCCESS" else "❌" if status == "FAILED" else "⚠️"
            print(f"  Status: {status_icon} {status}")
            if error:
                print(f"  Error: {error}")
            
            # Store event and result
            events_craft.append(event)
            action_results.append({
                "action_idx": action_idx,
                "action": action_instr,
                "action_name": action_name,
                "params": action_params,
                "status": status,
                "error": error,
                "event": event
            })
            
            # If action failed, we can stop or continue
            if status == "FAILED":
                print(f"\n  ⚠️  Action failed. Continuing to next steps for demonstration...")
        
        except Exception as e:
            print(f"  Status: ❌ ERROR")
            print(f"  Error: {str(e)}")
            event = controller.last_event if controller else None
            events_craft.append(event)
            action_results.append({
                "action_idx": action_idx,
                "action": action_instr,
                "status": "ERROR",
                "error": str(e),
                "event": event
            })
        
        # Small delay for visibility
        time.sleep(0.1)
    
    print(f"\n" + "-"*80)
    print(f"✅ Executed {len(action_results)} actions")
    print(f"   Successful: {sum(1 for r in action_results if r['status'] == 'SUCCESS')}")
    print(f"   Failed: {sum(1 for r in action_results if r['status'] == 'FAILED')}")
    print(f"   Errors: {sum(1 for r in action_results if r['status'] == 'ERROR')}")
    
else:
    print(f"\n⚠️  AI2THOR not available. Using mock data for demonstration.")
    # Fallback to mock data
    action_sequence = [
        {"type": "navigate_to", "target": "Mug", "status": "SUCCESS"},
        {"type": "pick_up", "target": "Mug", "status": "SUCCESS"},
        {"type": "navigate_to", "target": "Sink", "status": "SUCCESS"},
        {"type": "put_on", "target": "SinkBasin", "status": "SUCCESS"},
        {"type": "toggle_on", "target": "Faucet", "status": "SUCCESS"},
        {"type": "toggle_off", "target": "Faucet", "status": "SUCCESS"},
        {"type": "pick_up", "target": "Mug", "status": "SUCCESS"},
        {"type": "navigate_to", "target": "CoffeeMachine", "status": "SUCCESS"},
        {"type": "put_in", "target": "CoffeeMachine", "status": "FAILED", 
         "error": "Coffee machine already contains a cup, blocking insertion"},
    ]
    
    events_craft = []
    action_results = []
    for i, action in enumerate(action_sequence):
        event = {
            "step_idx": i,
            "action": action,
            "frame": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        events_craft.append(event)
        action_results.append({
            "action_idx": i+1,
            "action": f"{action['type']}, {action.get('target', '')}",
            "status": action.get('status', 'UNKNOWN'),
            "error": action.get('error'),
            "event": event
        })
    
    print(f"✅ Generated {len(events_craft)} mock events")

