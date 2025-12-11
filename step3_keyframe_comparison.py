# Step 3 with two keyframe selection methods comparison
# This code will be integrated into demo1.ipynb

# STEP 3: SCENE GRAPH GENERATION WITH TWO KEYFRAME METHODS
# ============================================================================
print("\n" + "="*80)
print("STEP 3: SCENE GRAPH GENERATION - KEYFRAME METHOD COMPARISON")
print("="*80)

# Method 1: Original method (one scene graph per action)
print("\n" + "="*80)
print("METHOD 1: ORIGINAL (One Scene Graph Per Action)")
print("="*80)

scene_graphs_original = []

print(f"\n📊 Processing {len(events_craft)} events to generate scene graphs...")
print("-"*80)

for event_idx, (event, action_result) in enumerate(zip(events_craft, action_results)):
    print(f"\nProcessing Event {event_idx + 1}/{len(events_craft)}...")
    print(f"  Action: {action_result.get('action', 'N/A')}")
    print(f"  Status: {action_result.get('status', 'N/A')}")
    
    sg = generate_scene_graph_from_event(event, action_result, task_info_craft)
    scene_graphs_original.append(sg)
    
    print(f"  ✅ Generated scene graph:")
    print(f"     Nodes: {len(sg.nodes)}")
    print(f"     Edges: {len(sg.edges)}")
    if len(sg.nodes) > 0:
        print(f"     Description: {sg.to_text()[:100]}...")

print(f"\n" + "-"*80)
print(f"✅ Method 1: Generated {len(scene_graphs_original)} scene graphs (one per action)")

# Method 2: Smart keyframe selection (Scheme 1)
print("\n" + "="*80)
print("METHOD 2: SMART KEYFRAME SELECTION (Scheme 1)")
print("="*80)

scene_graphs_smart = []
prev_sg = None
keyframe_indices = []  # Track which events were selected as keyframes

print(f"\n📊 Processing {len(events_craft)} events with smart keyframe selection...")
print("-"*80)

for event_idx, (event, action_result) in enumerate(zip(events_craft, action_results)):
    action_name = action_result.get('action_name', '')
    action_str = action_result.get('action', 'N/A')
    
    # Generate scene graph for this event
    current_sg = generate_scene_graph_from_event(event, action_result, task_info_craft)
    
    # Decide if this should be a keyframe
    should_keep = should_generate_scene_graph(prev_sg, current_sg, action_name)
    
    if should_keep:
        scene_graphs_smart.append(current_sg)
        keyframe_indices.append(event_idx)
        print(f"\n✅ Selected as KEYFRAME - Event {event_idx + 1}/{len(events_craft)}")
        print(f"  Action: {action_str}")
        print(f"  Reason: {'Initial state' if prev_sg is None else 'State changed significantly' if state_changed_significantly(prev_sg, current_sg) else 'Key action'}
")
        print(f"  Scene graph:")
        print(f"    Nodes: {len(current_sg.nodes)}")
        print(f"    Edges: {len(current_sg.edges)}")
        if len(current_sg.nodes) > 0:
            print(f"    Description: {current_sg.to_text()[:100]}...")
        prev_sg = current_sg
    else:
        print(f"\n⏭️  Skipped - Event {event_idx + 1}/{len(events_craft)}: {action_str}")
        print(f"  Reason: State did not change significantly")

print(f"\n" + "-"*80)
print(f"✅ Method 2: Generated {len(scene_graphs_smart)} scene graphs (smart keyframes)")
print(f"   Keyframe indices: {keyframe_indices}")

# Comparison Summary
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)
print(f"\nMethod 1 (Original):")
print(f"  - Total scene graphs: {len(scene_graphs_original)}")
print(f"  - Keyframe selection: Fixed (one per action)")
print(f"  - Reduction: 0% (baseline)")

print(f"\nMethod 2 (Smart):")
print(f"  - Total scene graphs: {len(scene_graphs_smart)}")
print(f"  - Keyframe selection: Dynamic (based on state changes)")
print(f"  - Reduction: {((len(events_craft) - len(scene_graphs_smart)) / len(events_craft) * 100):.1f}%")
print(f"  - Selected keyframes: {len(keyframe_indices)} out of {len(events_craft)} events")

# Keep original for backward compatibility
scene_graphs_craft = scene_graphs_original  # Default to original method

# Display scene graph information for both methods
print(f"\n📊 Method 1 - Scene Graph Summary:")
if len(scene_graphs_original) > 0:
    for i, sg in enumerate([scene_graphs_original[0], scene_graphs_original[-1]], 1):
        state_name = "Initial" if i == 1 else "Final"
        print(f"\n   {state_name} State:")
        print(f"   - Nodes: {len(sg.nodes)}")
        print(f"   - Edges: {len(sg.edges)}")
        if len(sg.nodes) > 0:
            print(f"   - Description: {sg.to_text()[:150]}...")

print(f"\n📊 Method 2 - Scene Graph Summary:")
if len(scene_graphs_smart) > 0:
    for i, sg in enumerate([scene_graphs_smart[0], scene_graphs_smart[-1]], 1):
        state_name = "Initial" if i == 1 else "Final"
        print(f"\n   {state_name} State:")
        print(f"   - Nodes: {len(sg.nodes)}")
        print(f"   - Edges: {len(sg.edges)}")
        if len(sg.nodes) > 0:
            print(f"   - Description: {sg.to_text()[:150]}...")

# Visualize initial and final scene graphs for both methods
if len(scene_graphs_original) > 0 and len(scene_graphs_original[0].nodes) > 0:
    print(f"\n📈 Visualizing Method 1 Scene Graphs...")
    visualize_scene_graph_fixed(scene_graphs_original[0], "Method 1: Initial Scene Graph")
    if len(scene_graphs_original) > 1 and len(scene_graphs_original[-1].nodes) > 0:
        visualize_scene_graph_fixed(scene_graphs_original[-1], "Method 1: Final Scene Graph")

if len(scene_graphs_smart) > 0 and len(scene_graphs_smart[0].nodes) > 0:
    print(f"\n📈 Visualizing Method 2 Scene Graphs...")
    visualize_scene_graph_fixed(scene_graphs_smart[0], "Method 2: Initial Scene Graph (Smart)")
    if len(scene_graphs_smart) > 1 and len(scene_graphs_smart[-1].nodes) > 0:
        visualize_scene_graph_fixed(scene_graphs_smart[-1], "Method 2: Final Scene Graph (Smart)")

# ============================================================================

