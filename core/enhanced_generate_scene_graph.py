"""
Enhanced generate_scene_graph_from_event function
Combines CRAFT's dynamic approach with REFLECT's rich features
"""

from typing import Dict, Optional
from .scene_graph import SceneGraph, Node, Edge
from .enhanced_scene_graph_utils import (
    extract_composite_state,
    add_rich_spatial_relations
)


def generate_scene_graph_from_event_enhanced(
    event, 
    task_info: Dict, 
    timestep: Optional[int] = None,
    action: Optional[str] = None,
    use_point_cloud: bool = False,
    use_rich_relations: bool = True
) -> SceneGraph:
    """
    Enhanced scene graph generation from AI2THOR event
    
    Features:
    - Composite states (e.g., "filled with coffee and dirty")
    - Rich relation types (above, below, blocking, left_of, right_of)
    - Point cloud-based precise calculations (if available)
    - Dynamic state and relation extraction (no hardcoded types)
    
    Args:
        event: AI2THOR event object
        task_info: Task information dictionary
        timestep: Timestep for Action-aware SG
        action: Action for Action-aware SG
        use_point_cloud: Whether to use point cloud data for calculations
        use_rich_relations: Whether to add rich spatial relations
        
    Returns:
        SceneGraph object
    """
    # Create Action-aware Scene Graph
    sg = SceneGraph(task=task_info, timestep=timestep, action=action)
    
    # Get camera info for rich relations (if available)
    camera_world_xyz = None
    rotation = None
    horizon = None
    if hasattr(event, 'metadata') and event.metadata:
        agent = event.metadata.get('agent', {})
        if agent:
            pos = agent.get('position', {})
            if pos:
                camera_world_xyz = (
                    pos.get('x', 0),
                    pos.get('y', 0),
                    pos.get('z', 0)
                )
            rotation = agent.get('rotation', {}).get('y', 0)
            horizon = agent.get('cameraHorizon', 0)
    
    # Extract objects from event.metadata
    if hasattr(event, 'metadata') and event.metadata:
        objects = event.metadata.get('objects', [])
        
        # Step 1: Create nodes with composite states
        for obj in objects:
            # Extract composite state (supports multiple states)
            state = extract_composite_state(obj)
            
            # Get object type
            obj_type = obj.get('objectType', '')
            
            # Get position
            position = None
            if obj.get('position'):
                pos = obj['position']
                if isinstance(pos, dict):
                    position = (pos.get('x', 0), pos.get('y', 0), pos.get('z', 0))
                elif isinstance(pos, (list, tuple)) and len(pos) >= 3:
                    position = tuple(pos[:3])
            
            # Create node with enhanced attributes
            # 关键：每个 event frame 都同步更新所有状态属性
            # 确保 isToggled, isOpen, isFilled 等状态在每个时间步都正确反映
            node = Node(
                name=obj.get('name', 'unknown'),
                object_type=obj_type,
                state=state,  # Composite state
                position=position,
                attributes={
                    # 状态属性：从 metadata 中直接读取，每个 frame 都更新
                    'isFilled': obj.get('isFilledWithLiquid', False) or obj.get('isFilled', False),
                    'isOpen': obj.get('isOpen', False),
                    # isToggled: 优先使用 isToggledOn，回退到 isToggled，再检查 toggleable 属性
                    # AI2THOR 中，某些对象（如 Faucet）的 toggle 状态可能在不同的字段中
                    'isToggled': (
                        obj.get('isToggledOn', False) or 
                        obj.get('isToggled', False) or
                        (obj.get('toggleable', False) and obj.get('isOn', False)) or
                        obj.get('isOn', False)
                    ),
                    'isPickedUp': obj.get('isPickedUp', False),
                    'fillLiquid': obj.get('fillLiquid', None),
                    'isDirty': obj.get('isDirty', False),
                    'isSliced': obj.get('isSliced', False),
                    'isBroken': obj.get('isBroken', False),
                    # 其他可能的状态属性（用于调试）
                    'toggleable': obj.get('toggleable', None),
                    'openable': obj.get('openable', False),
                    'isOn': obj.get('isOn', None),  # Faucet 等可能使用 isOn
                    'isToggledOn_raw': obj.get('isToggledOn', None),  # 原始值用于调试
                    'isToggled_raw': obj.get('isToggled', None),  # 原始值用于调试
                }
            )
            
            # Add point cloud data if available (from event or pre-computed)
            # Note: In AI2THOR, point clouds are typically computed from depth frames
            # This is a placeholder - actual implementation would extract from depth frame
            if use_point_cloud and hasattr(event, 'depth_frame'):
                # Point cloud extraction would happen here
                # For now, we'll leave it as None
                node.pcd = None  # Would be computed from depth_frame
            
            sg.add_node(node)
        
        # Step 2: Add basic relations (holding, inside, on_top_of)
        for obj in objects:
            node = sg.get_node(obj.get('name', 'unknown'))
            if not node:
                continue
            
            # Holding relation
            if obj.get('isPickedUp', False):
                robot_node = sg.get_node("Robot")
                if not robot_node:
                    robot_node = Node(name="Robot", object_type="Robot")
                    sg.add_node(robot_node)
                sg.add_edge(Edge(robot_node, node, "holding"))
            
            # Container relations (parentReceptacles)
            if obj.get('parentReceptacles'):
                for parent_id in obj.get('parentReceptacles', []):
                    for other_obj in objects:
                        if other_obj.get('objectId') == parent_id:
                            parent_node = sg.get_node(other_obj.get('name', 'unknown'))
                            if parent_node:
                                # Dynamic relation type judgment
                                has_receptacle = bool(other_obj.get('receptacleObjectIds', []))
                                is_openable_container = 'isOpen' in other_obj or other_obj.get('openable', False)
                                receptacle_count = len(other_obj.get('receptacleObjectIds', [])) if isinstance(other_obj.get('receptacleObjectIds'), list) else 0
                                
                                # 改进：Sink, SinkBasin 等容器类型即使 receptacleObjectIds 为空也应被识别为容器
                                parent_obj_type = other_obj.get('objectType', '').lower()
                                is_container_type = any(kw in parent_obj_type for kw in ['sink', 'sinkbasin', 'bowl', 'pot', 'pan', 'mug', 'cup'])
                                
                                if is_openable_container or (has_receptacle and receptacle_count > 0) or is_container_type:
                                    edge_key = (node.name, parent_node.name)
                                    if edge_key not in sg.edges:
                                        sg.add_edge(Edge(node, parent_node, "inside"))
                                else:
                                    # Surface type (CounterTop, Table, etc.)
                                    edge_key = (node.name, parent_node.name)
                                    if edge_key not in sg.edges:
                                        sg.add_edge(Edge(node, parent_node, "on_top_of"))
                            break
        
        # Step 3: Add on_top_of relations based on position (for objects not in containers)
        # Using hybrid method: metadata first, then position, then point cloud
        from .enhanced_scene_graph_utils import determine_spatial_relation_hybrid
        
        for obj in objects:
            node = sg.get_node(obj.get('name', 'unknown'))
            if not node:
                continue
            
            # Skip if already determined by metadata (Step 2)
            if obj.get('parentReceptacles'):
                continue
            
            # Check if on top of other objects using hybrid method
            for other_obj in objects:
                if obj.get('objectId') == other_obj.get('objectId'):
                    continue
                
                other_node = sg.get_node(other_obj.get('name', 'unknown'))
                if not other_node:
                    continue
                
                # Skip if relation already exists
                edge_key = (node.name, other_node.name)
                if edge_key in sg.edges:
                    continue
                
                # Use hybrid method to determine relation
                relation_result = determine_spatial_relation_hybrid(
                    obj, other_obj, node, other_node, use_point_cloud
                )
                
                if relation_result:
                    rel_type, confidence = relation_result
                    # Add relation if confidence is reasonable
                    # Metadata-based relations (confidence=1.0) should already be added in Step 2
                    # This step handles position-based and point cloud-based relations
                    if confidence >= 0.75:  # Only add medium to high confidence relations
                        sg.add_edge(Edge(node, other_node, rel_type))
        
        # Step 4: Add rich spatial relations (above, below, blocking, left_of, right_of)
        if use_rich_relations:
            add_rich_spatial_relations(
                sg, objects,
                use_point_cloud=use_point_cloud,
                camera_world_xyz=camera_world_xyz,
                rotation=rotation,
                horizon=horizon
            )
    
    return sg

