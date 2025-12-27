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
            node = Node(
                name=obj.get('name', 'unknown'),
                object_type=obj_type,
                state=state,  # Composite state
                position=position,
                attributes={
                    'isFilled': obj.get('isFilledWithLiquid', False) or obj.get('isFilled', False),
                    'isOpen': obj.get('isOpen', False),
                    'isToggled': obj.get('isToggledOn', False) or obj.get('isToggled', False),
                    'isPickedUp': obj.get('isPickedUp', False),
                    'fillLiquid': obj.get('fillLiquid', None),
                    'isDirty': obj.get('isDirty', False),
                    'isSliced': obj.get('isSliced', False),
                    'isBroken': obj.get('isBroken', False),
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
                                
                                if is_openable_container or (has_receptacle and receptacle_count > 0):
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
        for obj in objects:
            node = sg.get_node(obj.get('name', 'unknown'))
            if not node or not node.position:
                continue
            
            # Skip if already in container
            if obj.get('parentReceptacles'):
                continue
            
            obj_pos = node.position
            
            # Check if on top of other objects
            for other_obj in objects:
                if obj.get('objectId') == other_obj.get('objectId'):
                    continue
                
                other_node = sg.get_node(other_obj.get('name', 'unknown'))
                if not other_node or not other_node.position:
                    continue
                
                other_pos = other_node.position
                other_type = other_obj.get('objectType', '').lower()
                
                # Calculate spatial relationship
                z_diff = obj_pos[2] - other_pos[2]
                horizontal_dist = ((obj_pos[0] - other_pos[0])**2 + (obj_pos[1] - other_pos[1])**2)**0.5
                
                # Dynamic surface type detection
                is_surface = any(kw in other_type for kw in ['countertop', 'table', 'stoveburner', 'burner', 'sink'])
                
                if (0.05 < z_diff < 0.5 and horizontal_dist < 0.2 and is_surface):
                    edge_key = (node.name, other_node.name)
                    existing_edge = sg.edges.get(edge_key)
                    if not existing_edge or existing_edge.edge_type != 'inside':
                        sg.add_edge(Edge(node, other_node, "on_top_of"))
        
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

