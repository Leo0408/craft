"""
Enhanced Scene Graph Generation Utilities
Combines CRAFT's dynamic approach with REFLECT's rich features:
- Composite states (e.g., "filled with coffee and dirty")
- Rich relation types (above, below, blocking, left_of, right_of)
- Point cloud-based precise calculations (if available)
- Hybrid method: Prioritize metadata, fallback to point cloud/position
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from .scene_graph import SceneGraph, Node, Edge

# Spatial relation thresholds (from REFLECT)
IN_CONTACT_DISTANCE = 0.1  # 10cm
CLOSE_DISTANCE = 0.4  # 40cm
INSIDE_THRESH = 0.5
ON_TOP_OF_THRESH = 0.7
NORM_THRESH_FRONT_BACK = 0.9
NORM_THRESH_UP_DOWN = 0.9
NORM_THRESH_LEFT_RIGHT = 0.8
OCCLUDE_RATIO_THRESH = 0.5
DEPTH_THRESH = 0.9

# CRAFT thresholds (world space, in meters)
CRAFT_Z_DIFF_MIN = 0.05  # 5cm minimum height difference
CRAFT_Z_DIFF_MAX = 0.5   # 50cm maximum height difference
CRAFT_HORIZONTAL_DIST_MAX = 0.2  # 20cm maximum horizontal distance


def extract_composite_state(obj: Dict) -> Optional[str]:
    """
    Extract composite state from object metadata (supports multiple states)
    
    Examples:
        - "open and filled with coffee"
        - "turned on and dirty"
        - "filled with water and clean"
    
    Args:
        obj: Object metadata dictionary
        
    Returns:
        Composite state string or None
    """
    states = []
    
    # Check openable objects
    if obj.get('openable') is not None or obj.get('isOpen') is not None:
        is_open = obj.get('isOpen', False)
        states.append('open' if is_open else 'closed')
    
    # Check toggleable objects
    if obj.get('toggleable') is not None or obj.get('isToggledOn') is not None:
        is_toggled = obj.get('isToggledOn', False) or obj.get('isToggled', False)
        states.append('turned on' if is_toggled else 'turned off')
    
    # Check fillable objects
    if obj.get('canFillWithLiquid') is not None or obj.get('isFilled') is not None:
        is_filled = obj.get('isFilledWithLiquid', False) or obj.get('isFilled', False)
        if is_filled:
            fill_liquid = obj.get('fillLiquid', 'liquid')
            if fill_liquid:
                states.append(f'filled with {fill_liquid}')
            else:
                states.append('filled')
        else:
            states.append('empty')
    
    # Check sliceable objects
    if obj.get('sliceable') is not None:
        is_sliced = obj.get('isSliced', False)
        if not is_sliced:
            states.append('not sliced')
    
    # Check breakable objects (e.g., Egg)
    if obj.get('isBroken') is not None:
        is_broken = obj.get('isBroken', False)
        if not is_broken:
            states.append('not cracked')
    
    # Check dirtyable objects
    if obj.get('dirtyable') is not None or obj.get('isDirty') is not None:
        is_dirty = obj.get('isDirty', False)
        states.append('dirty' if is_dirty else 'clean')
    
    if states:
        return ' and '.join(states)
    return None


def get_point_cloud_distance(pcd1, pcd2) -> Optional[float]:
    """
    Calculate minimum distance between two point clouds
    
    Args:
        pcd1: Point cloud 1 (N x 3 array)
        pcd2: Point cloud 2 (M x 3 array)
        
    Returns:
        Minimum distance or None if point clouds are not available
    """
    if pcd1 is None or pcd2 is None:
        return None
    
    try:
        # Convert to numpy if needed
        if hasattr(pcd1, 'cpu'):  # torch tensor
            pcd1 = pcd1.cpu().numpy()
        if hasattr(pcd2, 'cpu'):  # torch tensor
            pcd2 = pcd2.cpu().numpy()
        
        pcd1 = np.array(pcd1)
        pcd2 = np.array(pcd2)
        
        if len(pcd1) == 0 or len(pcd2) == 0:
            return None
        
        # Calculate pairwise distances
        # pcd1: N x 3, pcd2: M x 3
        # distances: N x M
        diff = pcd1[:, np.newaxis, :] - pcd2[np.newaxis, :, :]  # N x M x 3
        distances = np.sqrt(np.sum(diff ** 2, axis=2))  # N x M
        
        # Return minimum distance
        return float(np.min(distances))
    except Exception:
        return None


def is_inside_point_cloud(src_pts, target_pts, thresh: float = INSIDE_THRESH) -> bool:
    """
    Check if source points are inside target points (based on REFLECT logic)
    
    Args:
        src_pts: Source point cloud (N x 3)
        target_pts: Target point cloud (M x 3)
        thresh: Threshold ratio (default 0.5)
        
    Returns:
        True if enough source points are inside target
    """
    if src_pts is None or target_pts is None or len(src_pts) == 0 or len(target_pts) == 0:
        return False
    
    try:
        src_pts = np.array(src_pts)
        target_pts = np.array(target_pts)
        
        # Get bounding box of target
        target_min = np.min(target_pts, axis=0)
        target_max = np.max(target_pts, axis=0)
        
        # Count how many source points are inside target bbox
        inside_mask = np.all((src_pts >= target_min) & (src_pts <= target_max), axis=1)
        inside_ratio = np.sum(inside_mask) / len(src_pts)
        
        return inside_ratio >= thresh
    except Exception:
        return False


def calculate_camera_space_vector(pos1: Tuple[float, float, float],
                                 pos2: Tuple[float, float, float],
                                 camera_world_xyz: Optional[Tuple[float, float, float]] = None,
                                 rotation: Optional[float] = None,
                                 horizon: Optional[float] = None) -> Optional[np.ndarray]:
    """
    Calculate normalized vector in camera space (from REFLECT)
    
    Args:
        pos1: Position 1 (world space)
        pos2: Position 2 (world space)
        camera_world_xyz: Camera position in world space
        rotation: Camera rotation (yaw)
        horizon: Camera horizon angle
        
    Returns:
        Normalized vector in camera space or None
    """
    if camera_world_xyz is None or rotation is None or horizon is None:
        # Fallback: simple world space vector
        vec = np.array([pos2[i] - pos1[i] for i in range(3)])
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return None
    
    try:
        # Try to use REFLECT's coordinate transformation if available
        try:
            from reflect.main.point_cloud_utils import world_space_xyz_to_camera_space_xyz
            import torch
            
            pos1_tensor = torch.tensor(np.array([pos1])).reshape(3, 1)
            pos2_tensor = torch.tensor(np.array([pos2])).reshape(3, 1)
            camera_xyz = torch.tensor(camera_world_xyz)
            
            cam_pos1 = world_space_xyz_to_camera_space_xyz(pos1_tensor, camera_xyz, rotation, horizon).flatten()
            cam_pos2 = world_space_xyz_to_camera_space_xyz(pos2_tensor, camera_xyz, rotation, horizon).flatten()
            
            vec = cam_pos2 - cam_pos1
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm
            return None
        except ImportError:
            # Fallback: simple world space vector
            vec = np.array([pos2[i] - pos1[i] for i in range(3)])
            norm = np.linalg.norm(vec)
            if norm > 0:
                return vec / norm
            return None
    except Exception:
        return None


def determine_spatial_relation_hybrid(
    obj1: Dict, obj2: Dict, node1: Node, node2: Node,
    use_point_cloud: bool = False
) -> Optional[Tuple[str, float]]:
    """
    Hybrid method to determine spatial relation: Prioritize metadata, fallback to point cloud/position
    
    Priority:
    1. Metadata-based (parentReceptacles) - Highest confidence (1.0)
    2. Position-based (CRAFT method) - Medium confidence (0.85)
    3. Point cloud-based (REFLECT method) - Lower confidence (0.75)
    
    Args:
        obj1: Object 1 metadata dictionary
        obj2: Object 2 metadata dictionary
        node1: Node 1
        node2: Node 2
        use_point_cloud: Whether to use point cloud data
        
    Returns:
        Tuple of (relation_type, confidence) or None
        - relation_type: "inside", "on_top_of", or None
        - confidence: 0.0 to 1.0
    """
    # Priority 1: Based on parentReceptacles metadata (CRAFT method, highest confidence)
    # Check if obj1 is inside obj2 based on metadata
    if obj1.get('parentReceptacles'):
        for parent_id in obj1.get('parentReceptacles', []):
            if obj2.get('objectId') == parent_id:
                # Dynamic relation type judgment (CRAFT's dynamic approach)
                has_receptacle = bool(obj2.get('receptacleObjectIds', []))
                is_openable_container = 'isOpen' in obj2 or obj2.get('openable', False)
                receptacle_count = len(obj2.get('receptacleObjectIds', [])) if isinstance(obj2.get('receptacleObjectIds'), list) else 0
                
                # 改进：Sink, SinkBasin 等容器类型即使 receptacleObjectIds 为空也应被识别为容器
                obj2_type = obj2.get('objectType', '').lower()
                is_container_type = any(kw in obj2_type for kw in ['sink', 'sinkbasin', 'bowl', 'pot', 'pan', 'mug', 'cup'])
                
                if is_openable_container or (has_receptacle and receptacle_count > 0) or is_container_type:
                    return ("inside", 1.0)  # Highest confidence
                else:
                    return ("on_top_of", 1.0)  # Highest confidence
    
    # Priority 2: Based on position information (CRAFT method, medium confidence)
    if node1.position is not None and node2.position is not None:
        pos1 = node1.position
        pos2 = node2.position
        
        z_diff = pos1[2] - pos2[2]
        horizontal_dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
        # Dynamic surface type detection (CRAFT's dynamic approach)
        obj2_type = obj2.get('objectType', '').lower()
        is_surface = any(kw in obj2_type for kw in ['countertop', 'table', 'stoveburner', 'burner', 'sink'])
        
        # CRAFT's on_top_of criteria
        if (CRAFT_Z_DIFF_MIN < z_diff < CRAFT_Z_DIFF_MAX and 
            horizontal_dist < CRAFT_HORIZONTAL_DIST_MAX and 
            is_surface):
            return ("on_top_of", 0.85)  # Medium confidence
    
    # Priority 3: Based on point cloud (REFLECT method, lower confidence, if available)
    if use_point_cloud and node1.pcd is not None and node2.pcd is not None:
        dist = get_point_cloud_distance(node1.pcd, node2.pcd)
        if dist is not None and dist < IN_CONTACT_DISTANCE:
            # Check inside relation using point cloud
            if is_inside_point_cloud(node1.pcd, node2.pcd, INSIDE_THRESH):
                # Dynamic surface type detection (avoiding hardcoded checks)
                obj2_type = obj2.get('objectType', '').lower()
                is_surface = any(kw in obj2_type for kw in ['countertop', 'stoveburner', 'burner', 'sink', 'table'])
                if is_surface:
                    return ("on_top_of", 0.75)  # Lower confidence (point cloud)
                else:
                    return ("inside", 0.75)  # Lower confidence (point cloud)
            
            # Check on_top_of using point cloud (REFLECT method)
            # This requires corner points (bounding box) which we might not have
            # Skip this for now as it requires additional data
    
        return None


def add_rich_spatial_relations(sg: SceneGraph, objects: List[Dict], 
                              use_point_cloud: bool = False,
                              camera_world_xyz: Optional[Tuple[float, float, float]] = None,
                              rotation: Optional[float] = None,
                              horizon: Optional[float] = None) -> None:
    """
    Add rich spatial relations (above, below, blocking, left_of, right_of)
    using hybrid method: Prioritize metadata, fallback to point cloud/position
    
    Args:
        sg: Scene graph to add relations to
        objects: List of object metadata dictionaries
        use_point_cloud: Whether to use point cloud data for calculations
        camera_world_xyz: Camera position (for camera space calculations)
        rotation: Camera rotation (for camera space calculations)
        horizon: Camera horizon (for camera space calculations)
    """
    for i, obj1 in enumerate(objects):
        node1 = sg.get_node(obj1.get('name', 'unknown'))
        if not node1:
            continue
        
        for j, obj2 in enumerate(objects):
            if i == j:
                continue
            
            node2 = sg.get_node(obj2.get('name', 'unknown'))
            if not node2:
                continue
            
            # Skip if object is being held
            if obj1.get('isPickedUp', False) or obj2.get('isPickedUp', False):
                continue
            
            # Check if relation already exists
            edge_key = (node1.name, node2.name)
            if edge_key in sg.edges:
                continue  # Skip if relation already determined by metadata
            
            # Try hybrid method to determine inside/on_top_of
            # (Note: This is a fallback - metadata-based relations should already be added in Step 2)
            relation_result = determine_spatial_relation_hybrid(
                obj1, obj2, node1, node2, use_point_cloud
            )
            
            if relation_result:
                rel_type, confidence = relation_result
                # Only add if not already exists and confidence is reasonable
                if edge_key not in sg.edges and confidence >= 0.75:
                    sg.add_edge(Edge(node1, node2, rel_type))
            
            # Calculate distance for rich relations
            if use_point_cloud and node1.pcd is not None and node2.pcd is not None:
                dist = get_point_cloud_distance(node1.pcd, node2.pcd)
            elif node1.position is not None and node2.position is not None:
                pos1 = node1.position
                pos2 = node2.position
                dist = np.sqrt(sum((pos1[k] - pos2[k])**2 for k in range(3)))
            else:
                continue
            
            if dist is None:
                continue
            
            # IN CONTACT relations (distance < 0.1m) - Additional refinement
            if dist < IN_CONTACT_DISTANCE:
                # Use point cloud for additional inside/on_top_of refinement if available
                if use_point_cloud and node1.pcd is not None and node2.pcd is not None:
                    if is_inside_point_cloud(node1.pcd, node2.pcd, INSIDE_THRESH):
                        # Dynamic surface type detection
                        obj2_type = obj2.get('objectType', '').lower()
                        is_surface = any(kw in obj2_type for kw in ['countertop', 'stoveburner', 'burner', 'sink', 'table'])
                        
                        # Only add if not already exists (metadata takes priority)
                        if edge_key not in sg.edges:
                        if is_surface:
                                sg.add_edge(Edge(node1, node2, "on_top_of"))
                        else:
                                sg.add_edge(Edge(node1, node2, "inside"))
            
            # CLOSE TO relations (distance < 0.4m) - for above/below/left/right/blocking
            if dist < CLOSE_DISTANCE:
                # Calculate camera space vector for directional relations
                if node1.position is not None and node2.position is not None:
                    norm_vector = calculate_camera_space_vector(
                        node1.position, node2.position,
                        camera_world_xyz, rotation, horizon
                    )
                    
                    if norm_vector is not None:
                        # Above/Below relations
                        if abs(norm_vector[1]) > NORM_THRESH_UP_DOWN:
                            edge_key = (node1.name, node2.name)
                            if edge_key not in sg.edges:
                                if norm_vector[1] > 0:
                                    sg.add_edge(Edge(node1, node2, "above"))
                                else:
                                    sg.add_edge(Edge(node1, node2, "below"))
                        
                        # Left/Right relations
                        elif abs(norm_vector[0]) > NORM_THRESH_LEFT_RIGHT:
                            edge_key = (node1.name, node2.name)
                            if edge_key not in sg.edges:
                                if norm_vector[0] > 0:
                                    sg.add_edge(Edge(node1, node2, "right_of"))
                                else:
                                    sg.add_edge(Edge(node1, node2, "left_of"))
                        
                        # Blocking relation (based on occlusion)
                        elif (abs(norm_vector[2]) > NORM_THRESH_FRONT_BACK and
                              hasattr(node1, 'bbox2d') and hasattr(node2, 'bbox2d') and
                              hasattr(node1, 'depth') and hasattr(node2, 'depth') and
                              getattr(node1, 'bbox2d', None) is not None and 
                              getattr(node2, 'bbox2d', None) is not None and
                              getattr(node1, 'depth', None) is not None and 
                              getattr(node2, 'depth', None) is not None):
                            # Calculate IoU and occlusion ratio
                            try:
                                bbox1 = np.array(getattr(node1, 'bbox2d')).flatten()
                                bbox2 = np.array(getattr(node2, 'bbox2d')).flatten()
                                
                                if len(bbox1) >= 4 and len(bbox2) >= 4:
                                    # Calculate IoU
                                    ixmin = max(bbox1[0], bbox2[0])
                                    ixmax = min(bbox1[2], bbox2[2])
                                    iymin = max(bbox1[1], bbox2[1])
                                    iymax = min(bbox1[3], bbox2[3])
                                    
                                    iw = max(ixmax - ixmin, 0)
                                    ih = max(iymax - iymin, 0)
                                    inters = iw * ih
                                    
                                    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                                    if area2 > 0:
                                        occlude_ratio = inters / area2
                                        
                                        # Check depth occlusion
                                        depth1 = np.array(getattr(node1, 'depth'))
                                        depth2 = np.array(getattr(node2, 'depth'))
                                        if len(depth1) > 0 and len(depth2) > 0:
                                            depth_occlude = np.sum(depth1 <= np.min(depth2)) / len(depth1)
                                            
                                            if (occlude_ratio > OCCLUDE_RATIO_THRESH and
                                                depth_occlude > DEPTH_THRESH):
                                                edge_key = (node1.name, node2.name)
                                                if edge_key not in sg.edges:
                                                    sg.add_edge(Edge(node1, node2, "blocking"))
                            except Exception:
                                pass
