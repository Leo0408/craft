"""
Scene Analyzer Module
Analyzes spatial relationships and object states
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from ..core.scene_graph import SceneGraph, Node, Edge


class SceneAnalyzer:
    """Analyzes scene to extract spatial relationships and object states"""
    
    # Spatial relation thresholds (in meters)
    # Note: These will be converted to mm if positions are in mm
    IN_CONTACT_DISTANCE = 0.1  # 0.1 m = 100 mm
    CLOSE_DISTANCE = 1.5  # 1.5 m = 1500 mm (increased from 0.4m to handle real-world distances)
    INSIDE_THRESH = 0.5  # 0.5 m = 500 mm
    ON_TOP_OF_THRESH = 0.7  # 0.7 m = 700 mm
    
    def __init__(self, position_unit: str = 'auto'):
        """
        Initialize SceneAnalyzer
        
        Args:
            position_unit: Unit of position coordinates ('m' for meters, 'mm' for millimeters, 'auto' for auto-detect)
        """
        self.position_unit = position_unit
    
    def compute_spatial_relations(self, detections: List[Dict]) -> List[Tuple[str, str, str, float]]:
        """
        Compute spatial relationships between detected objects
        
        Enhanced version that uses 3D bounding boxes for more accurate relations:
        - inside: Check if one object's bbox is inside another's
        - on_top_of: Check vertical relationship with proper thresholds
        - near: Objects are close but not in specific spatial relationship
        
        Args:
            detections: List of object detections with 3D positions and optional bbox3d
            
        Returns:
            List of (obj1, obj2, relation_type, confidence) tuples
        """
        relations = []
        
        if len(detections) == 0:
            return relations
        
        # Auto-detect unit if needed
        unit = self.position_unit
        if unit == 'auto':
            # Check first position to determine unit
            # If values are > 10, likely in mm; if < 10, likely in m
            sample_pos = None
            for det in detections:
                if det.get('position_3d') is not None:
                    sample_pos = np.array(det['position_3d'])
                    break
            
            if sample_pos is not None:
                # If any coordinate is > 10, assume mm; otherwise assume m
                if np.any(np.abs(sample_pos) > 10):
                    unit = 'mm'
                else:
                    unit = 'm'
            else:
                unit = 'm'  # Default to meters
        
        # Convert thresholds based on unit
        if unit == 'mm':
            # Thresholds are in meters, convert to mm
            in_contact_thresh = self.IN_CONTACT_DISTANCE * 1000  # 100 mm
            close_thresh = self.CLOSE_DISTANCE * 1000  # 1500 mm (increased from 400mm)
            on_top_thresh = 0.05 * 1000  # 50 mm (reduced for real-world, was 700mm)
            inside_overlap_ratio = 0.7  # 70% overlap for inside relation
        else:
            # Thresholds are already in meters
            in_contact_thresh = self.IN_CONTACT_DISTANCE
            close_thresh = self.CLOSE_DISTANCE
            on_top_thresh = 0.05  # 5 cm (reduced for real-world)
            inside_overlap_ratio = 0.7
        
        # Debug: Print thresholds
        print(f"  🔍 Spatial relation computation:")
        print(f"     Unit: {unit}")
        print(f"     Thresholds: close={close_thresh:.2f} ({unit}), contact={in_contact_thresh:.2f} ({unit})")
        print(f"     Detections: {len(detections)} objects")
        
        for i, det1 in enumerate(detections):
            if det1.get('position_3d') is None:
                continue
            
            pos1 = np.array(det1['position_3d'])
            bbox1 = det1.get('bbox3d')  # Optional: 3D bounding box
            
            for j, det2 in enumerate(detections):
                if i >= j or det2.get('position_3d') is None:
                    continue
                
                pos2 = np.array(det2['position_3d'])
                bbox2 = det2.get('bbox3d')  # Optional: 3D bounding box
                distance = np.linalg.norm(pos1 - pos2)
                
                # Debug: Print distance for first pair
                if i == 0 and j == 1:
                    print(f"     Distance between '{det1['label']}' and '{det2['label']}': {distance:.2f} ({unit})")
                    print(f"     Distance < close_thresh? {distance:.2f} < {close_thresh:.2f} = {distance < close_thresh}")
                
                # Priority 1: Check "inside" relation using 3D bounding boxes
                if bbox1 is not None and bbox2 is not None:
                    # Try to get bbox bounds (support both open3d bbox and dict format)
                    try:
                        if hasattr(bbox1, 'get_min_bound') and hasattr(bbox1, 'get_max_bound'):
                            # Open3D AxisAlignedBoundingBox
                            min1 = np.array(bbox1.get_min_bound())
                            max1 = np.array(bbox1.get_max_bound())
                            min2 = np.array(bbox2.get_min_bound())
                            max2 = np.array(bbox2.get_max_bound())
                        elif isinstance(bbox1, dict):
                            # Dict format: {'min': [x,y,z], 'max': [x,y,z]}
                            min1 = np.array(bbox1.get('min', bbox1.get('min_bound', [0,0,0])))
                            max1 = np.array(bbox1.get('max', bbox1.get('max_bound', [0,0,0])))
                            min2 = np.array(bbox2.get('min', bbox2.get('min_bound', [0,0,0])))
                            max2 = np.array(bbox2.get('max', bbox2.get('max_bound', [0,0,0])))
                        else:
                            raise AttributeError("Unknown bbox format")
                        
                        # Check if obj1 is inside obj2
                        inside_12 = self._check_inside(min1, max1, min2, max2, inside_overlap_ratio)
                        # Check if obj2 is inside obj1
                        inside_21 = self._check_inside(min2, max2, min1, max1, inside_overlap_ratio)
                        
                        if inside_12:
                            relations.append((det1['label'], det2['label'], 'inside', 0.9))
                            continue  # Skip other relations if inside is detected
                        elif inside_21:
                            relations.append((det2['label'], det1['label'], 'inside', 0.9))
                            continue
                    except Exception as e:
                        # If bbox check fails, fall back to distance-based method
                        pass
                
                # Priority 2: Check "on_top_of" relation
                if distance < close_thresh:
                    z_diff = pos1[2] - pos2[2]
                    
                    # Enhanced on_top_of check: also consider horizontal distance
                    horizontal_dist = np.linalg.norm(pos1[:2] - pos2[:2])
                    
                    # Object 1 is on top of object 2 if:
                    # - z_diff is positive and significant
                    # - horizontal distance is small (object is above, not just higher)
                    if z_diff > on_top_thresh and horizontal_dist < close_thresh * 0.5:
                        relations.append((det1['label'], det2['label'], 'on_top_of', 0.85))
                        continue
                    elif z_diff < -on_top_thresh and horizontal_dist < close_thresh * 0.5:
                        relations.append((det2['label'], det1['label'], 'on_top_of', 0.85))
                        continue
                
                # Priority 3: Check "in_contact" (very close)
                if distance < in_contact_thresh:
                    relations.append((det1['label'], det2['label'], 'in_contact', 1.0))
                    continue
                
                # Priority 4: "near" relation (close but no specific spatial relationship)
                if distance < close_thresh:
                    relations.append((det1['label'], det2['label'], 'near', 0.7))
                    # Debug: Print when near relation is added
                    if i == 0 and j == 1:
                        print(f"     ✅ Added 'near' relation between '{det1['label']}' and '{det2['label']}'")
        
        # Debug: Print final relations count
        print(f"     Total relations computed: {len(relations)}")
        if len(relations) > 0:
            print(f"     Relations: {relations[:3]}")  # Show first 3
        
        return relations
    
    def _check_inside(self, min1: np.ndarray, max1: np.ndarray, 
                     min2: np.ndarray, max2: np.ndarray, 
                     overlap_ratio: float = 0.7) -> bool:
        """
        Check if bbox1 (min1, max1) is inside bbox2 (min2, max2)
        
        Args:
            min1, max1: Bounding box 1 (smaller object)
            min2, max2: Bounding box 2 (container)
            overlap_ratio: Minimum overlap ratio to consider "inside" (0.0-1.0)
            
        Returns:
            True if bbox1 is inside bbox2
        """
        # Check if bbox1's center is inside bbox2
        center1 = (min1 + max1) / 2
        if not (np.all(center1 >= min2) and np.all(center1 <= max2)):
            return False
        
        # Check if significant portion of bbox1 is inside bbox2
        # Compute intersection
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        
        # Check if intersection is valid
        if not np.all(intersection_min < intersection_max):
            return False
        
        # Compute volumes
        intersection_vol = np.prod(np.maximum(intersection_max - intersection_min, 0))
        bbox1_vol = np.prod(np.maximum(max1 - min1, 0))
        
        if bbox1_vol == 0:
            return False
        
        # Check if overlap ratio is sufficient
        overlap = intersection_vol / bbox1_vol
        return overlap >= overlap_ratio
    
    def detect_object_state(self, detection: Dict, object_type: str) -> Optional[str]:
        """
        Detect the state of an object (e.g., open/closed, filled/empty)
        
        Args:
            detection: Object detection result
            object_type: Type of object
            
        Returns:
            Object state string or None
        """
        # Placeholder for state detection
        # In actual implementation, this would use CLIP or other vision models
        state_dict = {
            "Fridge": ["open", "closed"],
            "Faucet": ["turned on", "turned off"],
            "CoffeeMachine": ["turned on", "turned off", "open", "closed"],
            "Mug": ["filled", "empty", "dirty", "clean"],
        }
        
        if object_type in state_dict:
            # Mock: return first state as default
            # In real implementation, use vision model to determine state
            return state_dict[object_type][0]
        
        return None
    
    def build_scene_graph(self, detections: List[Dict], relations: List[Tuple], 
                         task_info: Dict) -> SceneGraph:
        """
        Build a scene graph from detections and relations
        
        Args:
            detections: List of object detections
            relations: List of spatial relations
            task_info: Task information
            
        Returns:
            SceneGraph object
        """
        scene_graph = SceneGraph(task=task_info)
        
        # Add nodes
        for det in detections:
            obj_type = det.get('object_type', det['label'])
            state = self.detect_object_state(det, obj_type)
            node = Node(
                name=det['label'],
                object_type=obj_type,
                state=state,
                position=det.get('position_3d')
            )
            scene_graph.add_node(node)
        
        # Add edges
        for obj1_name, obj2_name, relation_type, confidence in relations:
            node1 = scene_graph.get_node(obj1_name)
            node2 = scene_graph.get_node(obj2_name)
            
            if node1 and node2:
                edge = Edge(
                    start=node1,
                    end=node2,
                    edge_type=relation_type,
                    confidence=confidence
                )
                scene_graph.add_edge(edge)
        
        return scene_graph

