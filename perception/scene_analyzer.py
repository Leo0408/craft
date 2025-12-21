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
    CLOSE_DISTANCE = 0.4  # 0.4 m = 400 mm
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
        
        Args:
            detections: List of object detections with 3D positions
            
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
            close_thresh = self.CLOSE_DISTANCE * 1000  # 400 mm
            on_top_thresh = self.ON_TOP_OF_THRESH * 1000  # 700 mm
        else:
            # Thresholds are already in meters
            in_contact_thresh = self.IN_CONTACT_DISTANCE
            close_thresh = self.CLOSE_DISTANCE
            on_top_thresh = self.ON_TOP_OF_THRESH
        
        for i, det1 in enumerate(detections):
            if det1.get('position_3d') is None:
                continue
            
            pos1 = np.array(det1['position_3d'])
            
            for j, det2 in enumerate(detections):
                if i >= j or det2.get('position_3d') is None:
                    continue
                
                pos2 = np.array(det2['position_3d'])
                distance = np.linalg.norm(pos1 - pos2)
                
                # Determine relationship based on distance and positions
                if distance < in_contact_thresh:
                    relations.append((det1['label'], det2['label'], 'in_contact', 1.0))
                elif distance < close_thresh:
                    # Check vertical relationship
                    z_diff = pos1[2] - pos2[2]
                    if z_diff > on_top_thresh:
                        relations.append((det1['label'], det2['label'], 'on_top_of', 0.8))
                    elif z_diff < -on_top_thresh:
                        relations.append((det2['label'], det1['label'], 'on_top_of', 0.8))
                    else:
                        relations.append((det1['label'], det2['label'], 'near', 0.7))
        
        return relations
    
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

