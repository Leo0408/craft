"""
Scene Analyzer Module
Analyzes spatial relationships and object states
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
from ..core.scene_graph import SceneGraph, Node, Edge


class SceneAnalyzer:
    """Analyzes scene to extract spatial relationships and object states"""
    
    # Spatial relation thresholds
    IN_CONTACT_DISTANCE = 0.1
    CLOSE_DISTANCE = 0.4
    INSIDE_THRESH = 0.5
    ON_TOP_OF_THRESH = 0.7
    
    def __init__(self):
        pass
    
    def compute_spatial_relations(self, detections: List[Dict]) -> List[Tuple[str, str, str, float]]:
        """
        Compute spatial relationships between detected objects
        
        Args:
            detections: List of object detections with 3D positions
            
        Returns:
            List of (obj1, obj2, relation_type, confidence) tuples
        """
        relations = []
        
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
                if distance < self.IN_CONTACT_DISTANCE:
                    relations.append((det1['label'], det2['label'], 'in_contact', 1.0))
                elif distance < self.CLOSE_DISTANCE:
                    # Check vertical relationship
                    z_diff = pos1[2] - pos2[2]
                    if z_diff > self.ON_TOP_OF_THRESH:
                        relations.append((det1['label'], det2['label'], 'on_top_of', 0.8))
                    elif z_diff < -self.ON_TOP_OF_THRESH:
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

