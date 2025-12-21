"""
Environment Memory Module for Real-World Robot Execution

This module implements temporal memory for handling occlusion, noise, and sensor errors
in real-world environments. It maintains object states across frames using Kalman filtering
and occlusion prediction.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class ObjectMemory:
    """Memory state for a single object"""
    name: str
    object_type: str
    
    # Position and motion
    position: Optional[np.ndarray] = None  # 3D position (x, y, z)
    velocity: Optional[np.ndarray] = None  # 3D velocity
    smoothed_position: Optional[np.ndarray] = None  # Kalman-filtered position
    
    # Detection confidence
    confidence: float = 0.0
    confidence_history: List[float] = field(default_factory=list)
    
    # Temporal information
    last_seen_ts: Optional[float] = None
    first_seen_ts: Optional[float] = None
    visibility_count: int = 0
    
    # Occlusion handling
    occluded: bool = False
    occlusion_duration: float = 0.0  # seconds since last seen
    
    # State attributes (from scene graph)
    attributes: Dict = field(default_factory=dict)
    
    # Bounding box (for tracking)
    bbox: Optional[np.ndarray] = None  # [x1, y1, x2, y2] or 3D bbox
    
    # Kalman filter state (simplified)
    position_variance: float = 0.1  # Position uncertainty
    velocity_variance: float = 0.05  # Velocity uncertainty


@dataclass
class RelationMemory:
    """Memory state for a spatial relation"""
    obj1_name: str
    obj2_name: str
    relation_type: str
    
    confidence: float = 0.0
    confidence_history: List[float] = field(default_factory=list)
    
    last_seen_ts: Optional[float] = None
    stable_count: int = 0  # Number of consecutive frames with this relation


class EnvironmentMemory:
    """
    Environment Memory for real-world robot execution
    
    Handles:
    - Object tracking across frames
    - Occlusion prediction and handling
    - Position smoothing using Kalman filtering
    - Confidence decay for unseen objects
    - Relation stability tracking
    """
    
    def __init__(self, 
                 position_noise: float = 0.05,
                 confidence_decay_rate: float = 0.1,
                 occlusion_threshold: float = 2.0):
        """
        Initialize Environment Memory
        
        Args:
            position_noise: Expected position noise (meters)
            confidence_decay_rate: Confidence decay per second when occluded
            occlusion_threshold: Time (seconds) before marking as occluded
        """
        self.object_memories: Dict[str, ObjectMemory] = {}
        self.relation_memories: Dict[Tuple[str, str], RelationMemory] = {}
        
        self.position_noise = position_noise
        self.confidence_decay_rate = confidence_decay_rate
        self.occlusion_threshold = occlusion_threshold
        
        self.current_time = time.time()
        self.frame_count = 0
    
    def update(self, 
               detections: List[Dict],
               relations: List[Tuple[str, str, str, float]],
               timestamp: Optional[float] = None) -> Dict:
        """
        Update memory with new detections and relations
        
        Args:
            detections: List of detections with keys: name, position, confidence, bbox, attributes
            relations: List of (obj1, obj2, relation_type, confidence) tuples
            timestamp: Current timestamp (defaults to current time)
        
        Returns:
            Dictionary with smoothed world state
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.current_time = timestamp
        self.frame_count += 1
        
        # Update object memories
        detected_names = set()
        for det in detections:
            obj_name = det.get('name') or det.get('label')
            if obj_name is None:
                continue
            
            detected_names.add(obj_name)
            
            # Get or create memory
            if obj_name not in self.object_memories:
                self.object_memories[obj_name] = ObjectMemory(
                    name=obj_name,
                    object_type=det.get('object_type', obj_name),
                    first_seen_ts=timestamp
                )
            
            mem = self.object_memories[obj_name]
            
            # Update position with Kalman-like smoothing
            new_position = det.get('position_3d') or det.get('position')
            if new_position is not None:
                new_position = np.array(new_position)
                
                if mem.position is None:
                    # First detection
                    mem.position = new_position
                    mem.smoothed_position = new_position
                    mem.velocity = np.zeros(3)
                else:
                    # Kalman-like update
                    position_diff = new_position - mem.position
                    
                    # Update velocity (simple exponential moving average)
                    if mem.velocity is None:
                        mem.velocity = position_diff / (timestamp - mem.last_seen_ts) if mem.last_seen_ts else np.zeros(3)
                    else:
                        alpha = 0.3  # Smoothing factor
                        dt = timestamp - mem.last_seen_ts if mem.last_seen_ts else 1.0
                        mem.velocity = alpha * (position_diff / dt) + (1 - alpha) * mem.velocity
                    
                    # Update position with smoothing
                    smoothing_factor = 0.7  # Higher = more smoothing
                    mem.position = smoothing_factor * mem.position + (1 - smoothing_factor) * new_position
                    mem.smoothed_position = mem.position
                    
                    # Check for teleportation (sudden large movement)
                    distance = np.linalg.norm(position_diff)
                    if distance > 0.3:  # 30cm threshold
                        # Likely sensor error or tracking loss, use new position but mark as uncertain
                        mem.position_variance = min(mem.position_variance * 2, 1.0)
            
            # Update confidence
            new_confidence = det.get('confidence', 1.0)
            mem.confidence = new_confidence
            mem.confidence_history.append(new_confidence)
            if len(mem.confidence_history) > 10:
                mem.confidence_history.pop(0)
            
            # Update temporal info
            mem.last_seen_ts = timestamp
            mem.visibility_count += 1
            mem.occluded = False
            mem.occlusion_duration = 0.0
            
            # Update attributes
            if 'attributes' in det:
                mem.attributes.update(det['attributes'])
            
            # Update bbox
            if 'bbox' in det:
                mem.bbox = np.array(det['bbox'])
        
        # Handle occluded objects (not detected in this frame)
        for obj_name, mem in self.object_memories.items():
            if obj_name not in detected_names:
                # Object not detected
                if mem.last_seen_ts is not None:
                    occlusion_duration = timestamp - mem.last_seen_ts
                    mem.occlusion_duration = occlusion_duration
                    
                    if occlusion_duration > self.occlusion_threshold:
                        mem.occluded = True
                    
                    # Decay confidence
                    mem.confidence *= (1 - self.confidence_decay_rate * occlusion_duration)
                    mem.confidence = max(0.0, mem.confidence)
                    
                    # Predict position if we have velocity
                    if mem.velocity is not None and mem.position is not None:
                        dt = timestamp - mem.last_seen_ts
                        predicted_position = mem.position + mem.velocity * dt
                        mem.smoothed_position = predicted_position
                        # Increase uncertainty
                        mem.position_variance = min(mem.position_variance * 1.1, 1.0)
        
        # Update relation memories
        for obj1, obj2, rel_type, conf in relations:
            key = (obj1, obj2)
            
            if key not in self.relation_memories:
                self.relation_memories[key] = RelationMemory(
                    obj1_name=obj1,
                    obj2_name=obj2,
                    relation_type=rel_type
                )
            
            rel_mem = self.relation_memories[key]
            
            # Update confidence
            rel_mem.confidence = conf
            rel_mem.confidence_history.append(conf)
            if len(rel_mem.confidence_history) > 10:
                rel_mem.confidence_history.pop(0)
            
            rel_mem.last_seen_ts = timestamp
            rel_mem.stable_count += 1
        
        # Build smoothed world state
        return self.get_world_state()
    
    def get_world_state(self) -> Dict:
        """
        Get current smoothed world state
        
        Returns:
            Dictionary with objects, relations, and metadata
        """
        objects = {}
        for name, mem in self.object_memories.items():
            objects[name] = {
                'name': mem.name,
                'object_type': mem.object_type,
                'position': mem.smoothed_position.tolist() if mem.smoothed_position is not None else None,
                'velocity': mem.velocity.tolist() if mem.velocity is not None else None,
                'confidence': mem.confidence,
                'occluded': mem.occluded,
                'occlusion_duration': mem.occlusion_duration,
                'last_seen_ts': mem.last_seen_ts,
                'attributes': mem.attributes.copy(),
                'position_variance': mem.position_variance
            }
        
        relations = {}
        for key, rel_mem in self.relation_memories.items():
            relations[key] = {
                'obj1': rel_mem.obj1_name,
                'obj2': rel_mem.obj2_name,
                'relation_type': rel_mem.relation_type,
                'confidence': rel_mem.confidence,
                'stable_count': rel_mem.stable_count,
                'last_seen_ts': rel_mem.last_seen_ts
            }
        
        return {
            'objects': objects,
            'relations': relations,
            'timestamp': self.current_time,
            'frame_count': self.frame_count
        }
    
    def get_object_state(self, obj_name: str) -> Optional[ObjectMemory]:
        """Get memory state for a specific object"""
        return self.object_memories.get(obj_name)
    
    def is_object_occluded(self, obj_name: str) -> bool:
        """Check if an object is currently occluded"""
        mem = self.object_memories.get(obj_name)
        return mem.occluded if mem else True
    
    def get_object_confidence(self, obj_name: str) -> float:
        """Get current confidence for an object"""
        mem = self.object_memories.get(obj_name)
        return mem.confidence if mem else 0.0

