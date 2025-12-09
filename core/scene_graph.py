"""
Scene Graph Module
Represents the robot's understanding of the environment as a graph structure
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Node:
    """Represents an object or entity in the scene"""
    name: str
    object_type: str
    state: Optional[str] = None  # e.g., "open", "closed", "filled"
    position: Optional[Tuple[float, float, float]] = None
    attributes: Dict = None
    # Enhanced attributes for CRAFT++
    bbox: Optional[Dict] = None  # Bounding box: {"min": [x,y,z], "max": [x,y,z]}
    pose: Optional[Dict] = None  # Pose: {"position": [x,y,z], "rotation": [x,y,z]}
    confidence: float = 1.0  # Detection confidence (0.0-1.0)
    last_seen_ts: Optional[float] = None  # Timestamp when last seen
    velocity: Optional[Tuple[float, float, float]] = None  # Velocity vector
    
    def __init__(self, name: str, object_type: str, state: Optional[str] = None, 
                 position: Optional[Tuple[float, float, float]] = None, attributes: Dict = None,
                 bbox: Optional[Dict] = None, pose: Optional[Dict] = None,
                 confidence: float = 1.0, last_seen_ts: Optional[float] = None,
                 velocity: Optional[Tuple[float, float, float]] = None):
        self.name = name
        self.object_type = object_type
        self.state = state
        self.position = position
        self.attributes = attributes or {}
        self.bbox = bbox
        self.pose = pose
        self.confidence = confidence
        self.last_seen_ts = last_seen_ts
        self.velocity = velocity
    
    def get_name(self) -> str:
        """Get full name with state if available"""
        if self.state:
            return f"{self.name} ({self.state})"
        return self.name
    
    def __hash__(self):
        return hash((self.name, self.object_type))
    
    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.name == other.name and self.object_type == other.object_type


@dataclass
class Edge:
    """Represents a relationship between two nodes"""
    start: Node
    end: Node
    edge_type: str  # e.g., "on", "inside", "near", "holding"
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.start.name, self.end.name, self.edge_type))


class SceneGraph:
    """Hierarchical scene graph representation of robot's environment"""
    
    def __init__(self, task: Optional[Dict] = None, event: Optional[Dict] = None):
        self.nodes: Set[Node] = set()
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.task = task
        self.event = event
    
    def add_node(self, node: Node):
        """Add a node to the scene graph"""
        self.nodes.add(node)
    
    def add_edge(self, edge: Edge):
        """Add an edge to the scene graph"""
        key = (edge.start.name, edge.end.name)
        self.edges[key] = edge
    
    def get_node(self, name: str) -> Optional[Node]:
        """Get a node by name"""
        for node in self.nodes:
            if node.name == name:
                return node
        return None
    
    def to_text(self) -> str:
        """Convert scene graph to natural language description"""
        output = ""
        
        # List all objects
        node_names = [node.get_name() for node in self.nodes]
        if node_names:
            output += ", ".join(node_names) + ". "
        
        # Describe relationships
        visited = set()
        for (start_name, end_name), edge in self.edges.items():
            reverse_key = (end_name, start_name)
            if (start_name, end_name) not in visited and reverse_key not in visited:
                output += f"{edge.start.get_name()} is {edge.edge_type} {edge.end.get_name()}. "
                visited.add((start_name, end_name))
        
        return output.strip()
    
    def __str__(self):
        return self.to_text()
    
    def __eq__(self, other):
        """Check if two scene graphs are equivalent"""
        if not isinstance(other, SceneGraph):
            return False
        return (self.nodes == other.nodes and 
                set(self.edges.keys()) == set(other.edges.keys()))

