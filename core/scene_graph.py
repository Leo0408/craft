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
    
    def extract_task_relevant_subgraph(self, task_info: Dict) -> 'SceneGraph':
        """
        从完整场景图中裁剪出与当前子任务相关的最小子图
        
        该方法根据任务信息（actions、success_condition等）提取相关对象，
        只保留这些对象及其直接关系，从而减少场景图的复杂度。
        
        Args:
            task_info: 任务信息字典，包含:
                - actions: 动作列表，例如 ["(pick_up, Mug)", "(put_in, Mug, CoffeeMachine)"]
                - success_condition: 成功条件，例如 "a clean mug is filled with coffee"
                - preactions: 前置动作（可选）
        
        Returns:
            裁剪后的最小子图，只包含与任务相关的节点和边
        
        Algorithm:
            1. 从 actions 中提取所有对象名称（动作参数）
            2. 从 success_condition 中提取对象名称（大写开头的单词）
            3. 从 preactions 中提取对象名称（如果有）
            4. 在场景图中查找匹配的节点（支持精确匹配和部分匹配）
            5. 只保留相关节点及其之间的边
            6. 如果边的端点至少有一个在子图中，也保留该边（保留直接关系）
        """
        from typing import Set
        import re
        
        # 提取任务相关对象名称
        relevant_object_names: Set[str] = set()
        
        # 从 actions 中提取对象
        actions = task_info.get('actions', [])
        for action_str in actions:
            # 解析动作字符串，例如: "(pick_up, Mug)" 或 "(put_in, Mug, CoffeeMachine)"
            matches = re.findall(r'\(([^)]+)\)', action_str)
            for match in matches:
                parts = [p.strip() for p in match.split(',')]
                # 跳过第一个（动作类型），提取后面的对象名
                for obj_name in parts[1:]:
                    if obj_name:
                        relevant_object_names.add(obj_name)
        
        # 从 success_condition 中提取对象
        success_condition = task_info.get('success_condition', '')
        if success_condition:
            # 提取大写开头的单词（通常是对象名）
            obj_matches = re.findall(r'\b([A-Z][a-zA-Z]+)\b', success_condition)
            relevant_object_names.update(obj_matches)
        
        # 从 preactions 中提取对象
        preactions = task_info.get('preactions', [])
        for preaction in preactions:
            matches = re.findall(r'\(([^)]+)\)', preaction)
            for match in matches:
                parts = [p.strip() for p in match.split(',')]
                for obj_name in parts[1:]:
                    if obj_name:
                        relevant_object_names.add(obj_name)
        
        # 创建子图
        subgraph = SceneGraph()
        
        # 查找相关节点（支持部分匹配，因为对象名可能有变体）
        relevant_nodes = []
        for node in self.nodes:
            # 精确匹配
            if node.name in relevant_object_names:
                relevant_nodes.append(node)
                continue
            
            # 部分匹配（对象名称或类型匹配）
            for obj_name in relevant_object_names:
                if (obj_name.lower() in node.name.lower() or 
                    node.name.lower() in obj_name.lower() or
                    obj_name.lower() in node.object_type.lower() or 
                    node.object_type.lower() in obj_name.lower()):
                    relevant_nodes.append(node)
                    break
        
        # 添加相关节点到子图
        for node in relevant_nodes:
            subgraph.add_node(node)
        
        # 添加相关边（只保留两个端点都在子图中的边，或至少一个端点在子图中的边）
        for (start_name, end_name), edge in self.edges.items():
            start_node = subgraph.get_node(start_name)
            end_node = subgraph.get_node(end_name)
            
            if start_node and end_node:
                # 两个节点都在子图中，添加边
                subgraph.add_edge(edge)
            elif start_node or end_node:
                # 至少一个节点在子图中，也添加边（保留与相关对象的直接关系）
                if start_node:
                    # 如果终点不在子图中，尝试添加终点节点
                    end_node_in_full = self.get_node(end_name)
                    if end_node_in_full:
                        subgraph.add_node(end_node_in_full)
                        subgraph.add_edge(edge)
                elif end_node:
                    # 如果起点不在子图中，尝试添加起点节点
                    start_node_in_full = self.get_node(start_name)
                    if start_node_in_full:
                        subgraph.add_node(start_node_in_full)
                        subgraph.add_edge(edge)
        
        return subgraph

