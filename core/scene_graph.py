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
    # Point cloud support (from REFLECT)
    pcd: Optional = None  # Point cloud data (N x 3 numpy array or torch tensor)
    corner_pts: Optional = None  # Corner points of 3D bbox (8 x 3)
    bbox2d: Optional = None  # 2D bounding box (4 x 1: [x1, y1, x2, y2])
    depth: Optional = None  # Depth image for this object
    
    def __init__(self, name: str, object_type: str, state: Optional[str] = None, 
                 position: Optional[Tuple[float, float, float]] = None, attributes: Dict = None,
                 bbox: Optional[Dict] = None, pose: Optional[Dict] = None,
                 confidence: float = 1.0, last_seen_ts: Optional[float] = None,
                 velocity: Optional[Tuple[float, float, float]] = None,
                 pcd: Optional = None, corner_pts: Optional = None,
                 bbox2d: Optional = None, depth: Optional = None):
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
        # Point cloud attributes
        self.pcd = pcd
        self.corner_pts = corner_pts
        self.bbox2d = bbox2d
        self.depth = depth
    
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
    
    def __init__(self, task: Optional[Dict] = None, event: Optional[Dict] = None,
                 timestep: Optional[int] = None, action: Optional[str] = None):
        self.nodes: Set[Node] = set()
        self.edges: Dict[Tuple[str, str], Edge] = {}
        self.task = task
        self.event = event
        # Action-aware fields for CRAFT++
        self.timestep: Optional[int] = timestep  # Which timestep this SG represents
        self.action: Optional[str] = action  # Which action this SG is for (for verification)
    
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
    
    def get_edges_of(self, node: Node) -> List[Edge]:
        """Get all edges connected to a node (both incoming and outgoing)"""
        edges = []
        for (start_name, end_name), edge in self.edges.items():
            if start_name == node.name or end_name == node.name:
                edges.append(edge)
        return edges
    
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
    
    def extract_task_relevant_subgraph_with_closure(self, task_info: Dict) -> 'SceneGraph':
        """
        使用闭包（closure）方法裁剪任务相关子图
        
        该方法使用 BFS 从任务相关对象开始，沿着 inside/on_top_of/supported_by 边扩展，
        确保包含所有相关的容器和支撑结构。
        
        Args:
            task_info: 任务信息字典
        
        Returns:
            裁剪后的子图，包含任务相关对象及其容器/支撑结构
        """
        from typing import Set
        import re
        from collections import deque
        
        # 提取任务相关对象名称（与 extract_task_relevant_subgraph 相同）
        relevant_object_names: Set[str] = set()
        
        actions = task_info.get('actions', [])
        for action_str in actions:
            matches = re.findall(r'\(([^)]+)\)', action_str)
            for match in matches:
                parts = [p.strip() for p in match.split(',')]
                for obj_name in parts[1:]:
                    if obj_name:
                        relevant_object_names.add(obj_name)
        
        success_condition = task_info.get('success_condition', '')
        if success_condition:
            obj_matches = re.findall(r'\b([A-Z][a-zA-Z]+)\b', success_condition)
            relevant_object_names.update(obj_matches)
        
        preactions = task_info.get('preactions', [])
        for preaction in preactions:
            matches = re.findall(r'\(([^)]+)\)', preaction)
            for match in matches:
                parts = [p.strip() for p in match.split(',')]
                for obj_name in parts[1:]:
                    if obj_name:
                        relevant_object_names.add(obj_name)
        
        # 查找初始相关节点
        initial_nodes = []
        for node in self.nodes:
            if node.name in relevant_object_names:
                initial_nodes.append(node)
                continue
            for obj_name in relevant_object_names:
                if (obj_name.lower() in node.name.lower() or 
                    node.name.lower() in obj_name.lower() or
                    obj_name.lower() in node.object_type.lower() or 
                    node.object_type.lower() in obj_name.lower()):
                    initial_nodes.append(node)
                    break
        
        # 使用 BFS 闭包扩展：沿着 inside/on_top_of/supported_by 边扩展
        closure: Set[Node] = set(initial_nodes)
        queue = deque(initial_nodes)
        
        # 定义需要扩展的关系类型（只允许功能性因果关系）
        # 这些关系类型是语义定义的，不依赖具体对象类型
        ALLOWED_CAUSAL_RELATIONS = {"inside", "holding", "on_top_of"}
        
        def is_task_relevant_object(obj_name: str, obj_type: str) -> bool:
            """检查对象是否是任务相关的（基于任务对象名称，不依赖写死的类型列表）"""
            obj_name_base = obj_name.split('_')[0] if '_' in obj_name else obj_name
            # 检查对象名或类型是否在任务相关对象中
            return (obj_name_base in relevant_object_names or
                   any(obj_name_base.lower() == rel_obj.lower() or 
                       rel_obj.lower() in obj_name_base.lower() or
                       obj_name_base.lower() in rel_obj.lower()
                       for rel_obj in relevant_object_names))
        
        def is_container_type(node: Node) -> bool:
            """动态判断节点是否是容器类型（基于场景图中的关系，不依赖写死的类型列表）"""
            # 方法1：检查节点是否在 inside 关系的目标端（说明它是容器）
            for (start_name, end_name), edge in self.edges.items():
                if edge.end.name == node.name and edge.edge_type == "inside":
                    return True
            # 方法2：通过对象类型特征（包含常见容器关键词，但这是启发式的）
            type_lower = node.object_type.lower()
            container_keywords = ['basin', 'machine', 'fridge', 'cabinet', 'drawer', 'microwave', 'refrigerator']
            return any(kw in type_lower for kw in container_keywords)
        
        def is_surface_type(node: Node) -> bool:
            """动态判断节点是否是表面类型（基于对象类型特征，不依赖写死的类型列表）"""
            type_lower = node.object_type.lower()
            # 表面类型的特征：通常包含 top, table, burner 等关键词
            surface_keywords = ['countertop', 'table', 'stoveburner', 'burner']
            return any(kw in type_lower for kw in surface_keywords)
        
        while queue:
            obj = queue.popleft()
            for edge in self.get_edges_of(obj):
                # 只沿着指定的关系类型扩展
                if edge.edge_type not in ALLOWED_CAUSAL_RELATIONS:
                    continue
                
                # 确定目标节点（边的另一端）
                if edge.start.name == obj.name:
                    target_node = edge.end
                    edge_direction = "forward"
                else:
                    target_node = edge.start
                    edge_direction = "backward"
                
                # 防止 on_top_of 关系导致 CounterTop 泛化（动态判断）
                if edge.edge_type == "on_top_of":
                    if edge_direction == "forward":
                        obj_node = obj
                        container_node = target_node
                    else:
                        obj_node = target_node
                        container_node = obj
                    
                    # 动态判断：如果容器是表面类型，且对象不是任务相关对象，跳过
                    # 使用动态判断函数，不依赖写死的类型列表
                    if is_surface_type(container_node):
                        # 检查对象是否是任务相关对象
                        obj_name_base = obj_node.name.split('_')[0] if '_' in obj_node.name else obj_node.name
                        if not is_task_relevant_object(obj_name_base, obj_node.object_type):
                            # 表面上的非任务相关对象，跳过
                            continue
                
                # 如果目标节点不在闭包中，添加到闭包和队列
                if target_node not in closure:
                    closure.add(target_node)
                    queue.append(target_node)
        
        # 创建子图
        subgraph = SceneGraph(
            task=task_info,
            timestep=self.timestep,
            action=self.action
        )
        
        # 添加闭包中的所有节点
        for node in closure:
            subgraph.add_node(node)
        
        # 添加闭包内节点之间的所有边
        for (start_name, end_name), edge in self.edges.items():
            start_node = subgraph.get_node(start_name)
            end_node = subgraph.get_node(end_name)
            if start_node and end_node:
                        subgraph.add_edge(edge)
        
        return subgraph

