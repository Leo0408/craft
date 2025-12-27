"""
Action-relevant Scene Graph Generation
只包含与当前 action 相关的节点，使用闭包方法扩展相关节点
"""

from typing import Dict, List, Set, Optional, Tuple
import re
from collections import deque
from .scene_graph import SceneGraph, Node, Edge


def extract_action_relevant_objects(action_str: str) -> Set[str]:
    """
    从单个 action 字符串中提取相关对象名称
    
    Args:
        action_str: 动作字符串，例如: "(pick_up, Mug)" 或 "(put_in, Mug, CoffeeMachine)"
        
    Returns:
        相关对象名称的集合
    """
    relevant_objects = set()
    
    # 解析动作字符串，例如: "(pick_up, Mug)" 或 "(put_in, Mug, CoffeeMachine)"
    matches = re.findall(r'\(([^)]+)\)', action_str)
    for match in matches:
        parts = [p.strip() for p in match.split(',')]
        # 跳过第一个（动作类型），提取后面的对象名
        for obj_name in parts[1:]:
            if obj_name:
                relevant_objects.add(obj_name)
    
    return relevant_objects


def extract_action_relevant_subgraph_with_closure(
    full_scene_graph: SceneGraph, 
    action_str: Optional[str] = None,
    task_info: Optional[Dict] = None
) -> SceneGraph:
    """
    使用闭包方法从完整场景图中裁剪出与当前 action 相关的子图
    
    该方法使用 BFS 从 action 相关对象开始，沿着 inside/on_top_of/holding 边扩展，
    确保包含所有相关的容器和支撑结构。
    
    Args:
        full_scene_graph: 完整的场景图
        action_str: 当前动作字符串，例如: "(pick_up, Mug)"
        task_info: 任务信息字典（可选，用于提取 success_condition 中的对象）
        
    Returns:
        裁剪后的子图，包含 action 相关对象及其容器/支撑结构
    """
    from typing import Set
    from collections import deque
    
    # 提取 action 相关对象名称
    relevant_object_names: Set[str] = set()
    
    # 从 action 中提取对象
    if action_str:
        relevant_object_names.update(extract_action_relevant_objects(action_str))
    
    # 如果提供了 task_info，也可以从 success_condition 中提取（可选）
    if task_info:
        success_condition = task_info.get('success_condition', '')
        if success_condition:
            obj_matches = re.findall(r'\b([A-Z][a-zA-Z]+)\b', success_condition)
            relevant_object_names.update(obj_matches)
    
    if not relevant_object_names:
        # 如果没有相关对象，返回空图
        return SceneGraph(
            task=task_info,
            timestep=full_scene_graph.timestep,
            action=action_str
        )
    
    # 查找初始相关节点
    initial_nodes = []
    for node in full_scene_graph.nodes:
        # 精确匹配
        if node.name in relevant_object_names:
            initial_nodes.append(node)
            continue
        
        # 部分匹配（对象名称或类型匹配）
        for obj_name in relevant_object_names:
            if (obj_name.lower() in node.name.lower() or 
                node.name.lower() in obj_name.lower() or
                obj_name.lower() in node.object_type.lower() or 
                node.object_type.lower() in obj_name.lower()):
                initial_nodes.append(node)
                break
    
    if not initial_nodes:
        # 如果没有找到相关节点，返回空图
        return SceneGraph(
            task=task_info,
            timestep=full_scene_graph.timestep,
            action=action_str
        )
    
    # 使用 BFS 闭包扩展：沿着 inside/on_top_of/holding 边扩展
    closure: Set[Node] = set(initial_nodes)
    queue = deque(initial_nodes)
    
    # 定义需要扩展的关系类型（只允许功能性因果关系）
    ALLOWED_CAUSAL_RELATIONS = {"inside", "holding", "on_top_of"}
    
    def is_action_relevant_object(obj_name: str, obj_type: str) -> bool:
        """检查对象是否是 action 相关的"""
        obj_name_base = obj_name.split('_')[0] if '_' in obj_name else obj_name
        return (obj_name_base in relevant_object_names or
               any(obj_name_base.lower() == rel_obj.lower() or 
                   rel_obj.lower() in obj_name_base.lower() or
                   obj_name_base.lower() in rel_obj.lower()
                   for rel_obj in relevant_object_names))
    
    def is_surface_type(node: Node) -> bool:
        """动态判断节点是否是表面类型"""
        type_lower = node.object_type.lower()
        surface_keywords = ['countertop', 'table', 'stoveburner', 'burner']
        return any(kw in type_lower for kw in surface_keywords)
    
    while queue:
        obj = queue.popleft()
        for edge in full_scene_graph.get_edges_of(obj):
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
            
            # 防止 on_top_of 关系导致 CounterTop 泛化
            if edge.edge_type == "on_top_of":
                if edge_direction == "forward":
                    obj_node = obj
                    container_node = target_node
                else:
                    obj_node = target_node
                    container_node = obj
                
                # 如果容器是表面类型，且对象不是 action 相关对象，跳过
                if is_surface_type(container_node):
                    obj_name_base = obj_node.name.split('_')[0] if '_' in obj_node.name else obj_node.name
                    if not is_action_relevant_object(obj_name_base, obj_node.object_type):
                        continue
            
            # 如果目标节点不在闭包中，添加到闭包和队列
            if target_node not in closure:
                closure.add(target_node)
                queue.append(target_node)
    
    # 创建子图
    subgraph = SceneGraph(
        task=task_info,
        timestep=full_scene_graph.timestep,
        action=action_str
    )
    
    # 添加闭包中的所有节点
    for node in closure:
        subgraph.add_node(node)
    
    # 添加闭包内节点之间的所有边
    for (start_name, end_name), edge in full_scene_graph.edges.items():
        start_node = subgraph.get_node(start_name)
        end_node = subgraph.get_node(end_name)
        if start_node and end_node:
            subgraph.add_edge(edge)
    
    return subgraph

