"""
Constraint Evaluator Module
Evaluates executable AST expressions against scene graphs
"""

from typing import Dict, List, Optional, Tuple, Any
from ..core.scene_graph import SceneGraph, Node
import re


class ConstraintEvaluator:
    """Evaluates constraint AST expressions against scene graphs"""
    
    def __init__(self):
        pass
    
    def evaluate(self, condition_expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """
        Evaluate an AST expression against a scene graph
        
        Args:
            condition_expr: AST expression string, e.g., "(inside mug sink)"
            scene_graph: Scene graph to evaluate against
            
        Returns:
            (is_satisfied, reason, confidence) tuple
            - is_satisfied: True if constraint is satisfied
            - reason: Explanation of the result
            - confidence: Confidence level (0.0-1.0)
        """
        if not condition_expr:
            return False, "Empty condition expression", 0.0
        
        # Normalize expression
        expr = condition_expr.strip()
        
        # Remove outer parentheses if present
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()
        
        # Parse AST expression
        try:
            return self._evaluate_ast(expr, scene_graph)
        except Exception as e:
            return False, f"Evaluation error: {str(e)}", 0.0
    
    def _evaluate_ast(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """Evaluate AST expression recursively"""
        expr = expr.strip()
        
        # Handle negation: (not ...)
        if expr.startswith('not'):
            inner = expr[3:].strip()
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1].strip()
            result, reason, conf = self._evaluate_ast(inner, scene_graph)
            return not result, f"Negation: {reason}", conf
        
        # Handle logical AND: (and ...)
        if expr.startswith('and'):
            return self._evaluate_and(expr, scene_graph)
        
        # Handle logical OR: (or ...)
        if expr.startswith('or'):
            return self._evaluate_or(expr, scene_graph)
        
        # Handle atomic predicates
        return self._evaluate_atom(expr, scene_graph)
    
    def _evaluate_and(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """Evaluate AND expression: (and expr1 expr2 ...)"""
        # Extract arguments
        args = self._parse_args(expr[3:].strip())
        results = []
        reasons = []
        confidences = []
        
        for arg in args:
            result, reason, conf = self._evaluate_ast(arg, scene_graph)
            results.append(result)
            reasons.append(reason)
            confidences.append(conf)
        
        all_satisfied = all(results)
        combined_reason = " AND ".join([f"({r})" for r in reasons])
        min_confidence = min(confidences) if confidences else 1.0
        
        return all_satisfied, combined_reason, min_confidence
    
    def _evaluate_or(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """Evaluate OR expression: (or expr1 expr2 ...)"""
        # Extract arguments
        args = self._parse_args(expr[2:].strip())
        results = []
        reasons = []
        confidences = []
        
        for arg in args:
            result, reason, conf = self._evaluate_ast(arg, scene_graph)
            results.append(result)
            reasons.append(reason)
            confidences.append(conf)
        
        any_satisfied = any(results)
        combined_reason = " OR ".join([f"({r})" for r in reasons])
        max_confidence = max(confidences) if confidences else 1.0
        
        return any_satisfied, combined_reason, max_confidence
    
    def _evaluate_atom(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """Evaluate atomic predicate"""
        # Parse predicate: predicate_name arg1 arg2 ...
        parts = expr.split()
        if not parts:
            return False, "Empty predicate", 0.0
        
        predicate = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Location predicates
        if predicate in ['inside', 'in']:
            return self._check_location(expr, scene_graph, 'inside')
        elif predicate in ['on_top_of', 'on']:
            return self._check_location(expr, scene_graph, 'on_top_of')
        
        # State predicates
        elif predicate == 'eq':
            return self._check_equality(expr, scene_graph)
        elif predicate == 'empty':
            return self._check_empty(expr, scene_graph)  # Special handling for container empty
        elif predicate in ['open', 'closed', 'filled', 'clean']:
            return self._check_state(expr, scene_graph, predicate)
        
        # Default: try keyword matching
        return self._check_keyword(expr, scene_graph)
    
    def _check_location(self, expr: str, scene_graph: SceneGraph, relation_type: str) -> Tuple[bool, str, float]:
        """Check location relationship: (inside obj container) or (on_top_of obj surface)"""
        # Extract object names from expression
        parts = expr.split()
        if len(parts) < 3:
            return False, f"Insufficient arguments for {relation_type}", 0.0
        
        obj_name = parts[1].lower()
        location_name = parts[2].lower()
        
        # Find nodes
        obj_node = None
        location_node = None
        
        for node in scene_graph.nodes:
            if obj_name in node.name.lower() or node.name.lower() in obj_name:
                obj_node = node
            if location_name in node.name.lower() or node.name.lower() in location_name:
                location_node = node
        
        if not obj_node:
            return False, f"Object '{obj_name}' not found", 0.0
        if not location_node:
            return False, f"Location '{location_name}' not found", 0.0
        
        # Check edge
        edge_key = (obj_node.name, location_node.name)
        reverse_key = (location_node.name, obj_node.name)
        
        if edge_key in scene_graph.edges:
            edge = scene_graph.edges[edge_key]
            if relation_type in edge.edge_type.lower():
                return True, f"{obj_node.name} is {relation_type} {location_node.name}", 1.0
        elif reverse_key in scene_graph.edges:
            edge = scene_graph.edges[reverse_key]
            if relation_type in edge.edge_type.lower():
                return True, f"{obj_node.name} is {relation_type} {location_node.name}", 1.0
        
        return False, f"{obj_node.name} is not {relation_type} {location_node.name}", 1.0
    
    def _check_state(self, expr: str, scene_graph: SceneGraph, state: str) -> Tuple[bool, str, float]:
        """Check object state: (empty obj) or (open obj)"""
        parts = expr.split()
        if len(parts) < 2:
            return False, f"Insufficient arguments for state check", 0.0
        
        obj_name = parts[1].lower()
        
        # Find node
        obj_node = None
        for node in scene_graph.nodes:
            if obj_name in node.name.lower() or node.name.lower() in obj_name:
                obj_node = node
                break
        
        if not obj_node:
            return False, f"Object '{obj_name}' not found", 0.0
        
        # Check state
        node_state = (obj_node.state or '').lower()
        if state in node_state or node_state == state:
            return True, f"{obj_node.name} has state '{state}'", 1.0
        else:
            return False, f"{obj_node.name} has state '{node_state}' but required '{state}'", 1.0
    
    def _check_equality(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """Check equality: (eq obj.state 'open')"""
        # Parse: eq obj.state 'value'
        parts = expr.split()
        if len(parts) < 3:
            return False, "Insufficient arguments for equality check", 0.0
        
        # Extract obj and value
        obj_attr = parts[1]
        value = parts[2].strip("'\"")
        
        if '.' in obj_attr:
            obj_name, attr = obj_attr.split('.')
        else:
            obj_name = obj_attr
            attr = 'state'
        
        # Find node
        obj_node = None
        for node in scene_graph.nodes:
            if obj_name.lower() in node.name.lower() or node.name.lower() in obj_name.lower():
                obj_node = node
                break
        
        if not obj_node:
            return False, f"Object '{obj_name}' not found", 0.0
        
        # Check attribute
        if attr == 'state':
            node_value = (obj_node.state or '').lower()
            if node_value == value.lower():
                return True, f"{obj_node.name}.{attr} == '{value}'", 1.0
            else:
                return False, f"{obj_node.name}.{attr} == '{node_value}' but required '{value}'", 1.0
        
        return False, f"Unknown attribute '{attr}'", 0.0
    
    def _check_empty(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """Check if container is empty: (empty container)"""
        parts = expr.split()
        if len(parts) < 2:
            return False, "Insufficient arguments for empty check", 0.0
        
        container_name = parts[1].lower()
        
        # Find container node
        container_node = None
        for node in scene_graph.nodes:
            node_name_lower = node.name.lower()
            if container_name in node_name_lower or node_name_lower in container_name:
                container_node = node
                break
        
        if not container_node:
            return False, f"Container '{container_name}' not found", 0.0
        
        # Check if any object is inside this container
        items_inside = []
        for (start_name, end_name), edge in scene_graph.edges.items():
            if edge.end.name == container_node.name and edge.edge_type in ['inside', 'in']:
                items_inside.append(edge.start.name)
        
        if items_inside:
            return False, f"Container '{container_node.name}' is not empty: {', '.join(items_inside)} inside", 1.0
        else:
            return True, f"Container '{container_node.name}' is empty", 1.0
    
    def _check_keyword(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
        """Fallback: keyword-based checking"""
        expr_lower = expr.lower()
        
        # Check for common keywords
        if 'empty' in expr_lower:
            for node in scene_graph.nodes:
                if node.name.lower() in expr_lower:
                    if node.state and 'empty' in node.state.lower():
                        return False, f"{node.name} is empty", 1.0
                    return True, f"{node.name} is not empty", 1.0
        
        return False, f"Cannot evaluate expression: {expr}", 0.0
    
    def _parse_args(self, expr: str) -> List[str]:
        """Parse arguments from expression"""
        # Simple parser: split by spaces, handle nested parentheses
        args = []
        current = ""
        depth = 0
        
        for char in expr:
            if char == '(':
                depth += 1
                current += char
            elif char == ')':
                depth -= 1
                current += char
            elif char == ' ' and depth == 0:
                if current.strip():
                    args.append(current.strip())
                current = ""
            else:
                current += char
        
        if current.strip():
            args.append(current.strip())
        
        return args

