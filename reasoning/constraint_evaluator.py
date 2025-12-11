"""
Constraint Evaluator Module
Evaluates executable AST expressions against scene graphs
Enhanced with atom-level trace and confidence aggregation (CRAFT++ optimization)
"""

from typing import Dict, List, Optional, Tuple, Any
from ..core.scene_graph import SceneGraph, Node
import re


class AtomTrace:
    """Represents an atom-level evaluation result"""
    def __init__(self, atom_expr: str, value: bool, confidence: float, source: str, reason: str):
        self.atom_expr = atom_expr
        self.value = value
        self.confidence = confidence
        self.source = source  # e.g., "edge_relation", "geometry_iou", "state_check"
        self.reason = reason
    
    def __repr__(self):
        return f"AtomTrace({self.atom_expr}: {self.value}, conf={self.confidence:.2f}, source={self.source})"


class ConstraintEvaluator:
    """Evaluates constraint AST expressions against scene graphs with atom-level trace"""
    
    def __init__(self, min_confidence_threshold: float = 0.7):
        """
        Initialize constraint evaluator
        
        Args:
            min_confidence_threshold: Minimum confidence threshold for SATISFIED/VIOLATED (default: 0.7)
        """
        self.min_confidence_threshold = min_confidence_threshold
    
    def evaluate(self, condition_expr: str, scene_graph: SceneGraph, 
                 return_trace: bool = True) -> Tuple[bool, str, float, Optional[List[AtomTrace]]]:
        """
        Evaluate an AST expression against a scene graph with atom-level trace
        
        Args:
            condition_expr: AST expression string, e.g., "(inside mug sink)"
            scene_graph: Scene graph to evaluate against
            return_trace: If True, return atom-level trace (default: True)
            
        Returns:
            (is_satisfied, reason, confidence, atom_traces) tuple
            - is_satisfied: True if constraint is satisfied
            - reason: Explanation of the result
            - confidence: Aggregated confidence level (0.0-1.0)
            - atom_traces: List of AtomTrace objects (if return_trace=True)
        """
        if not condition_expr:
            return False, "Empty condition expression", 0.0, []
        
        # Normalize expression
        expr = condition_expr.strip()
        
        # Remove outer parentheses if present
        if expr.startswith('(') and expr.endswith(')'):
            expr = expr[1:-1].strip()
        
        # Parse AST expression
        try:
            return self._evaluate_ast(expr, scene_graph, return_trace=return_trace)
        except Exception as e:
            return False, f"Evaluation error: {str(e)}", 0.0, []
    
    def _evaluate_ast(self, expr: str, scene_graph: SceneGraph, 
                      return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Evaluate AST expression recursively with atom-level trace"""
        expr = expr.strip()
        
        # Handle negation: (not ...)
        if expr.startswith('not'):
            inner = expr[3:].strip()
            if inner.startswith('(') and inner.endswith(')'):
                inner = inner[1:-1].strip()
            result, reason, conf, traces = self._evaluate_ast(inner, scene_graph, return_trace=return_trace)
            # Negation: flip value, keep confidence and traces
            return not result, f"Negation: {reason}", conf, traces
        
        # Handle logical AND: (and ...)
        if expr.startswith('and'):
            return self._evaluate_and(expr, scene_graph, return_trace=return_trace)
        
        # Handle logical OR: (or ...)
        if expr.startswith('or'):
            return self._evaluate_or(expr, scene_graph, return_trace=return_trace)
        
        # Handle atomic predicates
        return self._evaluate_atom(expr, scene_graph, return_trace=return_trace)
    
    def _evaluate_and(self, expr: str, scene_graph: SceneGraph, 
                     return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Evaluate AND expression: (and expr1 expr2 ...)"""
        # Extract arguments
        args = self._parse_args(expr[3:].strip())
        results = []
        reasons = []
        confidences = []
        all_traces = []
        
        for arg in args:
            result, reason, conf, traces = self._evaluate_ast(arg, scene_graph, return_trace=return_trace)
            results.append(result)
            reasons.append(reason)
            confidences.append(conf)
            all_traces.extend(traces)
        
        all_satisfied = all(results)
        combined_reason = " AND ".join([f"({r})" for r in reasons])
        min_confidence = min(confidences) if confidences else 1.0
        
        return all_satisfied, combined_reason, min_confidence, all_traces
    
    def _evaluate_or(self, expr: str, scene_graph: SceneGraph, 
                   return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Evaluate OR expression: (or expr1 expr2 ...)"""
        # Extract arguments
        args = self._parse_args(expr[2:].strip())
        results = []
        reasons = []
        confidences = []
        all_traces = []
        
        for arg in args:
            result, reason, conf, traces = self._evaluate_ast(arg, scene_graph, return_trace=return_trace)
            results.append(result)
            reasons.append(reason)
            confidences.append(conf)
            all_traces.extend(traces)
        
        any_satisfied = any(results)
        combined_reason = " OR ".join([f"({r})" for r in reasons])
        max_confidence = max(confidences) if confidences else 1.0
        
        return any_satisfied, combined_reason, max_confidence, all_traces
    
    def _evaluate_atom(self, expr: str, scene_graph: SceneGraph, 
                      return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Evaluate atomic predicate with atom-level trace"""
        # Parse predicate: predicate_name arg1 arg2 ...
        parts = expr.split()
        if not parts:
            return False, "Empty predicate", 0.0, []
        
        predicate = parts[0].lower()
        args = parts[1:] if len(parts) > 1 else []
        
        # Location predicates
        if predicate in ['inside', 'in']:
            return self._check_location(expr, scene_graph, 'inside', return_trace=return_trace)
        elif predicate in ['on_top_of', 'on']:
            return self._check_location(expr, scene_graph, 'on_top_of', return_trace=return_trace)
        
        # State predicates
        elif predicate == 'eq':
            return self._check_equality(expr, scene_graph, return_trace=return_trace)
        elif predicate == 'empty':
            return self._check_empty(expr, scene_graph, return_trace=return_trace)
        elif predicate in ['open', 'closed', 'filled', 'clean']:
            return self._check_state(expr, scene_graph, predicate, return_trace=return_trace)
        
        # Default: try keyword matching
        return self._check_keyword(expr, scene_graph, return_trace=return_trace)
    
    def _check_location(self, expr: str, scene_graph: SceneGraph, relation_type: str, 
                       return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Check location relationship with atom-level trace"""
        # Extract object names from expression
        parts = expr.split()
        if len(parts) < 3:
            return False, f"Insufficient arguments for {relation_type}", 0.0, []
        
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
            return False, f"Object '{obj_name}' not found", 0.0, []
        if not location_node:
            return False, f"Location '{location_name}' not found", 0.0, []
        
        # Check edge with confidence from edge
        edge_key = (obj_node.name, location_node.name)
        reverse_key = (location_node.name, obj_node.name)
        
        traces = []
        confidence = 1.0
        
        if edge_key in scene_graph.edges:
            edge = scene_graph.edges[edge_key]
            edge_conf = getattr(edge, 'confidence', 1.0)
            if relation_type in edge.edge_type.lower():
                confidence = edge_conf
                if return_trace:
                    traces.append(AtomTrace(
                        atom_expr=expr,
                        value=True,
                        confidence=edge_conf,
                        source="edge_relation",
                        reason=f"Edge found: {edge.edge_type} with confidence {edge_conf:.2f}"
                    ))
                return True, f"{obj_node.name} is {relation_type} {location_node.name}", confidence, traces
        elif reverse_key in scene_graph.edges:
            edge = scene_graph.edges[reverse_key]
            edge_conf = getattr(edge, 'confidence', 1.0)
            if relation_type in edge.edge_type.lower():
                confidence = edge_conf
                if return_trace:
                    traces.append(AtomTrace(
                        atom_expr=expr,
                        value=True,
                        confidence=edge_conf,
                        source="edge_relation",
                        reason=f"Reverse edge found: {edge.edge_type} with confidence {edge_conf:.2f}"
                    ))
                return True, f"{obj_node.name} is {relation_type} {location_node.name}", confidence, traces
        
        # Not found - check geometry if available (for future enhancement)
        if return_trace:
            traces.append(AtomTrace(
                atom_expr=expr,
                value=False,
                confidence=0.0,
                source="edge_relation",
                reason=f"No edge found for {relation_type} relationship"
            ))
        
        return False, f"{obj_node.name} is not {relation_type} {location_node.name}", 0.0, traces
    
    def _check_state(self, expr: str, scene_graph: SceneGraph, state: str, 
                    return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Check object state with atom-level trace"""
        parts = expr.split()
        if len(parts) < 2:
            return False, f"Insufficient arguments for state check", 0.0, []
        
        obj_name = parts[1].lower()
        
        # Find node
        obj_node = None
        for node in scene_graph.nodes:
            if obj_name in node.name.lower() or node.name.lower() in obj_name:
                obj_node = node
                break
        
        if not obj_node:
            return False, f"Object '{obj_name}' not found", 0.0, []
        
        # Check state with confidence from node
        node_state = (obj_node.state or '').lower()
        node_conf = getattr(obj_node, 'confidence', 1.0)
        state_matches = state in node_state or node_state == state
        
        traces = []
        if return_trace:
            traces.append(AtomTrace(
                atom_expr=expr,
                value=state_matches,
                confidence=node_conf,
                source="state_check",
                reason=f"Node state: '{node_state}' vs required: '{state}'"
            ))
        
        if state_matches:
            return True, f"{obj_node.name} has state '{state}'", node_conf, traces
        else:
            return False, f"{obj_node.name} has state '{node_state}' but required '{state}'", node_conf, traces
    
    def _check_equality(self, expr: str, scene_graph: SceneGraph, 
                       return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Check equality with atom-level trace"""
        # Parse: eq obj.state 'value'
        parts = expr.split()
        if len(parts) < 3:
            return False, "Insufficient arguments for equality check", 0.0, []
        
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
            return False, f"Object '{obj_name}' not found", 0.0, []
        
        # Check attribute with confidence
        node_conf = getattr(obj_node, 'confidence', 1.0)
        traces = []
        
        if attr == 'state':
            node_value = (obj_node.state or '').lower()
            value_matches = node_value == value.lower()
            
            if return_trace:
                traces.append(AtomTrace(
                    atom_expr=expr,
                    value=value_matches,
                    confidence=node_conf,
                    source="state_equality",
                    reason=f"{obj_node.name}.{attr} == '{node_value}' vs required '{value}'"
                ))
            
            if value_matches:
                return True, f"{obj_node.name}.{attr} == '{value}'", node_conf, traces
            else:
                return False, f"{obj_node.name}.{attr} == '{node_value}' but required '{value}'", node_conf, traces
        
        return False, f"Unknown attribute '{attr}'", 0.0, []
    
    def _check_empty(self, expr: str, scene_graph: SceneGraph, 
                    return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Check if container is empty with atom-level trace"""
        parts = expr.split()
        if len(parts) < 2:
            return False, "Insufficient arguments for empty check", 0.0, []
        
        container_name = parts[1].lower()
        
        # Find container node
        container_node = None
        for node in scene_graph.nodes:
            node_name_lower = node.name.lower()
            if container_name in node_name_lower or node_name_lower in container_name:
                container_node = node
                break
        
        if not container_node:
            return False, f"Container '{container_name}' not found", 0.0, []
        
        # Check if any object is inside this container
        items_inside = []
        min_edge_conf = 1.0
        for (start_name, end_name), edge in scene_graph.edges.items():
            if edge.end.name == container_node.name and edge.edge_type in ['inside', 'in']:
                items_inside.append(edge.start.name)
                edge_conf = getattr(edge, 'confidence', 1.0)
                min_edge_conf = min(min_edge_conf, edge_conf)
        
        container_conf = getattr(container_node, 'confidence', 1.0)
        confidence = min(container_conf, min_edge_conf) if items_inside else container_conf
        is_empty = len(items_inside) == 0
        
        traces = []
        if return_trace:
            traces.append(AtomTrace(
                atom_expr=expr,
                value=is_empty,
                confidence=confidence,
                source="container_check",
                reason=f"Container '{container_node.name}' has {len(items_inside)} items inside: {items_inside}"
            ))
        
        if items_inside:
            return False, f"Container '{container_node.name}' is not empty: {', '.join(items_inside)} inside", confidence, traces
        else:
            return True, f"Container '{container_node.name}' is empty", confidence, traces
    
    def _check_keyword(self, expr: str, scene_graph: SceneGraph, 
                      return_trace: bool = True) -> Tuple[bool, str, float, List[AtomTrace]]:
        """Fallback: keyword-based checking with trace"""
        expr_lower = expr.lower()
        traces = []
        
        # Check for common keywords
        if 'empty' in expr_lower:
            for node in scene_graph.nodes:
                if node.name.lower() in expr_lower:
                    node_conf = getattr(node, 'confidence', 1.0)
                    is_empty = node.state and 'empty' in node.state.lower()
                    
                    if return_trace:
                        traces.append(AtomTrace(
                            atom_expr=expr,
                            value=not is_empty,
                            confidence=node_conf,
                            source="keyword_check",
                            reason=f"Keyword 'empty' check on {node.name}, state: {node.state}"
                        ))
                    
                    if is_empty:
                        return False, f"{node.name} is empty", node_conf, traces
                    return True, f"{node.name} is not empty", node_conf, traces
        
        return False, f"Cannot evaluate expression: {expr}", 0.0, []
    
    def validate_constraint(self, constraint: Dict, scene_graph: SceneGraph, 
                            evaluation_time: str = "now") -> Dict:
        """
        Validate a constraint with timing awareness (pre/post/invariant/goal)
        
        Args:
            constraint: Constraint dictionary with 'id', 'type', 'condition_expr', 'eval_time', 'severity'
            scene_graph: Scene graph to evaluate against
            evaluation_time: "pre" (before action), "post" (after action), "now" (current), "final" (task completion)
            
        Returns:
            Dictionary with:
            - id: Constraint ID
            - status: "SATISFIED", "VIOLATED", "UNCERTAIN", or "SKIP"
            - confidence: Aggregated confidence (0.0-1.0)
            - reason: Explanation
            - atom_traces: List of AtomTrace objects
        """
        constraint_id = constraint.get('id', 'UNKNOWN')
        constraint_type = constraint.get('type', 'postcondition')
        condition_expr = constraint.get('condition_expr', '')
        eval_time = constraint.get('eval_time', 'now')
        severity = constraint.get('severity', 'hard')
        min_confidence = constraint.get('min_confidence', self.min_confidence_threshold)
        
        # Normalize constraint type
        if constraint_type == 'pre':
            constraint_type = 'precondition'
        elif constraint_type == 'post':
            constraint_type = 'postcondition'
        
        # Check if evaluation time matches constraint type
        if constraint_type == 'precondition' and evaluation_time not in ['pre', 'now']:
            return {
                'id': constraint_id,
                'status': 'SKIP',
                'confidence': 0.0,
                'reason': f"Precondition skipped (evaluation_time={evaluation_time}, expected 'pre' or 'now')",
                'atom_traces': []
            }
        
        if constraint_type == 'postcondition' and evaluation_time not in ['post', 'final']:
            if evaluation_time == 'now':
                # Postcondition checked at initialization - likely violated
                return {
                    'id': constraint_id,
                    'status': 'VIOLATED',
                    'confidence': 1.0,
                    'reason': f"Postcondition checked at initialization (action not yet performed)",
                    'atom_traces': []
                }
            return {
                'id': constraint_id,
                'status': 'SKIP',
                'confidence': 0.0,
                'reason': f"Postcondition skipped (evaluation_time={evaluation_time}, expected 'post' or 'final')",
                'atom_traces': []
            }
        
        if constraint_type == 'goal' and evaluation_time != 'final':
            return {
                'id': constraint_id,
                'status': 'SKIP',
                'confidence': 0.0,
                'reason': f"Goal constraint skipped (evaluation_time={evaluation_time}, expected 'final')",
                'atom_traces': []
            }
        
        # If no condition_expr, try to compile from description
        if not condition_expr:
            from .constraint_generator import ConstraintGenerator
            generator = ConstraintGenerator(None)  # We only need compile_constraint
            condition_expr = generator.compile_constraint(constraint)
            if not condition_expr:
                return {
                    'id': constraint_id,
                    'status': 'UNCERTAIN',
                    'confidence': 0.0,
                    'reason': "Cannot compile constraint to executable AST",
                    'atom_traces': []
                }
        
        # Evaluate condition expression
        is_satisfied, reason, confidence, atom_traces = self.evaluate(
            condition_expr, scene_graph, return_trace=True
        )
        
        # Determine status based on result and confidence
        if confidence < min_confidence:
            status = 'UNCERTAIN'
        elif is_satisfied:
            status = 'SATISFIED'
        else:
            status = 'VIOLATED'
        
        return {
            'id': constraint_id,
            'status': status,
            'confidence': confidence,
            'reason': reason,
            'atom_traces': atom_traces,
            'condition_expr': condition_expr,
            'constraint_type': constraint_type,
            'severity': severity
        }
    
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

