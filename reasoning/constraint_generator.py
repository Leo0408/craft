"""
Constraint Generator Module
Generates logical constraints from scene graphs and task requirements

Supports two modes:
1. LLM-based generation (original method)
2. Template-based generation (improve3.md scheme) - deterministic, no eval
"""

from typing import List, Dict, Optional, Tuple, Callable, Any
from ..core.scene_graph import SceneGraph
from .llm_prompter import LLMPrompter


# ============================================================================
# Template-based Constraint Generation (improve3.md scheme)
# ============================================================================

# ============================================================================
# Action Classification (分层设计)
# ============================================================================

# A类：纯导航/定位动作（可安全忽略，不生成约束）
NAVIGATION_ACTIONS = {
    "navigate_to_obj",
    "navigate",
    "look_at",
    "turn_to",
    "move_to"
}

# B类：状态切换动作（必须建模，生成状态变量约束）
STATE_CHANGE_ACTIONS = {
    "toggle_on",
    "toggle_off",
    "open",
    "close"
}

# C类：语义复合动作（可由 LLM 辅助生成保守约束）
SEMANTIC_COMPOSITE_ACTIONS = {
    "pour",
    "wash",
    "heat",
    "clean",
    "fill"
}

# Action Semantic Templates - 动作语义模板库
ACTION_TEMPLATES = {
    "pick_up": {
        "pre": [
            ("gripper_empty", ["robot"]),
            ("reachable", ["X"])
        ],
        "post": [
            ("holding", ["X"])
        ],
        "category": "manipulation"
    },
    "put_in": {
        "pre": [
            ("holding", ["X"]),
            ("container_open", ["Y"]),
            ("container_empty", ["Y"])
        ],
        "post": [
            ("inside", ["X", "Y"])
        ],
        "category": "manipulation"
    },
    "put_on": {
        "pre": [
            ("holding", ["X"])
        ],
        "post": [
            ("on_top_of", ["X", "Y"])
        ],
        "category": "manipulation"
    },
    "toggle_on": {
        "pre": [
            ("reachable", ["X"])
        ],
        "post": [
            ("toggled_on", ["X"])
        ],
        "category": "state_change"
    },
    "toggle_off": {
        "pre": [
            ("reachable", ["X"])
        ],
        "post": [
            ("toggled_off", ["X"])
        ],
        "category": "state_change"
    },
    "open": {
        "pre": [
            ("reachable", ["Y"])
        ],
        "post": [
            ("container_open", ["Y"])
        ],
        "category": "state_change"
    },
    "close": {
        "pre": [
            ("reachable", ["Y"])
        ],
        "post": [
            ("container_closed", ["Y"])
        ],
        "category": "state_change"
    },
    "navigate_to_obj": {
        "pre": [],
        "post": [],
        "category": "navigation"
    },
    "pour": {
        "pre": [
            ("holding", ["X"])
        ],
        "post": [
            ("filled", ["Y"])
        ],
        "category": "semantic_composite"
    }
}


# Predicate Implementation - Predicate → 可执行函数映射
def predicate_holding(sg: SceneGraph, X: str) -> bool:
    """Check if robot is holding object X"""
    # Check if there's a holding edge from Robot to X
    robot_node = sg.get_node("Robot")
    if not robot_node:
        return False
    # Check for holding edge or isPickedUp attribute
    for node in sg.nodes:
        if node.name == X and node.attributes.get('isPickedUp', False):
            return True
    # Check for holding edge
    edge_key = (robot_node.name, X)
    if edge_key in sg.edges:
        return sg.edges[edge_key].edge_type == "holding"
    return False


def predicate_inside(sg: SceneGraph, X: str, Y: str) -> bool:
    """Check if X is inside Y"""
    edge_key = (X, Y)
    if edge_key in sg.edges:
        return sg.edges[edge_key].edge_type == "inside"
    return False


def predicate_container_empty(sg: SceneGraph, Y: str) -> bool:
    """Check if container Y is empty (no objects inside)"""
    items_inside = []
    for (start_name, end_name), edge in sg.edges.items():
        if edge.end.name == Y and edge.edge_type == "inside":
            items_inside.append(edge.start.name)
    return len(items_inside) == 0


def predicate_container_open(sg: SceneGraph, Y: str) -> bool:
    """Check if container Y is open"""
    node = sg.get_node(Y)
    if not node:
        return False
    return node.attributes.get("isOpen", False)


def predicate_gripper_empty(sg: SceneGraph, _: str = None) -> bool:
    """Check if gripper is empty (not holding anything)"""
    # Check if any node has isPickedUp=True
    for node in sg.nodes:
        if node.attributes.get('isPickedUp', False):
            return False
    return True


def predicate_on_top_of(sg: SceneGraph, X: str, Y: str) -> bool:
    """Check if X is on top of Y"""
    edge_key = (X, Y)
    if edge_key in sg.edges:
        edge_type = sg.edges[edge_key].edge_type.lower()
        return edge_type in ["on_top_of", "on", "on top of", "ontopof"]
    return False


def predicate_reachable(sg: SceneGraph, X: str) -> bool:
    """Check if X is reachable (simplified: object exists)"""
    return sg.get_node(X) is not None


def predicate_toggled_on(sg: SceneGraph, Y: str) -> bool:
    """Check if Y is toggled on"""
    node = sg.get_node(Y)
    if not node:
        return False
    return node.attributes.get("isToggled", False)


def predicate_toggled_off(sg: SceneGraph, Y: str) -> bool:
    """Check if Y is toggled off"""
    node = sg.get_node(Y)
    if not node:
        return True  # If not found, assume off
    return not node.attributes.get("isToggled", False)


def predicate_filled(sg: SceneGraph, Y: str) -> bool:
    """Check if Y is filled"""
    node = sg.get_node(Y)
    if not node:
        return False
    return node.attributes.get("isFilled", False)


# Predicate implementation mapping
PREDICATE_IMPL = {
    "holding": predicate_holding,
    "inside": predicate_inside,
    "container_empty": predicate_container_empty,
    "container_open": predicate_container_open,
    "gripper_empty": predicate_gripper_empty,
    "on_top_of": predicate_on_top_of,
    "reachable": predicate_reachable,
    "toggled_on": predicate_toggled_on,
    "toggled_off": predicate_toggled_off,
    "filled": predicate_filled,
}


class ConstraintGenerator:
    """Generates logical constraints for task execution"""
    
    def __init__(self, llm_prompter: LLMPrompter):
        self.llm_prompter = llm_prompter
    
    def generate_constraints(self, scene_graph: SceneGraph, task_info: Dict, 
                            goal: Optional[str] = None) -> List[Dict]:
        """
        Generate constraints from scene graph and task information
        
        Args:
            scene_graph: Current scene graph
            task_info: Task information dictionary
            goal: Optional goal description
            
        Returns:
            List of constraint dictionaries with 'description' and 'condition'
        """
        scene_text = scene_graph.to_text()
        task_name = task_info.get('name', '')
        goal_text = goal or task_info.get('success_condition', '')
        actions_text = ", ".join(task_info.get('actions', []))
        
        # Get prompt info with error handling
        try:
            prompt_info = self.llm_prompter.prompts['constraint-generator']
        except KeyError:
            raise KeyError(f"Prompt 'constraint-generator' not found. Available prompts: {list(self.llm_prompter.prompts.keys())}")
        
        # Check if prompt_info is a dict
        if not isinstance(prompt_info, dict):
            raise TypeError(f"prompt_info is not a dict, got {type(prompt_info)}: {prompt_info}")
        
        # Check for required keys
        if 'template-user' not in prompt_info:
            raise KeyError(f"'template-user' not found in prompt_info. Available keys: {list(prompt_info.keys())}")
        if 'template-system' not in prompt_info:
            raise KeyError(f"'template-system' not found in prompt_info. Available keys: {list(prompt_info.keys())}")
        
        # Get template string
        template_user = prompt_info['template-user']
        if not isinstance(template_user, str):
            raise TypeError(f"template-user is not a string, got {type(template_user)}: {template_user}")
        
        # Format the template with error handling
        try:
            user_prompt = template_user.format(
            task=task_name,
            actions=actions_text,
            scene_graph=scene_text,
            goal=goal_text
        )
        except KeyError as e:
            # Provide more context about the error
            required_keys = {'task', 'actions', 'scene_graph', 'goal'}
            raise KeyError(
                f"Error formatting template-user: {e}. "
                f"Template should use keys from: {required_keys}. "
                f"Template preview (first 200 chars): {template_user[:200]}"
            ) from e
        
        response, _ = self.llm_prompter.query(
            prompt_info['template-system'],
            user_prompt,
            max_tokens=2000  # Increase for JSON output
        )
        
        # Parse constraints from response
        constraints = self._parse_constraints(response)
        
        return constraints
    
    def generate_constraints_for_action(self, action: str, action_index: int, 
                                       action_type: Optional[str] = None) -> List[Dict]:
        """
        Generate constraints for a SINGLE action (Action-centric method)
        
        This method implements the improved approach from improve4.md:
        - Generate constraints for one action at a time
        - Do NOT use final_scene_graph
        - Bind action_index at generation time
        - Use Action-local prompt (no full task context)
        
        Args:
            action: Action string, e.g., "(put_in, Mug, CoffeeMachine)"
            action_index: Index of the action in the sequence
            action_type: Optional action type (pick_up, put_in, etc.)
                        If None, will be inferred from action string
        
        Returns:
            List of constraint dictionaries with action_index already bound
        """
        import re
        
        # Parse action to get type and arguments
        if action_type is None:
            # Extract action type from action string
            match = re.match(r'\((\w+),', action)
            if match:
                action_type = match.group(1)
            else:
                # Fallback: try to infer from action string
                action_lower = action.lower()
                if 'pick_up' in action_lower or 'pickup' in action_lower:
                    action_type = 'pick_up'
                elif 'put_in' in action_lower:
                    action_type = 'put_in'
                elif 'put_on' in action_lower:
                    action_type = 'put_on'
                elif 'toggle_on' in action_lower:
                    action_type = 'toggle_on'
                elif 'toggle_off' in action_lower:
                    action_type = 'toggle_off'
                elif 'navigate' in action_lower:
                    action_type = 'navigate_to_obj'
                elif 'open' in action_lower and 'container' not in action_lower:
                    action_type = 'open'
                elif 'close' in action_lower and 'container' not in action_lower:
                    action_type = 'close'
                else:
                    action_type = 'unknown'
        
        # Classify action type
        action_category = self._classify_action(action_type)
        
        # A类：导航动作 - 不生成约束（可安全忽略）
        if action_category == "navigation":
            return []  # 返回空列表，不生成约束
        
        # Extract action arguments
        match = re.findall(r'\(([^)]+)\)', action)
        if match:
            action_args = [arg.strip() for arg in match[0].split(',')]
            # First element is action type, rest are objects
            if len(action_args) > 1:
                action_args = action_args[1:]  # Remove action type
            else:
                action_args = []
        else:
            action_args = []
        
        # Get template for this action type
        template = ACTION_TEMPLATES.get(action_type)
        
        # B类：状态切换动作 - 如果没有模板，生成基本状态变量约束
        if template is None and action_category == "state_change":
            return self._generate_state_variable_constraints(action, action_index, action_type, action_args)
        
        # C类：语义复合动作 - 使用受限的 LLM 辅助生成
        if template is None and action_category == "semantic_composite":
            return self._generate_constraints_for_action_llm_constrained(action, action_index, action_type, action_args)
        
        # 完全未知的动作类型
        if template is None:
            # 对于完全未知的动作，返回空列表（不生成约束）
            return []
        
        # Generate constraints from template
        constraints = []
        constraint_id_base = f"C{action_index * 10}"  # Reserve IDs per action
        
        # Generate Preconditions
        for idx, (predicate, template_args) in enumerate(template.get("pre", [])):
            # Bind template arguments to actual action arguments
            bound_args = self._bind_template_args(template_args, action_args)
            if not bound_args:
                continue
            
            # Build constraint
            constraint = {
                'id': f"{constraint_id_base}_PRE{idx+1}",
                'type': 'precondition',
                'template': f"{predicate}({', '.join(bound_args)})",
                'description': self._generate_description(predicate, bound_args, 'precondition'),
                'condition_expr': f"{predicate}({', '.join(bound_args)})",
                'action': action,
                'action_index': action_index,
                'bound_action': action,
                'severity': 'hard'
            }
            constraints.append(constraint)
        
        # Generate Postconditions
        for idx, (predicate, template_args) in enumerate(template.get("post", [])):
            # Bind template arguments to actual action arguments
            bound_args = self._bind_template_args(template_args, action_args)
            if not bound_args:
                continue
            
            # 特殊处理：put_on 动作根据目标对象类型判断应该生成 inside 还是 on_top_of 约束
            # Sink 和 SinkBasin 视为等价（都是容器类型）
            if action_type == "put_on" and predicate == "on_top_of" and len(bound_args) >= 2:
                target_obj = bound_args[1]
                target_obj_lower = target_obj.lower()
                
                # 定义容器类型（包括 Sink/SinkBasin 的等价处理）
                CONTAINER_TYPES = {
                    'sink', 'sinkbasin',  # Sink 和 SinkBasin 视为等价
                    'bowl', 'pot', 'pan', 'mug', 'cup', 'coffeemachine',
                    'fridge', 'cabinet', 'drawer', 'microwave', 'oven'
                }
                
                # 判断目标对象是否为容器类型
                is_container = any(container_type in target_obj_lower for container_type in CONTAINER_TYPES)
                
                if is_container:
                    # 对于容器类型，生成 inside 约束而不是 on_top_of
                    predicate = "inside"
                    bound_args = bound_args  # 保持参数不变
            
            # Build constraint
            constraint = {
                'id': f"{constraint_id_base}_POST{idx+1}",
                'type': 'postcondition',
                'template': f"{predicate}({', '.join(bound_args)})",
                'description': self._generate_description(predicate, bound_args, 'postcondition'),
                'condition_expr': f"{predicate}({', '.join(bound_args)})",
                'action': action,
                'action_index': action_index,
                'bound_action': action,
                'severity': 'hard'
            }
            constraints.append(constraint)
        
        return constraints
    
    def _classify_action(self, action_type: str) -> str:
        """
        分类动作类型
        
        Returns:
            "navigation" | "state_change" | "semantic_composite" | "manipulation" | "unknown"
        """
        if action_type in NAVIGATION_ACTIONS:
            return "navigation"
        elif action_type in STATE_CHANGE_ACTIONS:
            return "state_change"
        elif action_type in SEMANTIC_COMPOSITE_ACTIONS:
            return "semantic_composite"
        elif action_type in ACTION_TEMPLATES:
            template = ACTION_TEMPLATES.get(action_type, {})
            return template.get("category", "manipulation")
        else:
            return "unknown"
    
    def _generate_state_variable_constraints(self, action: str, action_index: int,
                                             action_type: str, action_args: List[str]) -> List[Dict]:
        """
        为状态切换动作生成基本状态变量约束（B类动作）
        
        例如：toggle_on(Faucet) → POST: Faucet.state == ON
        """
        if not action_args:
            return []
        
        obj_name = action_args[0]
        constraints = []
        constraint_id_base = f"C{action_index * 10}"
        
        # 根据动作类型生成对应的状态变量约束
        if action_type == "toggle_on":
            constraint = {
                'id': f"{constraint_id_base}_POST1",
                'type': 'postcondition',
                'template': f"toggled_on({obj_name})",
                'description': f"{obj_name} must be toggled on",
                'condition_expr': f"node.attributes.get('isToggled', False)",
                'action': action,
                'action_index': action_index,
                'bound_action': action,
                'severity': 'hard'
            }
            constraints.append(constraint)
        elif action_type == "toggle_off":
            constraint = {
                'id': f"{constraint_id_base}_POST1",
                'type': 'postcondition',
                'template': f"toggled_off({obj_name})",
                'description': f"{obj_name} must be toggled off",
                'condition_expr': f"not node.attributes.get('isToggled', False)",
                'action': action,
                'action_index': action_index,
                'bound_action': action,
                'severity': 'hard'
            }
            constraints.append(constraint)
        elif action_type == "open":
            constraint = {
                'id': f"{constraint_id_base}_POST1",
                'type': 'postcondition',
                'template': f"container_open({obj_name})",
                'description': f"{obj_name} must be open",
                'condition_expr': f"node.attributes.get('isOpen', False)",
                'action': action,
                'action_index': action_index,
                'bound_action': action,
                'severity': 'hard'
            }
            constraints.append(constraint)
        elif action_type == "close":
            constraint = {
                'id': f"{constraint_id_base}_POST1",
                'type': 'postcondition',
                'template': f"container_closed({obj_name})",
                'description': f"{obj_name} must be closed",
                'condition_expr': f"not node.attributes.get('isOpen', True)",
                'action': action,
                'action_index': action_index,
                'bound_action': action,
                'severity': 'hard'
            }
            constraints.append(constraint)
        
        return constraints
    
    def _generate_constraints_for_action_llm_constrained(self, action: str, action_index: int,
                                                         action_type: str, action_args: List[str]) -> List[Dict]:
        """
        受限的 LLM 辅助生成（C类动作）
        
        LLM 只从预定义的约束类型中选择，不能自由创造规则
        """
        # 预定义的约束类型选项
        constraint_schema = [
            "1. State change (on/off, clean/dirty, open/closed)",
            "2. Containment relation (inside, on_top_of)",
            "3. Spatial relation (near, in_contact)",
            "4. No constraint needed"
        ]
        
        user_prompt = f"""Given an action: {action}

Select applicable constraints from the following schema (output only the numbers):
{chr(10).join(constraint_schema)}

Rules:
- You can only select from the listed constraint types
- Do NOT create new constraint types
- Output format: "Selected: 1, 2" or "Selected: 4" (if no constraint needed)"""
        
        system_prompt = """You are a constraint type selector. Your task is to select constraint types from a predefined schema for a given action. You cannot create new constraint types."""
        
        try:
            response, _ = self.llm_prompter.query(
                system_prompt,
                user_prompt,
                max_tokens=100
            )
            
            # 解析 LLM 的选择
            selected_types = []
            if "Selected:" in response:
                parts = response.split("Selected:")[1].strip()
                selected_types = [int(x.strip()) for x in parts.split(",") if x.strip().isdigit()]
            
            # 如果选择了 4（不需要约束），返回空列表
            if 4 in selected_types:
                return []
            
            # 根据选择的类型生成约束
            constraints = []
            constraint_id_base = f"C{action_index * 10}"
            
            if 1 in selected_types:  # State change
                # 为状态变化动作生成基本状态变量约束
                if action_args:
                    obj_name = action_args[0]
                    constraint = {
                        'id': f"{constraint_id_base}_POST1",
                        'type': 'postcondition',
                        'template': f"state_changed({obj_name})",
                        'description': f"{obj_name} state changed",
                        'condition_expr': f"node.attributes.get('state') is not None",
                        'action': action,
                        'action_index': action_index,
                        'bound_action': action,
                        'severity': 'hard'
                    }
                    constraints.append(constraint)
            
            if 2 in selected_types:  # Containment relation
                if len(action_args) >= 2:
                    constraint = {
                        'id': f"{constraint_id_base}_POST2",
                        'type': 'postcondition',
                        'template': f"inside({action_args[0]}, {action_args[1]})",
                        'description': f"{action_args[0]} must be inside {action_args[1]}",
                        'condition_expr': f"has_edge('{action_args[0]}', '{action_args[1]}', 'inside')",
                        'action': action,
                        'action_index': action_index,
                        'bound_action': action,
                        'severity': 'hard'
                    }
                    constraints.append(constraint)
            
            return constraints
            
        except Exception as e:
            # 如果 LLM 调用失败，返回空列表（保守策略）
            return []
    
    def _bind_template_args(self, template_args: List[str], action_args: List[str]) -> List[str]:
        """
        Bind template arguments (X, Y, robot) to actual action arguments
        
        Binding rules:
        - "robot" -> "Robot"
        - "X" -> action_args[0] (first argument)
        - "Y" -> action_args[1] if exists, else action_args[0] (fallback to first)
        - Direct match if template_arg is in action_args
        """
        bound = []
        for template_arg in template_args:
            if template_arg == "robot":
                bound.append("Robot")
            elif template_arg == "X" and len(action_args) > 0:
                bound.append(action_args[0])
            elif template_arg == "Y":
                # Y can be the second argument, or fallback to first if only one argument
                if len(action_args) > 1:
                    bound.append(action_args[1])
                elif len(action_args) > 0:
                    bound.append(action_args[0])  # Fallback: use first argument for single-arg actions
            elif template_arg in action_args:
                bound.append(template_arg)
        return bound
    
    def _generate_description(self, predicate: str, args: List[str], constraint_type: str) -> str:
        """Generate human-readable description for a constraint"""
        if predicate == "holding":
            return f"Robot must be holding the {args[0]}"
        elif predicate == "gripper_empty":
            return "Robot gripper must be empty"
        elif predicate == "reachable":
            return f"{args[0]} must be reachable"
        elif predicate == "container_open":
            return f"{args[0]} must be open"
        elif predicate == "container_empty":
            return f"{args[0]} must be empty"
        elif predicate == "inside":
            return f"{args[0]} must be inside {args[1]}"
        elif predicate == "on_top_of":
            return f"{args[0]} must be on top of {args[1]}"
        elif predicate == "toggled_on":
            return f"{args[0]} must be toggled on"
        elif predicate == "toggled_off":
            return f"{args[0]} must be toggled off"
        elif predicate == "filled":
            return f"{args[0]} must be filled"
        else:
            return f"{predicate}({', '.join(args)})"
    
    def _generate_constraints_for_action_llm(self, action: str, action_index: int,
                                            action_type: str, action_args: List[str]) -> List[Dict]:
        """
        Fallback: Use LLM to generate constraints for a single action
        (Action-local prompt, no scene graph)
        """
        # Get action template if available
        template = ACTION_TEMPLATES.get(action_type, {})
        pre_templates = template.get("pre", [])
        post_templates = template.get("post", [])
        
        # Build action-local prompt
        pre_templates_str = ", ".join([f"{pred}({', '.join(args)})" for pred, args in pre_templates])
        post_templates_str = ", ".join([f"{pred}({', '.join(args)})" for pred, args in post_templates])
        
        user_prompt = f"""Current Action: {action}
Action Index: {action_index}
Action Type: {action_type}
Action Arguments: {', '.join(action_args)}

Action Semantic Template:
    Preconditions: {pre_templates_str or 'None'}
    Postconditions: {post_templates_str or 'None'}

Rules:
- Generate constraints ONLY for this action
- Do NOT reference future actions
- Do NOT reference final goal state
- Instantiate templates with actual object names from action arguments

Output JSON format:
{{
  "constraints": [
    {{
      "id": "C1",
      "type": "precondition",
      "template": "holding(Mug)",
      "description": "Robot must be holding the Mug",
      "action_index": {action_index},
      "action": "{action}"
    }}
  ]
}}"""
        
        system_prompt = """You are a robot task analyzer. Generate constraints for a SINGLE action by 
instantiating the provided action semantic template. Do NOT reference other actions or the final goal."""
        
        try:
            response, _ = self.llm_prompter.query(
                system_prompt,
                user_prompt,
                max_tokens=1000
            )
            constraints = self._parse_constraints(response)
            # Ensure action_index is set
            for constraint in constraints:
                constraint['action_index'] = action_index
                constraint['bound_action'] = action
            return constraints
        except Exception as e:
            print(f"⚠️  LLM generation failed for action {action}: {e}")
            return []
    
    def _parse_constraints(self, llm_response: str) -> List[Dict]:
        """
        Parse LLM response into structured constraint list
        
        Args:
            llm_response: LLM response text (should be JSON format)
            
        Returns:
            List of constraint dictionaries with structured fields
        """
        constraints = []
        
        # Try to parse as JSON first
        try:
            import json
            # Extract JSON from response (handle markdown code blocks)
            response_text = llm_response.strip()
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0].strip()
            
            # Try to parse JSON
            data = json.loads(response_text)
            if 'constraints' in data:
                for constraint in data['constraints']:
                    # Normalize constraint type
                    raw_type = str(constraint.get('type', 'precondition')).lower()
                    if any(kw in raw_type for kw in ['pre', 'before']):
                        constraint_type = 'precondition'
                    elif any(kw in raw_type for kw in ['post', 'after']):
                        constraint_type = 'postcondition'
                    else:
                        constraint_type = 'precondition'
                    
                    # Ensure template is captured - check multiple potential keys
                    template_val = (
                        constraint.get('template') or 
                        constraint.get('template_id') or
                        constraint.get('condition_expr') or 
                        constraint.get('condition') or
                        'N/A'
                    )
                    
                    constraints.append({
                        'id': str(constraint.get('id', f'C{len(constraints)+1}')),
                        'type': constraint_type,
                        'template': str(template_val),
                        'action': str(constraint.get('action', '')),
                        'description': str(constraint.get('description', '')),
                        'condition_expr': str(constraint.get('condition_expr') or template_val),
                        'severity': str(constraint.get('severity', 'hard')),
                        'action_index': constraint.get('action_index'), # Preserve action_index if LLM provides it
                        'condition': str(constraint.get('condition_expr') or template_val)
                    })
                return constraints
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Fallback to old parsing method if JSON parsing fails
            print(f"⚠️  Failed to parse JSON constraints, falling back to text parsing: {e}")
        
        # Fallback: Parse as text format (backward compatibility)
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove numbering
            if line and line[0].isdigit():
                line = line.split('.', 1)[-1].strip()
            
            # Parse constraint format: description (condition)
            description = line
            condition = None
            constraint_type = None
            
            if '(' in line and ')' in line:
                description = line.split('(')[0].strip()
                condition = line.split('(')[1].split(')')[0].strip()
            
            # Infer constraint type from description
            description_lower = description.lower()
            
            # Precondition: "before", "must be opened", "must be empty"
            if any(keyword in description_lower for keyword in ['before', 'must be opened', 'must be empty', 'must be closed']):
                constraint_type = 'precondition'
            # Postcondition: "after", "must be moved from X to Y" (completed action)
            elif any(keyword in description_lower for keyword in ['after', 'must be moved from', 'must be placed']):
                constraint_type = 'postcondition'
            # Goal: "to complete task", "final", "success condition"
            elif any(keyword in description_lower for keyword in ['to complete', 'final', 'success', 'goal']):
                constraint_type = 'goal'
            # Invariant: "must not", "must always"
            elif 'must not' in description_lower or 'must always' in description_lower:
                constraint_type = 'invariant'
            else:
                # Default: try to infer from context
                if 'must be' in description_lower and not 'moved' in description_lower:
                    constraint_type = 'precondition'
                else:
                    constraint_type = 'postcondition'
            
            constraints.append({
                'id': f'C{len(constraints)+1}',
                'description': description,
                'template': 'N/A (Text Parsed)', # Add template key to fallback
                'condition': condition,
                'condition_expr': condition or '',  # Use condition as condition_expr
                'type': constraint_type,
                'severity': 'hard',
                'eval_time': 'pre' if constraint_type == 'precondition' else 'post' if constraint_type == 'postcondition' else 'now',
                'raw': line
            })
        
        return constraints
    
    def validate_constraint(self, constraint: Dict, scene_graph: SceneGraph, 
                          evaluation_time: str = "now") -> Tuple[bool, str]:
        """
        Validate if a constraint is satisfied in the current scene
        
        Args:
            constraint: Constraint dictionary with 'type', 'description', 'condition'
            scene_graph: Current scene graph
            evaluation_time: "now" (current state), "pre" (before action), 
                           "post" (after action), "final" (task completion)
            
        Returns:
            (is_satisfied, reason) tuple
        """
        description = constraint.get('description', '').lower()
        condition = constraint.get('condition', '')
        constraint_type = constraint.get('type', 'postcondition')  # Default type
        
        if not description:
            return True, "No description to check"
        
        # Check if evaluation time matches constraint type
        if constraint_type == 'precondition' and evaluation_time not in ['now', 'pre']:
            # Precondition should be checked before action
            pass  # Allow checking at 'now' for initial state
        elif constraint_type == 'postcondition' and evaluation_time not in ['post', 'final']:
            # Postcondition should be checked after action
            # If checking at 'now', it's likely VIOLATED (action hasn't happened yet)
            if evaluation_time == 'now':
                return False, f"Postcondition checked at initialization (action not yet performed)"
        elif constraint_type == 'goal' and evaluation_time != 'final':
            # Goal should only be checked at task completion
            if evaluation_time != 'final':
                return False, f"Goal constraint checked before task completion"
        
        # Extract key information from constraint description
        # Check for common constraint patterns
        
        # Pattern 1: "must be moved from X to Y" 
        # For postcondition: check if object is NOT at source and IS at destination
        # For precondition: this doesn't make sense, likely a parsing error
        if 'must be moved from' in description and 'to' in description:
            if constraint_type == 'postcondition':
                return self._check_movement_constraint(description, scene_graph)
            else:
                # Precondition with movement doesn't make sense
                return False, "Movement constraint should be postcondition, not precondition"
        
        # Pattern 2: "must be" + location - check if object is at that location
        if 'must be' in description and ('on' in description or 'inside' in description or 'in' in description):
            return self._check_location_constraint(description, scene_graph)
        
        # Pattern 3: "must be" + state - check if object has that state
        if 'must be' in description and ('open' in description or 'closed' in description or 
                                        'empty' in description or 'filled' in description):
            return self._check_state_constraint(description, scene_graph)
        
        # Pattern 4: "must not" - check if condition is NOT true
        if 'must not' in description:
            return self._check_negative_constraint(description, scene_graph)
        
        # Pattern 5: Container must be empty (precondition for put_in)
        if 'empty' in description and ('container' in description or 'machine' in description or 
                                      'coffee machine' in description):
            return self._check_container_empty(description, scene_graph)
        
        # Default: if we can't parse, return False (assume violated for safety)
        # Changed from True to False - better to catch violations than miss them
        return False, "Cannot parse constraint description"
    
    def _check_movement_constraint(self, description: str, scene_graph: SceneGraph) -> Tuple[bool, str]:
        """Check if movement constraint is satisfied"""
        # Example: "blue cup must be moved from inside coffee machine to table"
        # This means: blue cup should NOT be in coffee machine, and SHOULD be on table
        
        # Extract object name
        words = description.split()
        obj_name = None
        source_location = None
        dest_location = None
        
        # Find "from" and "to" keywords
        if 'from' in words and 'to' in words:
            from_idx = words.index('from')
            to_idx = words.index('to')
            
            # Object is before "must be moved"
            must_idx = words.index('must') if 'must' in words else -1
            if must_idx > 0:
                obj_name = ' '.join(words[:must_idx])
            
            # Source location is between "from" and "to"
            if from_idx < to_idx:
                source_location = ' '.join(words[from_idx+1:to_idx])
                dest_location = ' '.join(words[to_idx+1:])
        
        if not obj_name or not source_location or not dest_location:
            return False, "Cannot parse movement constraint"
        
        # Check if object is at source location (should NOT be)
        is_at_source = self._check_object_location(obj_name, source_location, scene_graph)
        
        # Check if object is at destination (should be)
        is_at_dest = self._check_object_location(obj_name, dest_location, scene_graph)
        
        # Constraint is satisfied if object is NOT at source AND is at destination
        if not is_at_source and is_at_dest:
            return True, f"{obj_name} successfully moved from {source_location} to {dest_location}"
        elif is_at_source:
            return False, f"{obj_name} is still at source location ({source_location})"
        elif not is_at_dest:
            return False, f"{obj_name} is not at destination ({dest_location})"
        else:
            return False, f"{obj_name} location unknown"
    
    def _check_location_constraint(self, description: str, scene_graph: SceneGraph) -> Tuple[bool, str]:
        """Check if location constraint is satisfied"""
        # Example: "purple cup must be on table"
        words = description.split()
        if 'must be' not in words:
            return False, "Cannot parse location constraint"
        
        must_idx = words.index('must')
        obj_name = ' '.join(words[:must_idx])
        
        # Find location after "must be"
        be_idx = words.index('be', must_idx) if 'be' in words[must_idx:] else must_idx + 1
        location = ' '.join(words[be_idx+1:])
        
        if not obj_name or not location:
            return False, "Cannot extract object name or location"
        
        is_at_location = self._check_object_location(obj_name, location, scene_graph)
        if is_at_location:
            return True, f"{obj_name} is at {location}"
        else:
            return False, f"{obj_name} is not at {location}"
    
    def _check_state_constraint(self, description: str, scene_graph: SceneGraph) -> Tuple[bool, str]:
        """Check if state constraint is satisfied"""
        # Example: "coffee machine must be open"
        words = description.split()
        if 'must be' not in words:
            return False, "Cannot parse state constraint"
        
        must_idx = words.index('must')
        obj_name = ' '.join(words[:must_idx])
        
        # Find state after "must be"
        be_idx = words.index('be', must_idx) if 'be' in words[must_idx:] else must_idx + 1
        state = ' '.join(words[be_idx+1:])
        
        if not obj_name or not state:
            return False, "Cannot extract object name or state"
        
        node = scene_graph.get_node(obj_name)
        if not node:
            return False, f"Object '{obj_name}' not found in scene graph"
        
        # Check state
        node_state = (node.state or '').lower()
        state_lower = state.lower()
        
        # Check if state matches
        state_matches = state_lower in node_state or node_state in state_lower
        
        if state_matches:
            return True, f"{obj_name} has state '{node.state}' (required: '{state}')"
        else:
            return False, f"{obj_name} has state '{node.state}' but required state is '{state}'"
    
    def _check_negative_constraint(self, description: str, scene_graph: SceneGraph) -> Tuple[bool, str]:
        """Check if negative constraint is satisfied (must NOT be true)"""
        # Example: "purple cup must not be inside coffee machine"
        words = description.split()
        if 'must not' not in words:
            return False, "Cannot parse negative constraint"
        
        must_idx = words.index('must')
        obj_name = ' '.join(words[:must_idx])
        
        # Find location/state after "must not be"
        not_idx = words.index('not', must_idx)
        be_idx = words.index('be', not_idx) if 'be' in words[not_idx:] else not_idx + 1
        condition = ' '.join(words[be_idx+1:])
        
        if not obj_name or not condition:
            return False, "Cannot extract object name or condition"
        
        # Check if condition is true, constraint is violated if it is
        is_true = self._check_object_location(obj_name, condition, scene_graph)
        if not is_true:
            return True, f"{obj_name} is not {condition} (as required)"
        else:
            return False, f"{obj_name} is {condition} (violates constraint)"
    
    def _check_container_empty(self, description: str, scene_graph: SceneGraph) -> Tuple[bool, str]:
        """Check if container is empty (precondition for put_in operations)"""
        # Example: "coffee machine must be empty" or "container must be empty"
        words = description.split()
        
        # Find container name
        container_name = None
        for word in words:
            if 'machine' in word or 'container' in word:
                # Get full name (e.g., "coffee machine")
                idx = words.index(word)
                if idx > 0:
                    container_name = f"{words[idx-1]} {word}"
                else:
                    container_name = word
                break
        
        if not container_name:
            # Try to find it from context
            if 'coffee machine' in description:
                container_name = 'coffee machine'
            else:
                return True  # Can't identify container
        
        # Check if any object is inside this container
        container_node = scene_graph.get_node(container_name)
        if not container_node:
            return False, f"Container '{container_name}' not found in scene graph"
        
        # Check all edges to see if anything is inside the container
        items_inside = []
        for (start_name, end_name), edge in scene_graph.edges.items():
            if edge.end.name == container_name and edge.edge_type in ['inside', 'in']:
                items_inside.append(edge.start.name)
        
        if items_inside:
            return False, f"Container '{container_name}' is not empty: {', '.join(items_inside)} inside"
        else:
            return True, f"Container '{container_name}' is empty"
    
    def _check_object_location(self, obj_name: str, location: str, scene_graph: SceneGraph) -> bool:
        """Check if object is at specified location"""
        obj_node = scene_graph.get_node(obj_name)
        if not obj_node:
            return False
        
        location_lower = location.lower()
        location_node = None
        
        # Try to find location node - match by name
        for node in scene_graph.nodes:
            node_name_lower = node.name.lower()
            # Check if location description contains node name or vice versa
            if (node_name_lower in location_lower or 
                location_lower in node_name_lower or
                any(word in node_name_lower for word in location_lower.split() if len(word) > 2)):
                location_node = node
                break
        
        if not location_node:
            return False
        
        # Check if there's an edge connecting object to location
        # Check both directions
        edge_key1 = (obj_node.name, location_node.name)
        edge_key2 = (location_node.name, obj_node.name)
        
        # Determine expected relationship type from location description
        expected_relations = []
        if 'inside' in location_lower or 'in' in location_lower:
            expected_relations = ['inside', 'in']
        elif 'on' in location_lower or 'on_top_of' in location_lower or 'top' in location_lower:
            expected_relations = ['on_top_of', 'on', 'on top of']
        elif 'near' in location_lower:
            expected_relations = ['near']
        elif 'contact' in location_lower:
            expected_relations = ['in_contact', 'contact']
        else:
            # Default: check all relationship types
            expected_relations = ['on', 'inside', 'in', 'on_top_of', 'near', 'in_contact']
        
        # Check edges
        for edge_key in [edge_key1, edge_key2]:
            if edge_key in scene_graph.edges:
                edge = scene_graph.edges[edge_key]
                edge_type_lower = edge.edge_type.lower()
                
                # Check if edge type matches expected relationship
                if any(rel in edge_type_lower for rel in expected_relations):
                    return True
        
        return False
    
    def compile_constraint(self, constraint: Dict) -> Optional[str]:
        """
        Compile constraint description to executable condition expression (AST/DSL)
        
        Args:
            constraint: Constraint dictionary with 'description', 'type', 'condition'
            
        Returns:
            Executable condition expression string, or None if cannot compile
            Examples:
                "(empty coffee_machine)"
                "(inside mug coffee_machine)"
                "(eq machine.door 'open')"
        """
        description = constraint.get('description', '').lower()
        constraint_type = constraint.get('type', 'precondition')
        
        # If constraint already has a condition expression, use it
        if constraint.get('condition'):
            return constraint.get('condition')
        
        # Pattern 1: State constraints - "must be empty/open/closed/filled"
        if 'must be' in description:
            words = description.split()
            try:
                must_idx = words.index('must')
                be_idx = words.index('be', must_idx) if 'be' in words[must_idx:] else must_idx + 1
                
                # Extract object name (before "must")
                obj_name = ' '.join(words[:must_idx]).strip()
                if not obj_name:
                    # Try alternative: "The X must be..."
                    if words[0].lower() == 'the':
                        obj_name = ' '.join(words[1:must_idx]).strip()
                
                # Extract state/location (after "be")
                condition_part = ' '.join(words[be_idx+1:]).strip()
                
                # State constraints
                if any(state in condition_part for state in ['empty', 'open', 'closed', 'filled', 'clean']):
                    # Format: (eq obj.state 'state')
                    state = condition_part.split()[0] if condition_part.split() else condition_part
                    obj_var = obj_name.replace(' ', '_').lower()
                    # Use eq format for state checks
                    return f"(eq {obj_var}.state '{state}')"
                
                # Location constraints - "must be on/inside/in"
                if any(loc in condition_part for loc in ['on', 'inside', 'in', 'on top of']):
                    # Extract location
                    if 'on top of' in condition_part:
                        location = condition_part.replace('on top of', '').strip()
                        obj_var = obj_name.replace(' ', '_').lower()
                        loc_var = location.replace(' ', '_').lower()
                        return f"(on_top_of {obj_var} {loc_var})"
                    elif 'inside' in condition_part or 'in' in condition_part:
                        location = condition_part.replace('inside', '').replace('in', '').strip()
                        obj_var = obj_name.replace(' ', '_').lower()
                        loc_var = location.replace(' ', '_').lower()
                        return f"(inside {obj_var} {loc_var})"
                    elif 'on' in condition_part:
                        location = condition_part.replace('on', '').strip()
                        obj_var = obj_name.replace(' ', '_').lower()
                        loc_var = location.replace(' ', '_').lower()
                        return f"(on_top_of {obj_var} {loc_var})"
            except (ValueError, IndexError):
                pass
        
        # Pattern 2: Negative constraints - "must not be"
        if 'must not' in description:
            words = description.split()
            try:
                must_idx = words.index('must')
                not_idx = words.index('not', must_idx)
                be_idx = words.index('be', not_idx) if 'be' in words[not_idx:] else not_idx + 1
                
                obj_name = ' '.join(words[:must_idx]).strip()
                if not obj_name and words[0].lower() == 'the':
                    obj_name = ' '.join(words[1:must_idx]).strip()
                
                condition_part = ' '.join(words[be_idx+1:]).strip()
                
                if 'inside' in condition_part or 'in' in condition_part:
                    location = condition_part.replace('inside', '').replace('in', '').strip()
                    obj_var = obj_name.replace(' ', '_').lower()
                    loc_var = location.replace(' ', '_').lower()
                    return f"(not (inside {obj_var} {loc_var}))"
            except (ValueError, IndexError):
                pass
        
        # Pattern 3: Movement constraints - "must be moved from X to Y"
        if 'must be moved from' in description and 'to' in description:
            words = description.split()
            try:
                from_idx = words.index('from')
                to_idx = words.index('to', from_idx)
                
                obj_name = ' '.join(words[:words.index('must')]).strip()
                if not obj_name and words[0].lower() == 'the':
                    obj_name = ' '.join(words[1:words.index('must')]).strip()
                
                source = ' '.join(words[from_idx+1:to_idx]).strip()
                dest = ' '.join(words[to_idx+1:]).strip()
                
                obj_var = obj_name.replace(' ', '_').lower()
                source_var = source.replace(' ', '_').lower()
                dest_var = dest.replace(' ', '_').lower()
                
                # Postcondition: object should be at destination, not at source
                return f"(and (inside {obj_var} {dest_var}) (not (inside {obj_var} {source_var})))"
            except (ValueError, IndexError):
                pass
        
        # Pattern 4: Container empty - "container must be empty"
        if 'empty' in description and ('container' in description or 'machine' in description):
            words = description.split()
            try:
                empty_idx = words.index('empty')
                # Find container name before "must be empty"
                must_idx = words.index('must') if 'must' in words else 0
                container_name = ' '.join(words[:must_idx]).strip()
                if not container_name and words[0].lower() == 'the':
                    container_name = ' '.join(words[1:must_idx]).strip()
                
                if container_name:
                    container_var = container_name.replace(' ', '_').lower()
                    return f"(empty {container_var})"
            except (ValueError, IndexError):
                pass
        
        # Default: try to extract simple condition from description
        # Use the constraint's condition field if available
        if constraint.get('condition'):
            return constraint.get('condition')
        
        # If we can't compile, return None (will be skipped)
        return None
    
    # ========================================================================
    # Template-based Constraint Generation Methods (improve3.md scheme)
    # ========================================================================
    
    def parse_action_string(self, action_str: str) -> Tuple[str, List[str]]:
        """
        Parse action string to (action_name, args)
        Example: "(put_in, Mug, CoffeeMachine)" -> ("put_in", ["Mug", "CoffeeMachine"])
        """
        action_str = action_str.strip()
        if action_str.startswith("(") and action_str.endswith(")"):
            action_str = action_str[1:-1]
        parts = [p.strip() for p in action_str.split(",")]
        return parts[0], parts[1:] if len(parts) > 1 else []
    
    def parse_actions(self, action_strings: List[str]) -> List[Dict]:
        """
        Parse action strings to action dictionaries
        
        Args:
            action_strings: List of action strings like ["(pick_up, Mug)", ...]
        
        Returns:
            List of action dicts with 'step_idx', 'name', 'args'
        """
        actions = []
        for idx, action_str in enumerate(action_strings):
            name, args = self.parse_action_string(action_str)
            actions.append({
                'step_idx': idx,
                'name': name,
                'args': args,
                'original': action_str
            })
        return actions
    
    def bind_args(self, template_args: List[str], action_args: List[str]) -> List[str]:
        """
        Bind template arguments to actual action arguments
        Example: ["X", "Y"] + ["Mug", "CoffeeMachine"] -> ["Mug", "CoffeeMachine"]
        """
        mapping = {}
        for i, arg in enumerate(template_args):
            if arg == "robot":
                mapping[arg] = "robot"
            else:
                mapping[arg] = action_args[i] if i < len(action_args) else None
        return [mapping.get(arg) for arg in template_args if mapping.get(arg) is not None]
    
    def instantiate_action_constraints(self, action: Dict) -> List[Dict]:
        """
        Instantiate constraints for a single action using templates
        
        Args:
            action: Action dict with 'step_idx', 'name', 'args'
        
        Returns:
            List of constraint dictionaries
        """
        action_name = action['name']
        template = ACTION_TEMPLATES.get(action_name)
        if template is None:
            return []
        
        constraints = []
        action_idx = action['step_idx']
        action_args = action['args']
        
        # Generate Preconditions
        for pred, t_args in template.get("pre", []):
            args = self.bind_args(t_args, action_args)
            if None in args:
                continue  # Skip if binding failed
            
            # Build description
            if pred == "gripper_empty":
                description = f"Robot gripper must be empty before {action_name}"
            elif pred == "holding":
                description = f"Robot must be holding {args[0]} before {action_name}"
            elif pred == "reachable":
                description = f"{args[0]} must be reachable before {action_name}"
            elif pred == "container_open":
                description = f"{args[0]} must be open before {action_name}"
            elif pred == "container_empty":
                description = f"{args[0]} must be empty before {action_name}"
            else:
                description = f"{pred}{tuple(args)} must hold before {action_name}"
            
            constraints.append({
                'id': f'C{len(constraints)+1}',
                'type': 'precondition',
                'predicate': pred,
                'args': args,
                'step': action_idx,
                'action': action,
                'description': description,
                'template': f"{pred}({', '.join(args)})"
            })
        
        # Generate Postconditions
        for pred, t_args in template.get("post", []):
            args = self.bind_args(t_args, action_args)
            if None in args:
                continue
            
            # Build description
            if pred == "holding":
                description = f"Robot must be holding {args[0]} after {action_name}"
            elif pred == "inside":
                description = f"{args[0]} must be inside {args[1]} after {action_name}"
            elif pred == "on_top_of":
                description = f"{args[0]} must be on top of {args[1]} after {action_name}"
            elif pred == "toggled_on":
                description = f"{args[0]} must be toggled on after {action_name}"
            elif pred == "toggled_off":
                description = f"{args[0]} must be toggled off after {action_name}"
            elif pred == "filled":
                description = f"{args[0]} must be filled after {action_name}"
            else:
                description = f"{pred}{tuple(args)} must hold after {action_name}"
            
            constraints.append({
                'id': f'C{len(constraints)+1}',
                'type': 'postcondition',
                'predicate': pred,
                'args': args,
                'step': action_idx,
                'action': action,
                'description': description,
                'template': f"{pred}({', '.join(args)})"
            })
        
        return constraints
    
    def compile_constraint_template(self, constraint: Dict) -> Dict:
        """
        Compile template-based constraint to executable form
        
        Args:
            constraint: Constraint dict with 'predicate', 'args'
        
        Returns:
            Constraint dict with 'executable' function added
        """
        predicate = constraint.get('predicate')
        args = constraint.get('args', [])
        
        pred_fn = PREDICATE_IMPL.get(predicate)
        if pred_fn is None:
            raise ValueError(f"Unknown predicate: {predicate}")
        
        def executable(scene_graph: SceneGraph) -> bool:
            """Executable constraint function"""
            return pred_fn(scene_graph, *args)
        
        constraint['executable'] = executable
        constraint['condition_expr'] = f"{predicate}({', '.join(args)})"
        return constraint
    
    def generate_constraints_from_templates(
        self, 
        action_strings: List[str],
        compile_constraints: bool = True
    ) -> List[Dict]:
        """
        Generate constraints from action sequence using templates (improve3.md scheme)
        
        Args:
            action_strings: List of action strings like ["(pick_up, Mug)", ...]
            compile_constraints: Whether to compile constraints to executable form
        
        Returns:
            List of compiled constraint dictionaries
        """
        # Parse actions
        actions = self.parse_actions(action_strings)
        
        # Instantiate constraints for each action
        constraints = []
        for action in actions:
            action_constraints = self.instantiate_action_constraints(action)
            constraints.extend(action_constraints)
        
        # Compile constraints to executable form
        if compile_constraints:
            compiled = []
            for constraint in constraints:
                try:
                    compiled_constraint = self.compile_constraint_template(constraint)
                    compiled.append(compiled_constraint)
                except Exception as e:
                    print(f"⚠️  Failed to compile constraint {constraint.get('id')}: {e}")
            return compiled
        
        return constraints

