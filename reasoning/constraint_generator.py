"""
Constraint Generator Module
Generates logical constraints from scene graphs and task requirements
"""

from typing import List, Dict, Optional, Tuple
from ..core.scene_graph import SceneGraph
from .llm_prompter import LLMPrompter


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
        
        prompt_info = self.llm_prompter.prompts['constraint-generator']
        user_prompt = prompt_info['template-user'].format(
            task=task_name,
            scene_graph=scene_text,
            goal=goal_text
        )
        
        response, _ = self.llm_prompter.query(
            prompt_info['template-system'],
            user_prompt,
            max_tokens=800
        )
        
        # Parse constraints from response
        constraints = self._parse_constraints(response)
        
        return constraints
    
    def _parse_constraints(self, llm_response: str) -> List[Dict]:
        """
        Parse LLM response into structured constraint list
        
        Args:
            llm_response: LLM response text
            
        Returns:
            List of constraint dictionaries with type annotation
        """
        constraints = []
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
                # If it's about current state check, it's likely precondition
                if 'must be' in description_lower and not 'moved' in description_lower:
                    constraint_type = 'precondition'
                else:
                    constraint_type = 'postcondition'  # Default for action-related constraints
            
            constraints.append({
                'description': description,
                'condition': condition,
                'type': constraint_type,
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

