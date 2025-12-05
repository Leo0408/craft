"""
Scene Consistency Verifier Module
Verifies consistency between scene graphs at different time points
"""

from typing import Dict, Optional, Tuple, List
from ..core.scene_graph import SceneGraph
from ..core.task_executor import Action
from .llm_prompter import LLMPrompter


class ConsistencyVerifier:
    """Verifies consistency between scene graphs across time"""
    
    def __init__(self, llm_prompter: LLMPrompter):
        self.llm_prompter = llm_prompter
    
    def verify_consistency(self, previous_scene: SceneGraph, current_scene: SceneGraph,
                         action: Optional[Action] = None) -> Tuple[bool, str, List[str]]:
        """
        Verify if scene transition is consistent
        
        Args:
            previous_scene: Scene graph at previous time point
            current_scene: Scene graph at current time point
            action: Optional action that was performed between scenes
            
        Returns:
            (is_consistent, explanation, inconsistencies) tuple
        """
        previous_text = previous_scene.to_text()
        current_text = current_scene.to_text()
        action_str = action.action_type + " " + (action.target or "") if action else "no action"
        
        prompt_info = self.llm_prompter.prompts['consistency-verifier']
        user_prompt = prompt_info['template-user'].format(
            previous_scene=previous_text,
            current_scene=current_text,
            action=action_str
        )
        
        response, _ = self.llm_prompter.query(
            prompt_info['template-system'],
            user_prompt,
            max_tokens=800
        )
        
        # Extract inconsistencies
        inconsistencies = self._extract_inconsistencies(response)
        
        # Determine if consistent
        is_consistent = len(inconsistencies) == 0
        
        return is_consistent, response, inconsistencies
    
    def _extract_inconsistencies(self, explanation: str) -> List[str]:
        """
        Extract list of inconsistencies from explanation
        
        Args:
            explanation: LLM explanation
            
        Returns:
            List of inconsistency descriptions
        """
        inconsistencies = []
        lines = explanation.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Look for inconsistency indicators
            inconsistency_keywords = [
                'inconsistent', 'contradict', 'violate', 'conflict',
                'does not match', 'unexpected', 'should not', 'cannot',
                'missing', 'disappeared', 'appeared unexpectedly'
            ]
            
            for keyword in inconsistency_keywords:
                if keyword in line.lower():
                    inconsistencies.append(line)
                    break
        
        return inconsistencies
    
    def verify_scene_sequence(self, scene_graphs: Dict[int, SceneGraph],
                            actions: List[Action]) -> Dict[int, Tuple[bool, str, List[str]]]:
        """
        Verify consistency across a sequence of scenes
        
        Args:
            scene_graphs: Dictionary mapping frame indices to scene graphs
            actions: List of actions performed
            
        Returns:
            Dictionary mapping frame indices to (is_consistent, explanation, inconsistencies)
        """
        results = {}
        sorted_frames = sorted(scene_graphs.keys())
        
        for i in range(1, len(sorted_frames)):
            prev_frame = sorted_frames[i-1]
            curr_frame = sorted_frames[i]
            
            # Find action between frames (simplified - assumes one action per frame)
            action = None
            if i-1 < len(actions):
                action = actions[i-1]
            
            prev_scene = scene_graphs[prev_frame]
            curr_scene = scene_graphs[curr_frame]
            
            is_consistent, explanation, inconsistencies = self.verify_consistency(
                prev_scene, curr_scene, action
            )
            
            results[curr_frame] = (is_consistent, explanation, inconsistencies)
        
        return results
    
    def compare_scene_graphs(self, scene1: SceneGraph, scene2: SceneGraph) -> Dict:
        """
        Compare two scene graphs and identify differences
        
        Args:
            scene1: First scene graph
            scene2: Second scene graph
            
        Returns:
            Dictionary with comparison results
        """
        # Find nodes that appeared/disappeared
        nodes1 = {node.name for node in scene1.nodes}
        nodes2 = {node.name for node in scene2.nodes}
        
        appeared = nodes2 - nodes1
        disappeared = nodes1 - nodes2
        common = nodes1 & nodes2
        
        # Find edges that changed
        edges1 = set(scene1.edges.keys())
        edges2 = set(scene2.edges.keys())
        
        new_edges = edges2 - edges1
        removed_edges = edges1 - edges2
        
        # Check state changes
        state_changes = []
        for node_name in common:
            node1 = scene1.get_node(node_name)
            node2 = scene2.get_node(node_name)
            
            if node1 and node2 and node1.state != node2.state:
                state_changes.append({
                    'node': node_name,
                    'old_state': node1.state,
                    'new_state': node2.state
                })
        
        return {
            'appeared_nodes': list(appeared),
            'disappeared_nodes': list(disappeared),
            'new_edges': list(new_edges),
            'removed_edges': list(removed_edges),
            'state_changes': state_changes,
            'common_nodes': list(common)
        }

