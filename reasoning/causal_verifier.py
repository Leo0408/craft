"""
Causal Logic Verifier Module
Verifies causal relationships and logical consistency in scene graphs
"""

from typing import Dict, Optional, Tuple
from ..core.scene_graph import SceneGraph
from ..core.task_executor import Action
from .llm_prompter import LLMPrompter


class CausalVerifier:
    """Verifies causal logic in robot actions and scene transitions"""
    
    def __init__(self, llm_prompter: LLMPrompter):
        self.llm_prompter = llm_prompter
    
    def verify_causal_logic(self, scene_graph: SceneGraph, action: Action,
                           expected_effect: str, actual_observation: str) -> Tuple[bool, str]:
        """
        Verify if the causal logic of an action is consistent
        
        Args:
            scene_graph: Scene graph before/after action
            action: Action that was performed
            expected_effect: Expected effect of the action
            actual_observation: Actual observation after action
            
        Returns:
            (is_consistent, explanation) tuple
        """
        scene_text = scene_graph.to_text()
        action_str = f"{action.action_type} {action.target or ''}"
        
        prompt_info = self.llm_prompter.prompts['causal-verifier']
        user_prompt = prompt_info['template-user'].format(
            scene_graph=scene_text,
            action=action_str,
            expected_effect=expected_effect,
            observation=actual_observation
        )
        
        response, _ = self.llm_prompter.query(
            prompt_info['template-system'],
            user_prompt,
            max_tokens=600
        )
        
        # Determine if causal logic is consistent
        is_consistent = self._check_consistency(response)
        
        return is_consistent, response
    
    def _check_consistency(self, explanation: str) -> bool:
        """
        Check if explanation indicates consistency
        
        Args:
            explanation: LLM explanation
            
        Returns:
            True if consistent, False otherwise
        """
        explanation_lower = explanation.lower()
        
        # Keywords indicating inconsistency
        inconsistent_keywords = [
            'inconsistent', 'contradict', 'violate', 'conflict',
            'does not match', 'unexpected', 'should not', 'cannot'
        ]
        
        # Keywords indicating consistency
        consistent_keywords = [
            'consistent', 'matches', 'correct', 'expected', 'valid',
            'logical', 'makes sense'
        ]
        
        # Check for inconsistency first
        for keyword in inconsistent_keywords:
            if keyword in explanation_lower:
                return False
        
        # Check for consistency
        for keyword in consistent_keywords:
            if keyword in explanation_lower:
                return True
        
        # Default: assume consistent if no clear indication
        return True
    
    def verify_action_effect(self, pre_scene: SceneGraph, post_scene: SceneGraph,
                           action: Action) -> Tuple[bool, str]:
        """
        Verify if an action's effect matches the scene transition
        
        Args:
            pre_scene: Scene graph before action
            post_scene: Scene graph after action
            action: Action that was performed
            
        Returns:
            (is_valid, explanation) tuple
        """
        pre_text = pre_scene.to_text()
        post_text = post_scene.to_text()
        action_str = f"{action.action_type} {action.target or ''}"
        
        # Infer expected effect from action type
        expected_effect = self._infer_expected_effect(action)
        
        return self.verify_causal_logic(
            scene_graph=post_scene,
            action=action,
            expected_effect=expected_effect,
            actual_observation=post_text
        )
    
    def _infer_expected_effect(self, action: Action) -> str:
        """
        Infer expected effect from action type
        
        Args:
            action: Action to analyze
            
        Returns:
            Expected effect description
        """
        action_type = action.action_type.lower()
        target = action.target or ''
        
        effect_map = {
            'pick_up': f"{target} should be held by the robot",
            'put_on': f"{target} should be placed on a surface",
            'put_in': f"{target} should be inside a container",
            'toggle_on': f"{target} should be turned on",
            'toggle_off': f"{target} should be turned off",
            'navigate_to': f"robot should be near {target}",
            'open': f"{target} should be open",
            'close': f"{target} should be closed"
        }
        
        for key, effect in effect_map.items():
            if key in action_type:
                return effect
        
        return f"Action {action_type} should affect {target}"

