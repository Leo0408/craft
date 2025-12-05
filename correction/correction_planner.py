"""
Correction Planner Module
Generates executable plans to recover from failures
"""

from typing import List, Dict, Optional
from ..core.task_executor import TaskExecutor, Action
from ..reasoning.llm_prompter import LLMPrompter
from ..core.scene_graph import SceneGraph


class CorrectionPlanner:
    """Generates correction plans based on failure analysis"""
    
    def __init__(self, llm_prompter: LLMPrompter):
        self.llm_prompter = llm_prompter
    
    def generate_correction_plan(self, task_info: Dict, original_plan: List[Action],
                                failure_explanation: str, final_state: SceneGraph,
                                expected_goal: str) -> List[Dict]:
        """
        Generate a correction plan to recover from failure
        
        Args:
            task_info: Original task information
            original_plan: List of original actions
            failure_explanation: Explanation of why the task failed
            final_state: Current scene graph state
            expected_goal: Expected goal state
            
        Returns:
            List of correction actions
        """
        # Build prompt for correction planning
        task_name = task_info.get('name', '')
        original_plan_str = "\n".join([
            f"{i+1}. {a.action_type} {a.target or ''}" 
            for i, a in enumerate(original_plan)
        ])
        
        final_state_str = final_state.to_text() if final_state else "Unknown"
        
        prompt = f"""Task: {task_name}
Original Plan:
{original_plan_str}

Failure Explanation: {failure_explanation}

Current State: {final_state_str}
Expected Goal: {expected_goal}

Generate a correction plan to complete the task. List actions in order.
Format: action_type(target) or action_type(source, target)
"""
        
        # Query LLM for correction plan
        system_prompt = "You are a robot task planner. Generate executable actions to correct failures and complete tasks."
        response, _ = self.llm_prompter.query(system_prompt, prompt, max_tokens=500)
        
        # Parse response into action list
        correction_actions = self._parse_correction_plan(response)
        
        return correction_actions
    
    def _parse_correction_plan(self, llm_response: str) -> List[Dict]:
        """
        Parse LLM response into structured action list
        
        Args:
            llm_response: LLM response text
            
        Returns:
            List of action dictionaries
        """
        actions = []
        lines = llm_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove numbering (e.g., "1. ", "2. ")
            if line and line[0].isdigit():
                line = line.split('.', 1)[-1].strip()
            
            # Parse action format: action_type(target) or action_type(source, target)
            if '(' in line and ')' in line:
                action_type = line.split('(')[0].strip()
                params = line.split('(')[1].split(')')[0].strip()
                
                if ',' in params:
                    source, target = [p.strip() for p in params.split(',')]
                    actions.append({
                        'type': action_type,
                        'source': source,
                        'target': target
                    })
                else:
                    actions.append({
                        'type': action_type,
                        'target': params
                    })
        
        return actions




