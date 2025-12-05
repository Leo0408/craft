"""
Failure Analyzer Module
Analyzes robot failures using hierarchical reasoning
"""

from typing import List, Dict, Optional, Tuple
from ..core.scene_graph import SceneGraph
from ..core.task_executor import TaskExecutor, Action
from .llm_prompter import LLMPrompter


class FailureAnalyzer:
    """Analyzes robot failures using progressive reasoning"""
    
    def __init__(self, llm_prompter: LLMPrompter):
        self.llm_prompter = llm_prompter
    
    def analyze_failure(self, task_executor: TaskExecutor, 
                       scene_graphs: Dict[int, SceneGraph],
                       task_info: Dict) -> Dict:
        """
        Analyze failures in task execution
        
        Args:
            task_executor: TaskExecutor with action history
            scene_graphs: Dictionary mapping frame indices to scene graphs
            task_info: Task information
            
        Returns:
            Dictionary with failure analysis results
        """
        failed_actions = task_executor.get_failed_actions()
        
        if not failed_actions:
            # Check if task goal was achieved
            return self._analyze_planning_failure(task_executor, scene_graphs, task_info)
        
        # Analyze execution failures
        failure_analysis = {
            'failure_type': 'execution',
            'failed_actions': [],
            'explanations': []
        }
        
        for action in failed_actions:
            # Get scene graph at failure point
            action_idx = task_executor.actions.index(action)
            scene_graph = self._get_scene_graph_at_action(scene_graphs, action_idx)
            
            observation = scene_graph.to_text() if scene_graph else "No observation available"
            
            explanation = self.llm_prompter.explain_execution_failure(
                task=task_info.get('name', ''),
                action=f"{action.action_type} {action.target or ''}",
                observation=observation
            )
            
            failure_analysis['failed_actions'].append({
                'action': action.action_type,
                'target': action.target,
                'explanation': explanation
            })
            failure_analysis['explanations'].append(explanation)
        
        return failure_analysis
    
    def _analyze_planning_failure(self, task_executor: TaskExecutor,
                                 scene_graphs: Dict[int, SceneGraph],
                                 task_info: Dict) -> Dict:
        """Analyze planning-level failures"""
        # Get final state
        final_scene_graph = None
        if scene_graphs:
            final_scene_graph = list(scene_graphs.values())[-1]
        
        final_state = final_scene_graph.to_text() if final_scene_graph else "Unknown"
        
        # Get plan
        plan = "\n".join([f"{i+1}. {a.action_type} {a.target or ''}" 
                         for i, a in enumerate(task_executor.actions)])
        
        goal = task_info.get('success_condition', 'Task completion')
        
        explanation = self.llm_prompter.explain_planning_failure(
            task=task_info.get('name', ''),
            plan=plan,
            final_state=final_state,
            goal=goal
        )
        
        return {
            'failure_type': 'planning',
            'explanation': explanation,
            'final_state': final_state,
            'expected_goal': goal
        }
    
    def _get_scene_graph_at_action(self, scene_graphs: Dict[int, SceneGraph], 
                                   action_idx: int) -> Optional[SceneGraph]:
        """Get scene graph closest to action index"""
        if not scene_graphs:
            return None
        
        # Find closest frame to action
        frame_indices = sorted(scene_graphs.keys())
        if not frame_indices:
            return None
        
        # Simple heuristic: use frame index closest to action index
        closest_frame = min(frame_indices, key=lambda x: abs(x - action_idx * 10))
        return scene_graphs[closest_frame]
    
    def verify_subgoals(self, subgoals: List[Dict], scene_graphs: Dict[int, SceneGraph],
                       task_info: Dict) -> List[Dict]:
        """
        Verify each subgoal and identify failures
        
        Args:
            subgoals: List of subgoal dictionaries with 'goal' and 'frame_idx'
            scene_graphs: Dictionary mapping frame indices to scene graphs
            task_info: Task information
            
        Returns:
            List of verification results
        """
        results = []
        
        for subgoal in subgoals:
            goal = subgoal.get('goal', '')
            frame_idx = subgoal.get('frame_idx', 0)
            
            scene_graph = scene_graphs.get(frame_idx)
            observation = scene_graph.to_text() if scene_graph else "No observation"
            
            is_success, explanation = self.llm_prompter.verify_subgoal(
                task=task_info.get('name', ''),
                subgoal=goal,
                observation=observation
            )
            
            results.append({
                'subgoal': goal,
                'frame_idx': frame_idx,
                'success': is_success,
                'explanation': explanation
            })
            
            if not is_success:
                # Found a failure, can stop here for execution analysis
                break
        
        return results


