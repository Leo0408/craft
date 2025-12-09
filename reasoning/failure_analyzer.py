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
    
    def analyze_failure(self, task_executor: Optional[TaskExecutor] = None,
                       initial_scene_graph: Optional[SceneGraph] = None,
                       final_scene_graph: Optional[SceneGraph] = None,
                       failed_constraints: Optional[List[Dict]] = None,
                       scene_graphs: Optional[Dict[int, SceneGraph]] = None,
                       task_info: Optional[Dict] = None) -> Dict:
        """
        Analyze failures in task execution
        
        Args:
            task_executor: TaskExecutor with action history (optional)
            initial_scene_graph: Initial scene graph (optional)
            final_scene_graph: Final scene graph (optional)
            failed_constraints: List of violated constraints with action context (optional)
            scene_graphs: Dictionary mapping frame indices to scene graphs (optional, for backward compatibility)
            task_info: Task information (optional)
            
        Returns:
            Dictionary with failure analysis results including root_cause, causal_chain, detailed_analysis
        """
        analysis = {
            'root_cause': None,
            'causal_chain': None,
            'detailed_analysis': None,
            'violated_constraints': failed_constraints or [],
            'failed_actions': []
        }
        
        # Get task info
        if task_info is None and task_executor is not None:
            task_info = task_executor.task_info if hasattr(task_executor, 'task_info') else {}
        task_info = task_info or {}
        task_name = task_info.get('name', 'Unknown task')
        
        # Priority 1: Analyze constraint violations
        if failed_constraints:
            # Build constraint violation descriptions
            constraint_descriptions = []
            for vc in failed_constraints:
                constraint = vc.get('constraint', {})
                action = vc.get('action', 'unknown action')
                action_idx = vc.get('action_idx', 0)
                reason = vc.get('reason', '')
                eval_time = vc.get('eval_time', 'unknown')
                
                constraint_descriptions.append(
                    f"- {eval_time.upper()} {action} (step {action_idx}): "
                    f"{constraint.get('description', 'Unknown constraint')} - {reason}"
                )
            
            # Get scene descriptions
            initial_state = initial_scene_graph.to_text() if initial_scene_graph else "Unknown"
            final_state = final_scene_graph.to_text() if final_scene_graph else "Unknown"
            
            # Generate root cause analysis using LLM
            prompt = f"""Task: {task_name}

Constraint Violations Detected:
{chr(10).join(constraint_descriptions)}

Initial State: {initial_state[:500]}
Final State: {final_state[:500]}

Analyze the root cause of these constraint violations. Explain:
1. What went wrong?
2. Why did the constraints fail?
3. What should have been done differently?

Provide a clear, concise root cause analysis."""
            
            root_cause, _ = self.llm_prompter.query(
                "You are a robot failure analyst. Analyze constraint violations and identify root causes.",
                prompt,
                max_tokens=500
            )
            analysis['root_cause'] = root_cause
            
            # Generate causal chain
            causal_prompt = f"""Task: {task_name}

Constraint Violations:
{chr(10).join(constraint_descriptions)}

Create a causal chain explaining the sequence of events that led to these violations.
Format as a numbered list showing the progression of the failure."""
            
            causal_chain, _ = self.llm_prompter.query(
                "You are a robot failure analyst. Create causal chains explaining failure progression.",
                causal_prompt,
                max_tokens=400
            )
            analysis['causal_chain'] = causal_chain
            
            # Generate detailed analysis
            detailed_prompt = f"""Task: {task_name}

Constraint Violations:
{chr(10).join(constraint_descriptions)}

Initial State: {initial_state[:300]}
Final State: {final_state[:300]}

Provide a detailed analysis including:
1. What specific constraints were violated and why
2. What actions led to these violations
3. What corrective actions should be taken"""
            
            detailed_analysis, _ = self.llm_prompter.query(
                "You are a robot failure analyst. Provide detailed failure analysis with corrective recommendations.",
                detailed_prompt,
                max_tokens=600
            )
            analysis['detailed_analysis'] = detailed_analysis
        
        # Priority 2: Analyze action failures (for backward compatibility)
        if task_executor is not None:
            failed_actions = task_executor.get_failed_actions() if hasattr(task_executor, 'get_failed_actions') else []
            
            if failed_actions and scene_graphs:
                for action in failed_actions:
                    action_idx = task_executor.actions.index(action) if hasattr(task_executor, 'actions') else 0
                    scene_graph = self._get_scene_graph_at_action(scene_graphs, action_idx)
                    
                    observation = scene_graph.to_text() if scene_graph else "No observation available"
                    
                    explanation = self.llm_prompter.explain_execution_failure(
                        task=task_name,
                        action=f"{action.action_type} {action.target or ''}",
                        observation=observation
                    )
                    
                    analysis['failed_actions'].append({
                        'action': action.action_type,
                        'target': action.target,
                        'explanation': explanation
                    })
        
        return analysis
    
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


