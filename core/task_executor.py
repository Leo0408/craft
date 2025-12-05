"""
Task Executor Module
Manages task execution and tracks robot actions
"""

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class ActionStatus(Enum):
    """Status of an action"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"


@dataclass
class Action:
    """Represents a robot action"""
    action_type: str  # e.g., "pick_up", "put_in", "navigate_to"
    target: Optional[str] = None
    source: Optional[str] = None
    status: ActionStatus = ActionStatus.PENDING
    timestamp: Optional[float] = None
    failure_reason: Optional[str] = None


class TaskExecutor:
    """Manages task execution and action tracking"""
    
    def __init__(self, task_name: str, actions: List[Dict]):
        self.task_name = task_name
        self.actions: List[Action] = []
        self.current_action_idx = 0
        self.execution_history: List[Dict] = []
        
        # Parse actions
        for action_dict in actions:
            action = Action(
                action_type=action_dict.get('type', ''),
                target=action_dict.get('target'),
                source=action_dict.get('source')
            )
            self.actions.append(action)
    
    def get_current_action(self) -> Optional[Action]:
        """Get the current action being executed"""
        if self.current_action_idx < len(self.actions):
            return self.actions[self.current_action_idx]
        return None
    
    def mark_action_success(self, action_idx: Optional[int] = None):
        """Mark an action as successful"""
        idx = action_idx if action_idx is not None else self.current_action_idx
        if idx < len(self.actions):
            self.actions[idx].status = ActionStatus.SUCCESS
            if idx == self.current_action_idx:
                self.current_action_idx += 1
    
    def mark_action_failed(self, failure_reason: str, action_idx: Optional[int] = None):
        """Mark an action as failed"""
        idx = action_idx if action_idx is not None else self.current_action_idx
        if idx < len(self.actions):
            self.actions[idx].status = ActionStatus.FAILED
            self.actions[idx].failure_reason = failure_reason
    
    def get_failed_actions(self) -> List[Action]:
        """Get all failed actions"""
        return [action for action in self.actions if action.status == ActionStatus.FAILED]
    
    def is_task_complete(self) -> bool:
        """Check if all actions are completed"""
        return all(action.status in [ActionStatus.SUCCESS, ActionStatus.FAILED] 
                  for action in self.actions)

