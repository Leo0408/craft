"""
Simulated Data Generator
Generates data from AI2THOR simulation environment
Adapted from reflect/main/gen_data.py
"""

import os
import json
import pickle
from typing import Dict, List, Optional
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering


class SimulatedDataGenerator:
    """Generates robot execution data from AI2THOR simulation"""
    
    def __init__(self, 
                 scene: str,
                 width: int = 960,
                 height: int = 960,
                 fov: int = 60):
        """
        Initialize AI2THOR controller
        
        Args:
            scene: Scene name (e.g., "FloorPlan16")
            width: Image width
            height: Image height
            fov: Field of view
        """
        self.scene = scene
        self.width = width
        self.height = height
        self.fov = fov
        self.controller = None
        self.events = []
    
    def initialize(self):
        """Initialize the controller"""
        self.controller = Controller(
            agentMode="default",
            massThreshold=None,
            scene=self.scene,
            visibilityDistance=1.5,
            gridSize=0.25,
            renderDepthImage=True,
            renderInstanceSegmentation=True,
            width=self.width,
            height=self.height,
            fieldOfView=self.fov,
            platform=CloudRendering
        )
        return self.controller
    
    def execute_actions(self, actions: List[str]) -> List:
        """
        Execute a sequence of actions in the simulation
        
        Args:
            actions: List of action strings (e.g., ["MoveAhead", "PickupObject", ...])
            
        Returns:
            List of events
        """
        if self.controller is None:
            self.initialize()
        
        events = []
        for action_str in actions:
            # Parse and execute action
            # This is simplified - in real implementation would parse action format
            event = self.controller.step(action=action_str)
            events.append(event)
        
        self.events = events
        return events
    
    def save_task_data(self, folder_name: str, task_info: Dict):
        """
        Save task data to disk
        
        Args:
            folder_name: Folder name to save data
            task_info: Task information dictionary
        """
        os.makedirs(f'thor_tasks/{folder_name}', exist_ok=True)
        
        # Save task info
        with open(f'thor_tasks/{folder_name}/task.json', 'w') as f:
            json.dump(task_info, f, indent=2)
        
        # Save events (simplified - in real implementation would save frames)
        # Note: Events contain frames, depth, instance masks, etc.
        print(f"Task data saved to thor_tasks/{folder_name}/")
    
    def get_reachable_positions(self) -> List:
        """Get reachable positions in the scene"""
        if self.controller is None:
            self.initialize()
        return self.controller.step(action="GetReachablePositions").metadata["actionReturn"]



