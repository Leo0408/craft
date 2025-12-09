"""
Task utility class for managing AI2THOR task execution
Simplified version adapted from REFLECT framework
"""

import os
import json
import pickle
import numpy as np
from typing import Dict, List, Optional
from .constants import TASK_DICT, FAILURE_TYPES


class TaskUtil:
    """Task utility class for managing task execution state and failures"""
    
    def __init__(self, 
                 folder_name: str,
                 controller,
                 reachable_positions: List[Dict],
                 failure_injection: bool = False,
                 index: int = 0,
                 repo_path: str = ".",
                 chosen_failure: Optional[str] = None,
                 failure_injection_params: Optional[Dict] = None,
                 counter: int = 0):
        """
        Initialize TaskUtil
        
        Args:
            folder_name: Task folder name
            controller: AI2THOR controller instance
            reachable_positions: List of reachable positions
            failure_injection: Whether to inject failures
            index: Sample index
            repo_path: Repository path
            chosen_failure: Specific failure type to inject
            failure_injection_params: Parameters for failure injection
            counter: Initial step counter
        """
        self.counter = counter
        self.repo_path = repo_path
        self.folder_name = folder_name
        self.controller = controller
        self.reachable_positions = reachable_positions
        
        # Create grid for path planning
        self.grid = self.create_graph()
        self.reachable_points = self.get_2d_reachable_points()
        
        # Action tracking
        self.interact_actions = {}
        self.nav_actions = {}
        
        # Failure injection
        self.failure_added = False
        self.failures = FAILURE_TYPES
        self.failure_injection_params = failure_injection_params or {}
        
        if failure_injection and chosen_failure is None:
            i = index % len(self.failures)
            self.chosen_failure = self.failures[i]
            print(f"[INFO] Chosen failure: {self.chosen_failure}")
        else:
            self.chosen_failure = chosen_failure
        
        # Ground truth failure tracking
        self.gt_failure = {}
        
        # Object location tracking
        self.objs_w_unk_loc = []
        
        # Unity name mapping for objects with multiple instances
        self.unity_name_map = self.get_unity_name_map()
        
        # Action primitives that are interactions
        self.interact_action_primitives = [
            'put_on', 'put_in', 'pick_up', 'slice_obj', 
            'toggle_on', 'toggle_off', 'open_obj', 'close_obj', 
            'pour', 'crack_obj'
        ]
        
        # Load previously injected failures
        self.failures_already_injected = []
        pickle_path = f'{self.repo_path}/thor_tasks/{folder_name.split("/")[0]}/{folder_name.split("/")[1]}.pickle'
        if os.path.exists(pickle_path):
            with open(pickle_path, 'rb') as handle:
                self.failures_already_injected = pickle.load(handle)
    
    def get_unity_name_map(self) -> Dict[str, str]:
        """Map Unity object names to object types with indices"""
        obj_list = ['CounterTop', 'StoveBurner', 'Cabinet', 'Faucet', 'Sink']
        obj_rep_map = {}
        
        for obj in self.controller.last_event.metadata["objects"]:
            if obj["objectType"] in obj_list:
                if obj["objectType"] in obj_rep_map:
                    obj_rep_map[obj["objectType"]] += 1
                else:
                    obj_rep_map[obj["objectType"]] = 1
        
        # Remove objects that appear only once
        for key in list(obj_rep_map.keys()):
            if obj_rep_map[key] == 1:
                obj_list.remove(key)
        
        unity_name_map = {}
        for obj_type in obj_list:
            counter = 0
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] == obj_type:
                    counter += 1
                    unity_name_map[obj['name']] = obj_type + '-' + str(counter)
        
        return unity_name_map
    
    def create_graph(self, gridSize: float = 0.25, min_val: float = -5, max_val: float = 5.1) -> np.ndarray:
        """Create a grid for path planning"""
        grid = np.mgrid[min_val:max_val:gridSize, min_val:max_val:gridSize].transpose(1, 2, 0)
        return grid
    
    def get_2d_reachable_points(self) -> np.ndarray:
        """Get 2D reachable points (x, z coordinates)"""
        reachable_points = []
        for p in self.reachable_positions:
            reachable_points.append([p['x'], p['z']])
        return np.array(reachable_points)


def closest_position(
    object_position: Dict[str, float],
    reachable_positions: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Find the closest reachable position to an object
    
    Args:
        object_position: Object position dict with 'x', 'y', 'z'
        reachable_positions: List of reachable position dicts
        
    Returns:
        Closest reachable position
    """
    out = reachable_positions[0]
    min_distance = float('inf')
    
    for pos in reachable_positions:
        # Only care about x/z ground positions (y is vertical)
        dist = sum([(pos[key] - object_position[key]) ** 2 for key in ["x", "z"]])
        if dist < min_distance:
            min_distance = dist
            out = pos
    
    return out


# BFS path finding classes and functions
class Node:
    """Node for BFS path finding"""
    def __init__(self, x: int, y: int, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
    
    def __repr__(self):
        return str((self.x, self.y))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


# Movement directions for BFS
ROW = [-1, 0, 0, 1]
COL = [0, -1, 1, 0]


def is_valid(x: int, y: int, N: int, reachable_points: np.ndarray, grid: np.ndarray) -> bool:
    """Check if a position is valid for path finding"""
    if x < 0 or y < 0 or x >= N or y >= N:
        return False
    
    val = grid[x][y]
    if val.tolist() not in reachable_points.tolist():
        return False
    
    return True


def get_path(node: Node, path: List = None) -> List[Node]:
    """Get path from root to node"""
    if path is None:
        path = []
    if node:
        get_path(node.parent, path)
        path.append(node)
    return path


def find_path(grid: np.ndarray, x: int, y: int, target_pos: List[int], 
              reachable_points: np.ndarray) -> Optional[List[Node]]:
    """
    Find path using BFS
    
    Args:
        grid: Grid array
        x: Start x coordinate
        y: Start y coordinate
        target_pos: Target position [x, y]
        reachable_points: Array of reachable points
        
    Returns:
        List of nodes representing the path, or None if no path found
    """
    N = grid.shape[0]
    
    # Check if start and end are valid
    if not is_valid(x, y, N, reachable_points, grid):
        return None
    
    if not is_valid(target_pos[0], target_pos[1], N, reachable_points, grid):
        return None
    
    # BFS
    visited = set()
    queue = [Node(x, y)]
    visited.add((x, y))
    
    while queue:
        node = queue.pop(0)
        
        # Check if reached target
        if node.x == target_pos[0] and node.y == target_pos[1]:
            return get_path(node)
        
        # Explore neighbors
        for i in range(4):
            new_x = node.x + ROW[i]
            new_y = node.y + COL[i]
            
            if is_valid(new_x, new_y, N, reachable_points, grid) and (new_x, new_y) not in visited:
                visited.add((new_x, new_y))
                queue.append(Node(new_x, new_y, node))
    
    return None

