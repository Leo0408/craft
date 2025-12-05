"""
REFLECT Data Loader
Loads data from REFLECT dataset format (zarr files)
"""

import os
import json
import zarr
import numpy as np
from PIL import Image
from typing import Dict, List, Optional
from imagecodecs import imread


class ReflectDataLoader:
    """Loads data from REFLECT dataset format"""
    
    def __init__(self, data_root: str):
        """
        Initialize data loader
        
        Args:
            data_root: Root directory containing REFLECT data
        """
        self.data_root = data_root
    
    def load_zarr_file(self, folder_name: str) -> zarr.Group:
        """
        Load zarr file for a task
        
        Args:
            folder_name: Task folder name (e.g., "makeCoffee2")
            
        Returns:
            Zarr group object
        """
        zarr_path = os.path.join(
            self.data_root,
            "reflect_dataset",
            "real_data",
            folder_name,
            "replay_buffer.zarr"
        )
        
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
        
        return zarr.open(zarr_path, 'r')
    
    def load_frame_rgb(self, zarr_group: zarr.Group, step_idx: int, folder_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load RGB image for a frame
        
        Args:
            zarr_group: Zarr group object
            step_idx: Frame index
            folder_name: Optional folder name for file-based loading
            
        Returns:
            RGB image array or None
        """
        try:
            # Method 1: Try loading from zarr group
            rgb_paths = [
                f'data/videos/color/{step_idx}.0.0.0',
                f'videos/color/{step_idx}.0.0.0',
                f'data/videos/color/{step_idx}',
                f'videos/color/{step_idx}',
            ]
            
            for rgb_path in rgb_paths:
                if rgb_path in zarr_group:
                    rgb_data = zarr_group[rgb_path]
                    rgb_array = np.array(rgb_data)
                    # Ensure it's RGB format (H, W, 3)
                    if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
                        return rgb_array
                    elif len(rgb_array.shape) == 2:
                        # Grayscale, convert to RGB
                        return np.stack([rgb_array] * 3, axis=-1)
            
            # Method 2: Try loading from file system (as in REFLECT demo)
            if folder_name:
                try:
                    from imagecodecs import imread
                    # Try different possible file paths
                    file_paths = [
                        os.path.join(self.data_root, "reflect_dataset", "real_data", folder_name, "videos", "color", f"{step_idx}.0.0.0"),
                        os.path.join(self.data_root, "real_data", folder_name, "videos", "color", f"{step_idx}.0.0.0"),
                        os.path.join(self.data_root, folder_name, "videos", "color", f"{step_idx}.0.0.0"),
                    ]
                    
                    for file_path in file_paths:
                        if os.path.exists(file_path):
                            rgb_array = imread(file_path)
                            if rgb_array is not None:
                                # Ensure RGB format
                                if len(rgb_array.shape) == 3 and rgb_array.shape[2] == 3:
                                    return rgb_array
                                elif len(rgb_array.shape) == 2:
                                    return np.stack([rgb_array] * 3, axis=-1)
                except ImportError:
                    pass  # imagecodecs not available
                except Exception as e:
                    print(f"  File loading error: {e}")
                    
        except Exception as e:
            print(f"Error loading RGB for frame {step_idx}: {e}")
        
        return None
    
    def load_frame_depth(self, zarr_group: zarr.Group, step_idx: int, folder_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Load depth image for a frame
        
        Args:
            zarr_group: Zarr group object
            step_idx: Frame index
            folder_name: Optional folder name for file-based loading
            
        Returns:
            Depth image array or None
        """
        try:
            # Method 1: Try loading from zarr group
            depth_paths = [
                f'data/videos/depth/{step_idx}.0.0',
                f'videos/depth/{step_idx}.0.0',
                f'data/videos/depth/{step_idx}',
                f'videos/depth/{step_idx}',
            ]
            
            for depth_path in depth_paths:
                if depth_path in zarr_group:
                    depth_data = zarr_group[depth_path]
                    depth_array = np.array(depth_data)
                    # Ensure it's 2D
                    if len(depth_array.shape) == 2:
                        return depth_array
                    elif len(depth_array.shape) == 3:
                        # Take first channel if multi-channel
                        return depth_array[:, :, 0]
            
            # Method 2: Try loading from file system (as in REFLECT demo)
            if folder_name:
                try:
                    from imagecodecs import imread
                    # Try different possible file paths
                    file_paths = [
                        os.path.join(self.data_root, "reflect_dataset", "real_data", folder_name, "videos", "depth", f"{step_idx}.0.0"),
                        os.path.join(self.data_root, "real_data", folder_name, "videos", "depth", f"{step_idx}.0.0"),
                        os.path.join(self.data_root, folder_name, "videos", "depth", f"{step_idx}.0.0"),
                    ]
                    
                    for file_path in file_paths:
                        if os.path.exists(file_path):
                            depth_array = imread(file_path)
                            if depth_array is not None:
                                # Ensure 2D
                                if len(depth_array.shape) == 2:
                                    return depth_array
                                elif len(depth_array.shape) == 3:
                                    return depth_array[:, :, 0]
                except ImportError:
                    pass  # imagecodecs not available
                except Exception as e:
                    print(f"  File loading error: {e}")
                    
        except Exception as e:
            print(f"Error loading depth for frame {step_idx}: {e}")
        
        return None
    
    def load_task_info(self, folder_name: str) -> Optional[Dict]:
        """
        Load task information from JSON file
        
        Args:
            folder_name: Task folder name
            
        Returns:
            Task info dictionary or None
        """
        # Try multiple possible locations
        possible_paths = [
            os.path.join(self.data_root, "real-world", "tasks_real_world.json"),
            os.path.join(self.data_root, "main", "tasks_real_world.json"),
            os.path.join(self.data_root, "tasks_real_world.json"),
        ]
        
        for json_path in possible_paths:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    tasks = json.load(f)
                    # Find task by folder name
                    for task_key, task_info in tasks.items():
                        if task_info.get("general_folder_name") == folder_name:
                            return task_info
        
        return None
    
    def get_total_frames(self, zarr_group: zarr.Group) -> int:
        """
        Get total number of frames in zarr file
        
        Args:
            zarr_group: Zarr group object
            
        Returns:
            Total number of frames
        """
        try:
            if 'data/stage' in zarr_group:
                return len(zarr_group['data/stage'])
            elif 'stage' in zarr_group:
                return len(zarr_group['stage'])
        except:
            pass
        
        return 0
    
    def get_gripper_position(self, zarr_group: zarr.Group, step_idx: int) -> Optional[float]:
        """
        Get gripper position for a frame
        
        Args:
            zarr_group: Zarr group object
            step_idx: Frame index
            
        Returns:
            Gripper position or None
        """
        try:
            if 'data/gripper_pos' in zarr_group:
                return float(zarr_group['data/gripper_pos'][step_idx])
            elif 'gripper_pos' in zarr_group:
                return float(zarr_group['gripper_pos'][step_idx])
        except:
            pass
        
        return None


