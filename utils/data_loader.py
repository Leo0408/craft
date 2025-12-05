"""
Data Loader
Loads robot execution data
"""

import json
import os
from typing import Dict, List, Optional
import numpy as np
from PIL import Image


class DataLoader:
    """Loads robot execution data from various formats"""
    
    def __init__(self, data_root: str):
        self.data_root = data_root
    
    def load_task_info(self, task_file: str) -> Dict:
        """Load task information from JSON file"""
        task_path = os.path.join(self.data_root, task_file)
        with open(task_path, 'r') as f:
            return json.load(f)
    
    def load_image(self, image_path: str) -> Image.Image:
        """Load RGB image"""
        full_path = os.path.join(self.data_root, image_path)
        return Image.open(full_path).convert('RGB')
    
    def load_depth(self, depth_path: str) -> np.ndarray:
        """Load depth image"""
        full_path = os.path.join(self.data_root, depth_path)
        # Placeholder - actual implementation depends on depth format
        # Could be .npy, .png, .zarr, etc.
        return np.load(full_path) if full_path.endswith('.npy') else np.array([])
    
    def load_frame_data(self, frame_idx: int, folder_name: str) -> Dict:
        """
        Load data for a specific frame
        
        Args:
            frame_idx: Frame index
            folder_name: Task folder name
            
        Returns:
            Dictionary with 'rgb', 'depth', and metadata
        """
        rgb_path = f"{folder_name}/videos/color/{frame_idx}.0.0.0"
        depth_path = f"{folder_name}/videos/depth/{frame_idx}.0.0"
        
        data = {
            'rgb': self.load_image(rgb_path) if os.path.exists(os.path.join(self.data_root, rgb_path)) else None,
            'depth': self.load_depth(depth_path) if os.path.exists(os.path.join(self.data_root, depth_path)) else None,
            'frame_idx': frame_idx
        }
        
        return data




