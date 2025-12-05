"""
Configuration Loader
"""

import json
import os
from typing import Dict, Any


def load_config(config_path: str = "config/config.json") -> Dict[str, Any]:
    """
    Load configuration from JSON file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        # Return default config
        return get_default_config()
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "camera_intrinsics": {
            "fx": 914.27246,
            "fy": 913.2658,
            "cx": 647.0733,
            "cy": 356.32526
        },
        "detection_thresholds": {
            "clip_confidence": 0.23,
            "iou_threshold": 0.25,
            "distance_threshold": 0.05
        },
        "audio_settings": {
            "volume_threshold": 0.03,
            "enable_audio": False
        },
        "model_settings": {
            "object_detector": "mdetr",
            "llm_model": "gpt-3.5-turbo",
            "device": "cuda:0"
        },
        "processing_settings": {
            "video_fps": 30,
            "key_frame_interval": 150,
            "voxel_size": 0.01
        }
    }




