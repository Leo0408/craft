"""
Simulated Scene Graph Builder
Builds scene graphs from AI2THOR simulation environment
Adapted from reflect/main/get_local_sg.py and reflect/main/exp.py
"""

import os
import numpy as np
import torch
import open3d as o3d
from typing import Dict, List, Optional, Tuple
from PIL import Image

from ..core.scene_graph import SceneGraph, Node, Edge
from .scene_analyzer import SceneAnalyzer


class SimulatedSceneGraphBuilder:
    """Builds scene graphs from AI2THOR simulation events"""
    
    def __init__(self, 
                 scene_analyzer: Optional[SceneAnalyzer] = None,
                 voxel_size: float = 0.01):
        """
        Initialize the simulated scene graph builder
        
        Args:
            scene_analyzer: Scene analyzer instance
            voxel_size: Voxel size for point cloud downsampling
        """
        self.scene_analyzer = scene_analyzer or SceneAnalyzer()
        self.voxel_size = voxel_size
        
        # Accumulated point clouds across frames
        self.total_points_dict: Dict[str, torch.Tensor] = {}
        self.bbox3d_dict: Dict[str, o3d.geometry.AxisAlignedBoundingBox] = {}
        self.obj_held_prev: Optional[str] = None
    
    def process_event(self,
                     step_idx: int,
                     event,
                     object_list: List[str],
                     task: Optional[Dict] = None) -> SceneGraph:
        """
        Process an AI2THOR event and generate scene graph
        
        Args:
            step_idx: Frame index
            event: AI2THOR event object
            object_list: List of objects to detect
            task: Task information dictionary
            
        Returns:
            SceneGraph object
        """
        # Try to import reflect utilities (optional dependency)
        try:
            from reflect.main.point_cloud_utils import (
                depth_frame_to_camera_space_xyz,
                camera_space_xyz_to_world_xyz
            )
            from reflect.main.utils import (
                get_label_from_object_id,
                is_receptacle,
                is_moving,
                is_picked_up
            )
        except ImportError:
            raise ImportError(
                "REFLECT utilities not found. "
                "For simulated environment, ensure reflect is in Python path or "
                "use real-world data option instead."
            )
        
        pcd_dict: Dict[str, torch.Tensor] = {}
        depth_dict: Dict[str, np.ndarray] = {}
        
        height, width, channel = event.frame.shape
        
        # Convert depth to camera space
        camera_space_xyz = depth_frame_to_camera_space_xyz(
            depth_frame=torch.as_tensor(event.depth_frame.copy()),
            mask=None,
            fov=event.metadata['fov']
        )
        
        # Get agent position
        x = event.metadata['agent']['position']['x']
        y = event.metadata['agent']['position']['y']
        z = event.metadata['agent']['position']['z']
        
        if not event.metadata['agent']['isStanding']:
            y = y - 0.22
        
        # Convert to world space
        world_points = camera_space_xyz_to_world_xyz(
            camera_space_xyzs=camera_space_xyz,
            camera_world_xyz=torch.as_tensor([x, y, z]),
            rotation=event.metadata['agent']['rotation']['y'],
            horizon=event.metadata['agent']['cameraHorizon'],
        ).reshape(channel, height, width).permute(1, 2, 0)
        
        # Process each object in instance masks
        sinkbasin_pts = None
        for object_id in event.instance_masks:
            # Skip background objects
            if object_id.split("|")[0] in ["Window", "Floor", "Wall", "Ceiling", "Cabinet"]:
                continue
            
            label = object_id
            
            # Get mask and points
            mask = event.instance_masks[object_id].reshape(height, width)
            obj_points = torch.as_tensor(world_points[mask])
            
            if len(obj_points) < 700:
                continue
            
            depth_dict[label] = event.depth_frame[mask]
            
            # Downsample point cloud
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(obj_points.numpy())
            voxel_down_pcd = obj_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
            # Denoise point cloud (similar to reflect)
            if "Pan" == label.split("|")[0] or "EggCracked" == label.split("|")[0] or \
               "Bowl" == label.split("|")[0] or "Pot" == label.split("|")[0]:
                _, ind = voxel_down_pcd.remove_radius_outlier(nb_points=30, radius=0.03)
                inlier = voxel_down_pcd.select_by_index(ind)
                pcd_dict[label] = torch.tensor(np.array(inlier.points))
            elif "CounterTop" == label.split("|")[0]:
                _, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.1)
                inlier = voxel_down_pcd.select_by_index(ind)
                pcd_dict[label] = torch.tensor(np.array(inlier.points))
            elif "SinkBasin" in label:
                sinkbasin_pts = torch.tensor(np.array(voxel_down_pcd.points))
            else:
                pcd_dict[label] = torch.tensor(np.array(voxel_down_pcd.points))
        
        # Accumulate point clouds
        for label in pcd_dict.keys():
            # Handle state changes (e.g., Sliced, Cracked)
            for keyword in ["Sliced", "Cracked"]:
                if keyword in label:
                    tmp = ""
                    for key in self.total_points_dict.keys():
                        if len(key.split("|")) == 4 and key.split("|")[0] == label.split("|")[0]:
                            tmp = key
                    if len(tmp) != 0 and (keyword not in tmp) and (tmp in self.total_points_dict):
                        print("remove object:", tmp)
                        del self.total_points_dict[tmp]
            
            if label not in self.total_points_dict:
                self.total_points_dict[label] = pcd_dict[label]
            
            # For receptacles, accumulate points
            if is_receptacle(label, event):
                if is_moving(label, event) or is_picked_up(label, event) or self.obj_held_prev == label:
                    self.total_points_dict[label] = pcd_dict[label]
                else:
                    self.total_points_dict[label] = torch.unique(
                        torch.cat((self.total_points_dict[label], pcd_dict[label]), 0), dim=0
                    )
            else:
                self.total_points_dict[label] = pcd_dict[label]
            
            # Handle sink basin
            if label.split("|")[0] == "Sink" and sinkbasin_pts is not None:
                self.total_points_dict[label] = torch.unique(
                    torch.cat((self.total_points_dict[label], sinkbasin_pts), 0), dim=0
                )
        
        # Remove dropped object
        if self.obj_held_prev not in pcd_dict.keys():
            if self.obj_held_prev in self.total_points_dict.keys():
                print("remove object:", self.obj_held_prev)
                del self.total_points_dict[self.obj_held_prev]
        
        # Compute 3D bounding boxes
        for label in self.total_points_dict.keys():
            boxes3d_pts = o3d.utility.Vector3dVector(self.total_points_dict[label].numpy())
            box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)
            self.bbox3d_dict[label] = box
        
        # Generate local scene graph
        local_sg = SceneGraph(task=task)
        for label in pcd_dict.keys():
            name = get_label_from_object_id(label, [event], task)
            if name is not None:
                # Get 2D bbox (simplified - in real implementation would project 3D to 2D)
                bbox = None  # Would compute from 3D points
                
                node = Node(
                    name=name,
                    object_type=label.split("|")[0],
                    position=tuple(self.bbox3d_dict[label].get_center()),
                    attributes={
                        'object_id': label,
                        'pos3d': self.bbox3d_dict[label].get_center(),
                        'corner_pts': np.array(self.bbox3d_dict[label].get_box_points()),
                        'bbox2d': bbox,
                        'pcd': self.total_points_dict[label].numpy(),
                        'depth': depth_dict.get(label)
                    }
                )
                local_sg.add_node(node)
        
        # Add objects from object_list
        for label in pcd_dict.keys():
            object_name = label.split("|")[0]
            if object_name in object_list:
                node = next((node for node in local_sg.nodes if 
                           hasattr(node, 'attributes') and 
                           node.attributes.get('object_id') == label), None)
                if node is not None:
                    # Node already added
                    pass
        
        # Add agent (robot gripper)
        self.obj_held_prev = self._add_agent(local_sg, event)
        
        return local_sg
    
    def _add_agent(self, scene_graph: SceneGraph, event) -> Optional[str]:
        """Add agent (robot gripper) to scene graph"""
        try:
            from reflect.main.utils import get_label_from_object_id
        except ImportError:
            # Fallback if reflect not available
            return None
        
        # Check if holding object
        for obj in event.metadata.get("objects", []):
            if obj.get("isPickedUp", False):
                obj_name = get_label_from_object_id(
                    obj["objectId"], [event], None
                )
                if obj_name:
                    # Add edge: object is inside robot gripper
                    node = scene_graph.get_node(obj_name)
                    if node:
                        gripper_node = Node("robot gripper", "RobotGripper")
                        scene_graph.add_node(gripper_node)
                        edge = Edge(node, gripper_node, "inside")
                        scene_graph.add_edge(edge)
                        return obj["objectId"]
        
        # No object held
        gripper_node = Node("robot gripper", "RobotGripper")
        scene_graph.add_node(gripper_node)
        nothing_node = Node("nothing", "Nothing")
        scene_graph.add_node(nothing_node)
        edge = Edge(nothing_node, gripper_node, "inside")
        scene_graph.add_edge(edge)
        return None
    
    def reset(self):
        """Reset accumulated state"""
        self.total_points_dict = {}
        self.bbox3d_dict = {}
        self.obj_held_prev = None

