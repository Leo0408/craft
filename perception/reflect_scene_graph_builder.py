"""
Reflect Scene Graph Builder
Builds scene graphs from REFLECT dataset format (zarr files)
Adapted from reflect/real-world/real_world_get_local_sg.py
"""

import numpy as np
import torch
import open3d as o3d
from PIL import Image
from typing import Dict, List, Optional, Tuple
import zarr
from imagecodecs import imread

from ..core.scene_graph import SceneGraph, Node, Edge
from .object_detector import ObjectDetector
from .scene_analyzer import SceneAnalyzer


class ReflectSceneGraphBuilder:
    """Builds scene graphs from REFLECT dataset format"""
    
    def __init__(self, 
                 detector: Optional[ObjectDetector] = None,
                 scene_analyzer: Optional[SceneAnalyzer] = None,
                 camera_intrinsics: Optional[Dict] = None,
                 voxel_size: float = 0.01):
        """
        Initialize the scene graph builder
        
        Args:
            detector: Object detector instance
            scene_analyzer: Scene analyzer instance
            camera_intrinsics: Camera intrinsic parameters
            voxel_size: Voxel size for point cloud downsampling
        """
        self.detector = detector
        self.scene_analyzer = scene_analyzer or SceneAnalyzer()
        self.voxel_size = voxel_size
        
        # Default camera intrinsics (RealSense)
        self.camera_intrinsics = camera_intrinsics or {
            "fx": 914.27246,
            "fy": 913.2658,
            "cx": 647.0733,
            "cy": 356.32526
        }
        
        # Accumulated point clouds across frames
        self.total_points_dict: Dict[str, np.ndarray] = {}
        self.bbox3d_dict: Dict[str, o3d.geometry.AxisAlignedBoundingBox] = {}
    
    def depth_to_point_cloud(self, depth: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Convert depth image to 3D point cloud
        
        Args:
            depth: Depth image (H x W)
            mask: Optional mask to filter points (boolean array)
            
        Returns:
            Point cloud array (N x 3) - points in camera coordinates
        """
        h, w = depth.shape
        fx = self.camera_intrinsics["fx"]
        fy = self.camera_intrinsics["fy"]
        cx = self.camera_intrinsics["cx"]
        cy = self.camera_intrinsics["cy"]
        
        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply mask if provided - ensure mask is boolean
        if mask is not None:
            if mask.dtype != bool:
                mask = mask.astype(bool)
            # Only use points that are: 1) in the mask, 2) have valid depth (> 0)
            valid_mask = (depth > 0) & mask
        else:
            valid_mask = depth > 0
        
        # Extract valid pixel coordinates and depths
        u_valid = u[valid_mask]
        v_valid = v[valid_mask]
        z_valid = depth[valid_mask]
        
        if len(z_valid) == 0:
            return np.array([]).reshape(0, 3)
        
        # Convert to 3D coordinates (camera space)
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        # z = depth
        x = (u_valid - cx) * z_valid / fx
        y = (v_valid - cy) * z_valid / fy
        
        # Stack into point cloud (N x 3)
        points = np.stack([x, y, z_valid], axis=1)
        
        return points
    
    def process_frame(self,
                     rgb: np.ndarray,
                     depth: np.ndarray,
                     step_idx: int,
                     object_list: List[str],
                     distractor_list: Optional[List[str]] = None,
                     task_info: Optional[Dict] = None) -> SceneGraph:
        """
        Process a single frame and generate scene graph
        
        Args:
            rgb: RGB image (H x W x 3)
            depth: Depth image (H x W)
            step_idx: Frame index
            object_list: List of objects to detect
            distractor_list: Optional list of distractors to filter
            task_info: Task information dictionary
            
        Returns:
            SceneGraph object
        """
        if distractor_list is None:
            distractor_list = []
        
        pcd_dict: Dict[str, np.ndarray] = {}
        bbox2d_dict: Dict[str, np.ndarray] = {}
        local_sg = SceneGraph(task=task_info)
        
        # Object detection
        if self.detector is None:
            raise ValueError("Object detector must be provided")
        
        # Convert RGB to PIL Image
        rgb_pil = Image.fromarray(rgb)
        
        # Detect objects
        detections = self.detector.detect_with_depth(
            rgb_pil,
            depth,
            object_list,
            self.camera_intrinsics
        )
        
        if len(detections) == 0:
            print(f"Nothing detected in frame {step_idx}")
            return local_sg
        
        # Process each detection
        for det in detections:
            label = det['label']
            
            # Filter distractors
            if label.split("-")[0] in distractor_list:
                print(f"Filtering out distractor: {label}")
                continue
            
            # Get mask from detection (if available)
            mask = det.get('mask')
            if mask is None:
                # Create mask from bbox
                bbox = det['bbox']
                print(f"  Processing {label} with bbox: {bbox}")
                
                mask = np.zeros(depth.shape, dtype=bool)
                # Ensure bbox coordinates are valid
                y1, y2 = max(0, int(bbox[1])), min(depth.shape[0], int(bbox[3]))
                x1, x2 = max(0, int(bbox[0])), min(depth.shape[1], int(bbox[2]))
                if y2 > y1 and x2 > x1:
                    mask[y1:y2, x1:x2] = True
                    mask_area = np.sum(mask)
                    print(f"    Mask area: {mask_area} pixels")
                else:
                    print(f"⚠️  Invalid bbox for {label}: {bbox}")
                    continue
            else:
                # Use provided mask
                if mask.dtype != bool:
                    mask = mask.astype(bool)
                mask_area = np.sum(mask)
                print(f"  Processing {label} with provided mask (area: {mask_area} pixels)")
            
            # Ensure mask is boolean
            if mask.dtype != bool:
                mask = mask.astype(bool)
            
            # Convert depth to point cloud - ONLY for this object's region
            # Use the mask to extract only points within this object's bounding box
            point_3d = self.depth_to_point_cloud(depth, mask)
            print(f"    Generated {len(point_3d)} points for {label}")
            
            if len(point_3d) < 100:  # Too few points
                continue
            
            # Downsample point cloud
            obj_pcd = o3d.geometry.PointCloud()
            obj_pcd.points = o3d.utility.Vector3dVector(point_3d)
            voxel_down_pcd = obj_pcd.voxel_down_sample(voxel_size=self.voxel_size)
            
            # Denoise point cloud
            _, ind = voxel_down_pcd.remove_statistical_outlier(
                nb_neighbors=20, 
                std_ratio=0.1
            )
            inlier = voxel_down_pcd.select_by_index(ind)
            
            # Store point cloud for this object (in camera coordinates)
            # IMPORTANT: Make a copy to avoid sharing references
            obj_points = np.array(inlier.points).copy()
            pcd_dict[label] = obj_points.copy()  # Store in pcd_dict
            
            print(f"    After processing: {len(obj_points)} points for {label}")
            
            # For static objects (tables, counters), accumulate points across frames
            # For movable objects, use current frame's points only
            if label in ["table", "coffee machine", "countertop", "counter", "table on the left of sink"] and label in self.total_points_dict:
                # Accumulate for static objects
                self.total_points_dict[label] = np.concatenate(
                    (self.total_points_dict[label], obj_points)
                )
            else:
                # Use current frame's points for movable objects
                self.total_points_dict[label] = obj_points.copy()  # Make sure it's a copy
            
            print(f"    Stored {len(self.total_points_dict[label])} points in total_points_dict for {label}")
            
            # Compute 3D bounding box from THIS OBJECT'S point cloud only
            if len(self.total_points_dict[label]) > 0:
                boxes3d_pts = o3d.utility.Vector3dVector(self.total_points_dict[label])
                box = o3d.geometry.AxisAlignedBoundingBox.create_from_points(boxes3d_pts)
                self.bbox3d_dict[label] = box
            else:
                print(f"⚠️  No points for {label}, skipping")
                continue
            
            bbox2d_dict[label] = np.array(det['bbox'])
        
        # Build scene graph nodes
        for label in pcd_dict.keys():
            if label not in self.bbox3d_dict:
                continue
                
            bbox3d = self.bbox3d_dict[label]
            
            # IMPORTANT: Use pcd_dict[label] which contains THIS FRAME's points for this object
            # NOT total_points_dict which might be accumulated
            obj_points = pcd_dict[label].copy()  # Use the frame-specific point cloud
            
            if len(obj_points) > 0:
                # Use centroid of THIS OBJECT'S point cloud
                pos3d = np.mean(obj_points, axis=0)
                print(f"  Node {label}: {len(obj_points)} points, centroid: ({pos3d[0]:.2f}, {pos3d[1]:.2f}, {pos3d[2]:.2f})")
            else:
                # Fallback to bbox center if no points
                pos3d = bbox3d.get_center()
                print(f"  Node {label}: Using bbox center (no points)")
            
            # Create node with unique 3D position
            node = Node(
                name=label,
                object_type=label.split("-")[0],
                position=tuple(pos3d),  # Unique position for each object
                attributes={
                    'object_id': label,
                    'bbox3d': bbox3d,
                    'bbox2d': bbox2d_dict[label],
                    'pcd': obj_points.copy(),  # This object's point cloud (make sure it's a copy)
                    'centroid': pos3d.copy() if isinstance(pos3d, np.ndarray) else pos3d  # Store centroid separately
                }
            )
            local_sg.add_node(node)
        
            # Compute spatial relationships using each object's unique centroid
            # IMPORTANT: Use pcd_dict (frame-specific) not total_points_dict (accumulated)
            detections_for_relations = []
            for label in pcd_dict.keys():
                if label not in self.bbox3d_dict:
                    continue
                # Use THIS FRAME's point cloud for this object
                obj_points = pcd_dict[label]
                if len(obj_points) > 0:
                    # Use centroid of THIS OBJECT'S point cloud
                    centroid = np.mean(obj_points, axis=0)
                    detections_for_relations.append({
                        'label': label,
                        'position_3d': tuple(centroid)  # Unique position for each object
                    })
                    print(f"  Relation input for {label}: {len(obj_points)} points, centroid: {centroid}")
            
            relations = self.scene_analyzer.compute_spatial_relations(detections_for_relations)
        
        # Add edges based on spatial relations
        for obj1_name, obj2_name, rel_type, confidence in relations:
            node1 = local_sg.get_node(obj1_name)
            node2 = local_sg.get_node(obj2_name)
            
            if node1 and node2:
                edge = Edge(node1, node2, rel_type, confidence=confidence)
                local_sg.add_edge(edge)
        
        return local_sg
    
    def process_frame_from_zarr(self,
                               zarr_group: zarr.Group,
                               step_idx: int,
                               object_list: List[str],
                               distractor_list: Optional[List[str]] = None,
                               task_info: Optional[Dict] = None) -> SceneGraph:
        """
        Process a frame from zarr group and generate scene graph
        
        Args:
            zarr_group: Zarr group object (opened zarr file)
            step_idx: Frame index
            object_list: List of objects to detect
            distractor_list: Optional list of distractors
            task_info: Task information
            
        Returns:
            SceneGraph object
        """
        # Load RGB and depth from zarr group
        try:
            # Try different possible paths
            rgb_paths = [
                f'data/videos/color/{step_idx}.0.0.0',
                f'videos/color/{step_idx}.0.0.0',
            ]
            depth_paths = [
                f'data/videos/depth/{step_idx}.0.0',
                f'videos/depth/{step_idx}.0.0',
            ]
            
            rgb = None
            depth = None
            
            for path in rgb_paths:
                if path in zarr_group:
                    rgb_data = zarr_group[path]
                    rgb = np.array(rgb_data)
                    break
            
            for path in depth_paths:
                if path in zarr_group:
                    depth_data = zarr_group[path]
                    depth = np.array(depth_data)
                    break
            
            if rgb is None or depth is None:
                print(f"Could not load frame {step_idx} from zarr")
                return SceneGraph()
            
        except Exception as e:
            print(f"Error loading frame {step_idx}: {e}")
            return SceneGraph()
        
        # Process frame
        return self.process_frame(
            rgb=rgb,
            depth=depth,
            step_idx=step_idx,
            object_list=object_list,
            distractor_list=distractor_list,
            task_info=task_info
        )
    
    def reset(self):
        """Reset accumulated state"""
        self.total_points_dict = {}
        self.bbox3d_dict = {}

