"""
MDETR Object Detector Wrapper
Wraps REFLECT's MDETR detector for use in CRAFT framework
"""

import os
import sys
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from collections import defaultdict
import cv2

# Add REFLECT real-world directory to path if available
REFLECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reflect')
if os.path.exists(REFLECT_ROOT):
    real_world_path = os.path.join(REFLECT_ROOT, 'real-world')
    if os.path.exists(real_world_path) and real_world_path not in sys.path:
        sys.path.insert(0, real_world_path)


class MDETRDetector:
    """MDETR-based object detector for real-world environments"""
    
    def __init__(self, device: str = "cuda:0", threshold: float = 0.7, pretrained: bool = True):
        """
        Initialize MDETR detector
        
        Args:
            device: Device to run on ('cuda:0' or 'cpu')
            threshold: Detection confidence threshold
            pretrained: Whether to use pretrained weights
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold
        self.model = None
        self.transform = T.Compose([
            T.Resize(800),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._load_model(pretrained)
    
    def _load_model(self, pretrained: bool):
        """Load MDETR model"""
        try:
            # Try to import from REFLECT
            from hubconf import mdetr_efficientnetB3_phrasecut
            self.model = mdetr_efficientnetB3_phrasecut(pretrained=pretrained).to(self.device)
            self.model.eval()
            torch.set_grad_enabled(False)
            print(f"✓ MDETR detector loaded on {self.device}")
        except ImportError:
            print("⚠️  Warning: Could not import MDETR from REFLECT")
            print("   Please ensure REFLECT real-world directory is accessible")
            self.model = None
        except Exception as e:
            print(f"⚠️  Error loading MDETR: {e}")
            self.model = None
    
    def _rescale_bboxes(self, out_bbox, size):
        """Rescale bounding boxes from normalized to image coordinates"""
        def box_cxcywh_to_xyxy(x):
            x_c, y_c, w, h = x.unbind(1)
            b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
                 (x_c + 0.5 * w), (y_c + 0.5 * h)]
            return torch.stack(b, dim=1)
        
        img_w, img_h = size
        b = box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b
    
    def detect_objects(self, rgb_image: Image.Image, object_list: List[str]) -> List[Dict]:
        """
        Detect objects in the image using MDETR
        
        Args:
            rgb_image: RGB image (PIL Image)
            object_list: List of object names to detect
            
        Returns:
            List of detections with bbox, mask, confidence, label
        """
        if self.model is None:
            print("⚠️  MDETR model not loaded, returning empty detections")
            return []
        
        detections = []
        
        # Process each object separately
        for obj_name in object_list:
            try:
                # Prepare image
                img = self.transform(rgb_image).unsqueeze(0).to(self.device)
                
                # Run MDETR
                outputs = self.model(img, [obj_name])
                
                # Get predictions
                probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
                keep = (probas > self.threshold).cpu()
                
                if keep.sum() == 0:
                    continue
                
                # Rescale bounding boxes
                bboxes_scaled = self._rescale_bboxes(
                    outputs['pred_boxes'].cpu()[0, keep], 
                    rgb_image.size
                )
                
                # Get masks
                w, h = rgb_image.size
                masks = F.interpolate(
                    outputs["pred_masks"], 
                    size=(h, w), 
                    mode="bilinear", 
                    align_corners=False
                )
                masks = masks.cpu()[0, keep].sigmoid() > 0.5
                
                # Shrink masks to remove noise
                shrinked_masks = []
                for mask in masks:
                    kernel = np.ones((3, 3), np.uint8)
                    eroded_mask = cv2.erode(
                        np.array(mask, dtype=np.float32), 
                        kernel, 
                        iterations=2
                    )
                    shrinked_masks.append(eroded_mask)
                shrinked_masks = np.array(shrinked_masks) if len(shrinked_masks) > 0 else masks
                
                # Extract text spans
                tokenized = self.model.detr.transformer.tokenizer.batch_encode_plus(
                    [obj_name], padding="longest", return_tensors="pt"
                ).to(img.device)
                
                positive_tokens = (outputs["pred_logits"].cpu()[0, keep].softmax(-1) > 0.1).nonzero().tolist()
                predicted_spans = defaultdict(str)
                for tok in positive_tokens:
                    item, pos = tok
                    if pos < 255:
                        span = tokenized.token_to_chars(0, pos)
                        predicted_spans[item] += " " + obj_name[span.start:span.end]
                
                labels = [predicted_spans[k] for k in sorted(list(predicted_spans.keys()))]
                
                # Create detections
                for i, (bbox, prob, mask) in enumerate(zip(bboxes_scaled, probas[keep], shrinked_masks)):
                    detection = {
                        'label': obj_name,
                        'bbox': bbox.tolist(),
                        'mask': mask.astype(bool) if isinstance(mask, np.ndarray) else mask,
                        'confidence': float(prob),
                        'position_3d': None
                    }
                    detections.append(detection)
                    
            except Exception as e:
                print(f"⚠️  Error detecting {obj_name}: {e}")
                continue
        
        return detections
    
    def detect_with_depth(self, rgb_image: Image.Image, depth_image: np.ndarray,
                         object_list: List[str], camera_intrinsics: Dict) -> List[Dict]:
        """
        Detect objects with 3D position estimation using depth
        
        Args:
            rgb_image: RGB image (PIL Image)
            depth_image: Depth image array (H x W)
            object_list: List of object names to detect
            camera_intrinsics: Camera intrinsic parameters
            
        Returns:
            List of detections with 3D positions
        """
        detections_2d = self.detect_objects(rgb_image, object_list)
        
        # Estimate 3D positions from depth
        fx = camera_intrinsics.get('fx', 914.27246)
        fy = camera_intrinsics.get('fy', 913.2658)
        cx = camera_intrinsics.get('cx', 647.0733)
        cy = camera_intrinsics.get('cy', 356.32526)
        
        for detection in detections_2d:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Get depth at center (or use mask if available)
            if detection.get('mask') is not None:
                mask = detection['mask']
                # Use median depth within mask
                mask_coords = np.where(mask)
                if len(mask_coords[0]) > 0:
                    depths = depth_image[mask_coords[0], mask_coords[1]]
                    depths = depths[depths > 0]  # Filter invalid depths
                    if len(depths) > 0:
                        depth = np.median(depths)
                    else:
                        continue
                else:
                    continue
            else:
                # Fallback to center point
                if 0 <= int(center_y) < depth_image.shape[0] and 0 <= int(center_x) < depth_image.shape[1]:
                    depth = depth_image[int(center_y), int(center_x)]
                    if depth <= 0:
                        continue
                else:
                    continue
            
            # Convert to 3D coordinates
            x = (center_x - cx) * depth / fx
            y = (center_y - cy) * depth / fy
            z = depth
            
            detection['position_3d'] = (x, y, z)
        
        return detections_2d

