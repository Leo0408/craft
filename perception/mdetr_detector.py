"""
MDETR Object Detector Wrapper
Wraps REFLECT's MDETR detector for use in CRAFT framework
"""

import os
import sys

# Configure Hugging Face mirror for faster downloads (especially in China)
# This must be set BEFORE importing transformers
# Set multiple environment variables for better compatibility
if 'HF_ENDPOINT' not in os.environ:
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
if 'HF_HUB_DOWNLOAD_TIMEOUT' not in os.environ:
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'  # 5 minutes
# Also set for huggingface_hub if it uses different env var
if 'HUGGINGFACE_HUB_CACHE' not in os.environ:
    # Use mirror endpoint
    pass

import torch
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from PIL import Image
from typing import List, Dict, Optional
from collections import defaultdict
import cv2

# Add REFLECT real-world directory to path if available
# Priority: real-world > mdetr (real-world is adapted for REFLECT tasks)
REFLECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'reflect')
if os.path.exists(REFLECT_ROOT):
    # Try real-world first (REFLECT's adapted version)
    real_world_path = os.path.join(REFLECT_ROOT, 'real-world')
    mdetr_path = os.path.join(REFLECT_ROOT, 'mdetr')
    
    if os.path.exists(real_world_path):
        if real_world_path not in sys.path:
            sys.path.insert(0, real_world_path)
        print(f"✓ Using REFLECT real-world MDETR path: {real_world_path}")
    elif os.path.exists(mdetr_path):
        if mdetr_path not in sys.path:
            sys.path.insert(0, mdetr_path)
        print(f"✓ Using original MDETR path: {mdetr_path}")


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
        """Load MDETR model - tries both real-world and mdetr directories"""
        # Try multiple paths: real-world first (REFLECT adapted), then mdetr (original)
        # Calculate REFLECT_ROOT from current file location
        current_file = os.path.abspath(__file__)
        # From /home/leo/craft/perception/mdetr_detector.py
        # Go up 3 levels: perception -> craft -> parent -> reflect
        craft_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
        reflect_root_calculated = os.path.join(craft_root, 'reflect')
        reflect_root_calculated = os.path.abspath(reflect_root_calculated)  # Resolve any .. in path
        
        # Also try common locations
        reflect_roots_to_try = [
            reflect_root_calculated,
            os.path.join(os.path.expanduser('~'), 'reflect'),
            '/home/leo/reflect',  # Explicit path as fallback
        ]
        
        # Find the first existing REFLECT root
        REFLECT_ROOT = None
        for root in reflect_roots_to_try:
            root_abs = os.path.abspath(root)
            if os.path.exists(root_abs):
                REFLECT_ROOT = root_abs
                break
        
        if REFLECT_ROOT is None:
            print(f"⚠️  Could not find REFLECT root directory")
            print(f"   Tried: {[os.path.abspath(r) for r in reflect_roots_to_try]}")
            self.model = None
            return
        
        paths_to_try = [
            os.path.join(REFLECT_ROOT, 'real-world'),  # Preferred: REFLECT's adapted version
            os.path.join(REFLECT_ROOT, 'mdetr')        # Fallback: original MDETR
        ]
        paths_to_try = [os.path.abspath(p) for p in paths_to_try]  # Resolve paths
        
        last_error = None
        for path in paths_to_try:
            if not os.path.exists(path):
                continue
            
            try:
                # Add to path if not already there
                if path not in sys.path:
                    sys.path.insert(0, path)
                
                # Ensure Hugging Face mirror is set before importing hubconf
                # This is critical because hubconf imports models that load tokenizers
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '300'
                
                # Force set huggingface_hub endpoint using API (more reliable than env vars)
                try:
                    import huggingface_hub
                    # Try multiple methods to set endpoint
                    if hasattr(huggingface_hub, 'set_endpoint'):
                        huggingface_hub.set_endpoint('https://hf-mirror.com')
                    elif hasattr(huggingface_hub, 'constants'):
                        # Some versions store endpoint in constants
                        huggingface_hub.constants.ENDPOINT = 'https://hf-mirror.com'
                    # Also try to patch file_utils if transformers uses it
                    try:
                        import transformers
                        if hasattr(transformers, 'file_utils'):
                            transformers.file_utils.HUGGINGFACE_CO_URL_HOME = 'https://hf-mirror.com'
                    except:
                        pass
                except Exception as e:
                    print(f"⚠️  Could not set huggingface_hub endpoint via API: {e}")
                    print("   Relying on environment variables only")
                
                # Try to import from this path
                from hubconf import mdetr_efficientnetB3_phrasecut
                self.model = mdetr_efficientnetB3_phrasecut(pretrained=pretrained).to(self.device)
                self.model.eval()
                torch.set_grad_enabled(False)
                print(f"✓ MDETR detector loaded from {os.path.basename(path)} on {self.device}")
                return  # Success!
            except ImportError as e:
                last_error = e
                # Remove from path if import failed
                if path in sys.path:
                    sys.path.remove(path)
                continue
            except Exception as e:
                last_error = e
                print(f"⚠️  Error loading MDETR from {path}: {e}")
                continue
        
        # If we get here, all paths failed
        print("⚠️  Warning: Could not import MDETR from any REFLECT path")
        print(f"   REFLECT_ROOT: {REFLECT_ROOT}")
        if last_error:
            print(f"   Last error: {last_error}")
        print("   Possible reasons:")
        print("   1. REFLECT real-world or mdetr directory not found")
        print("   2. Missing dependencies (timm, transformers)")
        print("   3. hubconf.py not found or cannot be imported")
        print("   Tried paths:")
        for path in paths_to_try:
            exists = "✅" if os.path.exists(path) else "❌"
            hubconf_path = os.path.join(path, "hubconf.py")
            hubconf_exists = "✅" if os.path.exists(hubconf_path) else "❌"
            print(f"     {exists} {path}")
            print(f"        hubconf.py: {hubconf_exists}")
        if last_error:
            import traceback
            traceback.print_exc()
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

