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
    
    def _expand_object_names(self, obj_name: str) -> List[str]:
        """
        Expand object name to multiple prompt variations for better detection
        
        Args:
            obj_name: Original object name
            
        Returns:
            List of prompt variations (simpler names first for better matching)
        """
        variations = []
        obj_lower = obj_name.lower()
        
        # Strategy: Try simpler names FIRST (more likely to match MDETR training data)
        # 1. Extract core object names (remove descriptors)
        core_names = []
        
        # Handle color + object patterns (e.g., "purple cup" -> "cup")
        color_words = ['red', 'blue', 'green', 'yellow', 'purple', 'white', 'black', 'brown', 'orange', 'pink']
        for color in color_words:
            if obj_lower.startswith(color):
                remaining = obj_lower[len(color):].strip()
                if remaining:
                    core_names.append(remaining)
                    break
        
        # Handle compound names
        words = obj_name.split()
        
        # Extract core object type (last significant word)
        if "cup" in obj_lower or "mug" in obj_lower:
            core_names.extend(["cup", "mug", "coffee cup"])
        if "machine" in obj_lower:
            core_names.extend(["coffee machine", "machine", "coffee maker"])
        if "table" in obj_lower:
            core_names.extend(["table", "desk"])
        
        # Remove duplicates while preserving order
        seen = set()
        for name in core_names:
            if name.lower() not in seen:
                variations.append(name)
                seen.add(name.lower())
        
        # 2. Try original name variations
        if obj_name.lower() not in seen:
            variations.append(obj_name)
            seen.add(obj_name.lower())
        
        # Add article variations
        if not obj_name.startswith(('a ', 'an ', 'the ')):
            article_vars = [f"a {obj_name}", f"the {obj_name}"]
            for var in article_vars:
                if var.lower() not in seen:
                    variations.append(var)
                    seen.add(var.lower())
        
        # 3. Add article variations for core names
        for core in core_names[:3]:  # Only first 3 core names to avoid too many variations
            if not core.startswith(('a ', 'an ', 'the ')):
                article_vars = [f"a {core}", f"the {core}"]
                for var in article_vars:
                    if var.lower() not in seen:
                        variations.append(var)
                        seen.add(var.lower())
        
        # If no variations found, at least return original
        if not variations:
            variations = [obj_name]
        
        return variations
    
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
            # Try multiple prompt variations for better detection
            prompt_variations = self._expand_object_names(obj_name)
            print(f"  🔍 Detecting '{obj_name}' with {len(prompt_variations)} prompt variations: {prompt_variations[:3]}...")  # Show first 3
            
            best_detection = None
            best_confidence = 0.0
            best_prompt = None
            
            for prompt in prompt_variations:
                try:
                    # Prepare image
                    img = self.transform(rgb_image).unsqueeze(0).to(self.device)
                    
                    # Run MDETR with this prompt variation
                    outputs = self.model(img, [prompt])
                    
                    # Get predictions
                    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
                    max_prob = probas.max().item() if len(probas) > 0 else 0.0
                    keep = (probas > self.threshold).cpu()
                    
                    # Always print debugging info to help diagnose issues
                    if keep.sum() == 0:
                        # Try with lower threshold for this variation
                        low_threshold = max(0.1, self.threshold * 0.5)
                        keep_low = (probas > low_threshold).cpu()
                        if keep_low.sum() > 0:
                            # Use lower threshold detections but mark them
                            keep = keep_low
                            print(f"    ⚠️  Prompt '{prompt}': max_conf={max_prob:.3f}, using lowered threshold {low_threshold:.2f}, found {keep.sum()} detections ✅")
                        else:
                            # No detections even with low threshold - always print for debugging
                            print(f"    📊 Prompt '{prompt}': max_conf={max_prob:.3f} < threshold {low_threshold:.2f}, no detections ❌")
                    else:
                        # Found detections with standard threshold
                        print(f"    ✅ Prompt '{prompt}': max_conf={max_prob:.3f} >= threshold {self.threshold:.2f}, found {keep.sum()} detections")
                    
                    if keep.sum() == 0:
                        continue
                    
                    # Get the best detection from this prompt variation
                    max_conf_idx = probas[keep].argmax()
                    max_confidence = float(probas[keep][max_conf_idx])
                    
                    # Print success message (if not already printed above)
                    if keep.sum() > 0 and max_prob >= self.threshold:
                        # Already printed above, skip
                        pass
                    
                    # Only use this variation if it's better than previous ones
                    if max_confidence > best_confidence:
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
                        
                        # Store best detection (use original object name as label)
                        best_detection = {
                            'label': obj_name,  # Use original object name
                            'bbox': bboxes_scaled[max_conf_idx].tolist(),
                            'mask': shrinked_masks[max_conf_idx].astype(bool) if isinstance(shrinked_masks[max_conf_idx], np.ndarray) else shrinked_masks[max_conf_idx],
                            'confidence': max_confidence,
                            'position_3d': None,
                            'prompt_used': prompt  # Store which prompt worked
                        }
                        best_confidence = max_confidence
                        best_prompt = prompt
                
                except Exception as e:
                    # Continue to next variation, but log the error
                    print(f"    ⚠️  Error with prompt '{prompt}': {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Add best detection if found
            if best_detection is not None:
                print(f"  ✅ Best detection for '{obj_name}': confidence={best_confidence:.3f}, prompt='{best_prompt}'")
                detections.append(best_detection)
            else:
                print(f"  ❌ No detections found for '{obj_name}' with any prompt variation")
                
                # Also try to get all detections above threshold (not just best)
                # This allows detecting multiple instances
                try:
                    img = self.transform(rgb_image).unsqueeze(0).to(self.device)
                    best_prompt = best_detection['prompt_used']
                    outputs = self.model(img, [best_prompt])
                    probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
                    keep = (probas > self.threshold).cpu()
                    
                    if keep.sum() > 1:  # Multiple detections
                        bboxes_scaled = self._rescale_bboxes(
                            outputs['pred_boxes'].cpu()[0, keep], 
                            rgb_image.size
                        )
                        w, h = rgb_image.size
                        masks = F.interpolate(
                            outputs["pred_masks"], 
                            size=(h, w), 
                            mode="bilinear", 
                            align_corners=False
                        )
                        masks = masks.cpu()[0, keep].sigmoid() > 0.5
                        
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
                        
                        # Add all detections (skip first as it's already added)
                        for i in range(1, keep.sum().item()):
                            detection = {
                                'label': obj_name,
                                'bbox': bboxes_scaled[i].tolist(),
                                'mask': shrinked_masks[i].astype(bool) if isinstance(shrinked_masks[i], np.ndarray) else shrinked_masks[i],
                                'confidence': float(probas[keep][i]),
                                'position_3d': None
                            }
                            detections.append(detection)
                except:
                    pass  # If getting multiple detections fails, just use the best one
        
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

