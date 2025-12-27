"""
DETIC + CLIP Object Detector
Combines DETIC for open-vocabulary detection with CLIP for semantic filtering and prompt expansion
"""

import os
import sys
import torch
import torchvision.transforms as T
import numpy as np
from PIL import Image
from typing import List, Dict, Optional, Tuple
import cv2

# Try to import DETIC
try:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.utils.logger import setup_logger
    setup_logger()
    DETIC_AVAILABLE = True
except ImportError:
    DETIC_AVAILABLE = False
    print("⚠️  DETIC (detectron2) not available. Install with: pip install detectron2")

# Try to import CLIP
try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("⚠️  CLIP not available. Install with: pip install clip-by-openai")

# Try to import ByteTrack for tracking
try:
    from byte_tracker import BYTETracker
    BYTETRACK_AVAILABLE = True
except ImportError:
    BYTETRACK_AVAILABLE = False
    print("⚠️  ByteTrack not available. Tracking will be disabled.")


class DeticClipDetector:
    """
    DETIC + CLIP detector for real-world environments
    
    Pipeline:
    1. DETIC: Open-vocabulary object detection
    2. CLIP: Semantic filtering and prompt expansion
    3. ByteTrack: Multi-object tracking (optional)
    """
    
    def __init__(self, 
                 device: str = "cuda:0",
                 detic_threshold: float = 0.3,
                 clip_threshold: float = 0.25,
                 use_tracking: bool = True):
        """
        Initialize DETIC + CLIP detector
        
        Args:
            device: Device to run on ('cuda:0' or 'cpu')
            detic_threshold: DETIC detection confidence threshold
            clip_threshold: CLIP semantic similarity threshold
            use_tracking: Whether to use ByteTrack for tracking
        """
        # Ensure torch is imported (in case of module reload issues)
        import torch as _torch
        self.device = device if _torch.cuda.is_available() else 'cpu'
        self._detic_threshold = detic_threshold  # Use private variable
        self.clip_threshold = clip_threshold
        # Store availability flags as instance variables to avoid global scope issues
        # These are set at module level, safe to read here
        self.detic_available = DETIC_AVAILABLE
        self.clip_available = CLIP_AVAILABLE
        self.bytetrack_available = BYTETRACK_AVAILABLE
        self.use_tracking = use_tracking and BYTETRACK_AVAILABLE
        
        self.detic_model = None
        self.clip_model = None
        self.clip_preprocess = None
        self.tracker = None
        
        self._load_models()
    
    @property
    def threshold(self):
        """Get detection threshold (for compatibility with MDETRDetector interface)"""
        return self._detic_threshold
    
    @threshold.setter
    def threshold(self, value):
        """Set detection threshold and update DETIC model config if loaded"""
        self._detic_threshold = value
        # Update DETIC model's threshold if it's already loaded
        if self.detic_model is not None and hasattr(self.detic_model, 'cfg'):
            self.detic_model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = value
    
    @property
    def detic_threshold(self):
        """Get DETIC threshold"""
        return self._detic_threshold
    
    @detic_threshold.setter
    def detic_threshold(self, value):
        """Set DETIC threshold (also updates threshold property)"""
        self._detic_threshold = value
        # Update DETIC model's threshold if it's already loaded
        if self.detic_model is not None and hasattr(self.detic_model, 'cfg'):
            self.detic_model.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = value
    
    def _find_local_weights(self, detic_root: str = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Find local DETIC model weights and corresponding config file
        
        Args:
            detic_root: Path to DETIC repository root. If None, tries to find it automatically.
            
        Returns:
            Tuple of (weights_path, config_path) or (None, None) if not found
        """
        # Try to find DETIC root directory
        if detic_root is None:
            # Check common locations
            possible_roots = [
                os.path.join(os.path.dirname(__file__), "..", "Detic"),
                os.path.join(os.getcwd(), "Detic"),
                "/home/fdse/zzy/craft/Detic",  # User's specific path
            ]
            for root in possible_roots:
                root = os.path.abspath(root)
                models_dir = os.path.join(root, "models")
                if os.path.exists(models_dir):
                    detic_root = root
                    break
        
        if detic_root is None or not os.path.exists(detic_root):
            return None, None
        
        models_dir = os.path.join(detic_root, "models")
        configs_dir = os.path.join(detic_root, "configs")
        
        # Find all .pth files in models directory
        if not os.path.exists(models_dir):
            return None, None
        
        pth_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if not pth_files:
            return None, None
        
        # Try to match weights file with config file
        # Common patterns:
        # - detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth -> Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml
        # - detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth -> Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml
        # - detic_LCOCOI21k_CLIP_R50_1x.pth -> Detic_OVCOCO_CLIP_R50_1x_max-size.yaml
        
        weights_path = None
        config_path = None
        
        # Priority order: prefer R5021k_640b32, then SwinB, then R50_1x
        priority_patterns = [
            ("R5021k_640b32", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"),
            ("SwinB_896b32", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"),
            ("R50_1x", "Detic_OVCOCO_CLIP_R50_1x_max-size.yaml"),
        ]
        
        for pattern, config_name in priority_patterns:
            for pth_file in pth_files:
                if pattern in pth_file:
                    weights_path = os.path.join(models_dir, pth_file)
                    config_path = os.path.join(configs_dir, config_name)
                    if os.path.exists(config_path):
                        return weights_path, config_path
        
        # If no exact match, use the first .pth file and try to find matching config
        # Extract model name from pth file
        pth_file = pth_files[0]
        weights_path = os.path.join(models_dir, pth_file)
        
        # Try to infer config name from pth file name
        # Remove common prefixes/suffixes
        base_name = pth_file.replace("detic_", "").replace("Detic_", "").replace(".pth", "")
        
        # Try to find matching config
        if os.path.exists(configs_dir):
            config_files = [f for f in os.listdir(configs_dir) if f.endswith('.yaml')]
            for config_file in config_files:
                if base_name in config_file or config_file.replace("Detic_", "").replace(".yaml", "") in base_name:
                    config_path = os.path.join(configs_dir, config_file)
                    if os.path.exists(config_path):
                        return weights_path, config_path
        
        # If still no match, return weights path and let DETIC handle config
        return weights_path, None
    
    def _load_models(self):
        """Load DETIC and CLIP models"""
        # Use instance variables to avoid global scope issues
        # These were set in __init__ from the module-level globals
        detic_available = self.detic_available
        clip_available = self.clip_available
        bytetrack_available = self.bytetrack_available
        
        # Load DETIC
        if detic_available:
            try:
                # DETIC configuration
                # Note: DETIC requires specific setup. For production use, you may need to:
                # 1. Clone DETIC repository: https://github.com/facebookresearch/Detic
                # 2. Install dependencies
                # 3. Download model weights
                
                # Try to use DETIC if available in path
                # This is a simplified version - full DETIC setup may require more configuration
                try:
                    # Add Detic directory to path if it exists
                    # Use official DETIC import method: sys.path.insert(0, 'third_party/CenterNet2/')
                    detic_root = None
                    possible_roots = [
                        os.path.join(os.path.dirname(__file__), "..", "Detic"),
                        os.path.join(os.getcwd(), "Detic"),
                        "/home/fdse/zzy/craft/Detic",
                    ]
                    for root in possible_roots:
                        root = os.path.abspath(root)
                        if os.path.exists(root):
                            detic_root = root
                            # Add DETIC root to path
                            if root not in sys.path:
                                sys.path.insert(0, root)
                            # Add CenterNet2 using official method (relative to DETIC root)
                            centernet_path = os.path.join(root, "third_party", "CenterNet2")
                            if os.path.exists(centernet_path) and centernet_path not in sys.path:
                                sys.path.insert(0, centernet_path)
                                print(f"📁 Added CenterNet2 path: {centernet_path}")
                            break
                    
                    # Import using official DETIC method
                    # Clear cache to ensure we get the latest version (sys is already imported at top of file)
                    if 'centernet.config' in sys.modules:
                        del sys.modules['centernet.config']
                    if 'centernet' in sys.modules:
                        del sys.modules['centernet']
                    
                    # IMPORTANT: Import order matters!
                    # Do NOT import adet.modeling BEFORE detic.config
                    # This causes registration conflicts (build_mnv2_backbone already registered)
                    # Correct order: centernet.config -> detic.config -> adet.modeling
                    
                    from centernet.config import add_centernet_config
                    
                    # Import add_detic_config - handle registration conflicts gracefully
                    # In Jupyter notebooks, modules may be partially loaded causing registration conflicts
                    # We need to handle this gracefully by catching the error and using cached module
                    add_detic_config = None
                    try:
                        from detic.config import add_detic_config
                        print("✅ Loaded add_detic_config (normal import)")
                    except AssertionError as e:
                        if 'already registered' in str(e):
                            # Registration conflict - module loading was interrupted
                            # Check if detic.config was partially loaded
                            if 'detic.config' in sys.modules:
                                print(f"ℹ️  Registration conflict (modules partially loaded)")
                                try:
                                    add_detic_config = sys.modules['detic.config'].add_detic_config
                                    print("✅ Loaded add_detic_config (from cached module)")
                                except (AttributeError, KeyError) as cache_err:
                                    print(f"⚠️  Could not get add_detic_config from cache: {cache_err}")
                                    print("⚠️  This usually means the module import was interrupted")
                                    print("💡  Recommendation: Restart Jupyter kernel and re-run all cells")
                                    # Re-raise to trigger fallback to CLIP-only mode
                                    raise ImportError("detic.config import failed due to registration conflict")
                            else:
                                print(f"⚠️  Registration conflict but detic.config not in sys.modules")
                                print("💡  Recommendation: Restart Jupyter kernel and re-run all cells")
                                raise ImportError("detic.config import failed - module not loaded")
                        else:
                            # Different AssertionError, re-raise
                            raise
                    
                    if add_detic_config is None:
                        raise ImportError("Failed to load add_detic_config")
                    
                    # Import reset_cls_test (may not be needed for detection, but try to import it)
                    # If it fails due to registration conflicts, we can continue without it
                    reset_cls_test = None
                    try:
                        from detic.modeling.utils import reset_cls_test
                        print("✅ Loaded reset_cls_test")
                    except (AssertionError, ImportError, ModuleNotFoundError) as e:
                        # This function is used for zero-shot detection, may not be needed for basic detection
                        print(f"ℹ️  Could not import reset_cls_test (may not be needed): {e}")
                        # Continue without it - it's only needed for zero-shot class changes
                    
                    # Register CenterNet proposal generator (which is actually FCOS)
                    # This is needed because DETIC configs use NAME: "CenterNet" but it's not registered by default
                    # DETIC uses FCOS as the proposal generator but calls it "CenterNet" in configs
                    try:
                        from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
                        from adet.modeling.fcos import FCOS
                        
                        # Check if CenterNet is already registered
                        if "CenterNet" not in PROPOSAL_GENERATOR_REGISTRY._obj_map:
                            # Register FCOS as "CenterNet" (DETIC uses FCOS as CenterNet proposal generator)
                            # Create a wrapper class and register it, then manually add to registry
                            class CenterNet(FCOS):
                                pass
                            CenterNet.__name__ = "CenterNet"
                            PROPOSAL_GENERATOR_REGISTRY.register(CenterNet)
                            # Manually add FCOS to registry with "CenterNet" name
                            PROPOSAL_GENERATOR_REGISTRY._obj_map["CenterNet"] = FCOS
                            print("✅ Registered CenterNet proposal generator (using FCOS)")
                        else:
                            print("ℹ️  CenterNet proposal generator already registered")
                    except Exception as reg_err:
                        print(f"⚠️  Warning: Could not register CenterNet: {reg_err}")
                        import traceback
                        traceback.print_exc()
                        # Don't fail completely - maybe it's already registered or we can continue
                    
                    # Import detic.modeling to register CustomRCNN meta architecture
                    # This is needed for DETIC to work properly
                    # Handle registration conflicts - they just mean modules are already loaded
                    try:
                        from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
                        
                        # Check if CustomRCNN is already registered
                        if "CustomRCNN" not in META_ARCH_REGISTRY._obj_map:
                            try:
                                # Try to import detic.modeling (this will trigger registration)
                                import detic.modeling.meta_arch.custom_rcnn
                                print("✅ Imported detic.modeling.meta_arch.custom_rcnn (CustomRCNN registered)")
                            except AssertionError as e:
                                if 'already registered' in str(e):
                                    print(f"ℹ️  Registration conflict (ignored - modules already loaded)")
                                    # Check if CustomRCNN got registered despite the conflict
                                    if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
                                        print("✅ CustomRCNN is registered")
                                    else:
                                        # Try importing full detic.modeling package
                                        try:
                                            import detic.modeling
                                            if "CustomRCNN" in META_ARCH_REGISTRY._obj_map:
                                                print("✅ CustomRCNN registered (via full detic.modeling import)")
                                            else:
                                                print("⚠️  CustomRCNN may not be registered")
                                        except:
                                            print("⚠️  Could not import detic.modeling")
                                else:
                                    raise
                            except Exception as e:
                                print(f"⚠️  Warning: Could not import detic.modeling: {type(e).__name__}: {e}")
                                print("   DETIC may not work correctly without CustomRCNN")
                        else:
                            print("ℹ️  CustomRCNN already registered")
                    except Exception as e:
                        print(f"⚠️  Warning: Could not check/register CustomRCNN: {type(e).__name__}: {e}")
                    
                    # Try to find local weights first
                    weights_path, config_path = self._find_local_weights()
                    
                    # Get DETIC root directory for config paths
                    detic_root = None
                    if weights_path:
                        detic_root = os.path.dirname(os.path.dirname(weights_path))
                    else:
                        # Try to find DETIC root
                        possible_roots = [
                            os.path.join(os.path.dirname(__file__), "..", "Detic"),
                            os.path.join(os.getcwd(), "Detic"),
                            "/home/fdse/zzy/craft/Detic",
                        ]
                        for root in possible_roots:
                            root = os.path.abspath(root)
                            if os.path.exists(root):
                                detic_root = root
                                break
                    
                    cfg = get_cfg()
                    # Add configs in the correct order (official DETIC method)
                    # IMPORTANT: add_centernet_config must be called BEFORE merge_from_file
                    add_centernet_config(cfg)
                    add_detic_config(cfg)
                    
                    # Verify CENTERNET was added (debug check)
                    if not hasattr(cfg.MODEL, 'CENTERNET'):
                        raise RuntimeError("add_centernet_config failed to add MODEL.CENTERNET - check centernet/config/__init__.py")
                    
                    # Use local config if found, otherwise use default
                    if config_path and os.path.exists(config_path):
                        print(f"📁 Using local config: {config_path}")
                        cfg.merge_from_file(config_path)
                    elif detic_root:
                        # Try to find config in DETIC root
                        default_config = os.path.join(detic_root, "configs", "Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
                        if os.path.exists(default_config):
                            print(f"📁 Using default config: {default_config}")
                            cfg.merge_from_file(default_config)
                        else:
                            # Fallback to SwinB config
                            fallback_config = os.path.join(detic_root, "configs", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
                            if os.path.exists(fallback_config):
                                print(f"📁 Using fallback config: {fallback_config}")
                                cfg.merge_from_file(fallback_config)
                            else:
                                print("⚠️  Config file not found, using default settings")
                    else:
                        # Try relative paths (if running from Detic directory)
                        default_config = "configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"
                        if os.path.exists(default_config):
                            cfg.merge_from_file(default_config)
                        else:
                            # Fallback to SwinB config
                            cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
                    
                    # Use local weights if found, otherwise use URL
                    if weights_path and os.path.exists(weights_path):
                        print(f"📁 Using local weights: {weights_path}")
                        cfg.MODEL.WEIGHTS = weights_path
                    else:
                        # Fallback to URL (will download automatically)
                        print("⚠️  Local weights not found, will download from URL")
                        cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"
                    
                    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self._detic_threshold
                    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
                    cfg.MODEL.DEVICE = self.device
                    
                    self.detic_model = DefaultPredictor(cfg)
                    print("✅ DETIC model loaded")
                except ImportError as import_err:
                    # Fallback: Use detectron2 with custom vocabulary
                    print("⚠️  Full DETIC not available (ImportError), using detectron2 with custom setup")
                    print(f"   Import error: {import_err}")
                    print("   For best results, install DETIC: https://github.com/facebookresearch/Detic")
                    print("   Try: cd Detic && pip install -e .")
                    detic_available = False
                    self.detic_available = False
                except Exception as detic_err:
                    # Catch other exceptions during DETIC loading
                    print(f"⚠️  Failed to load DETIC model: {type(detic_err).__name__}: {detic_err}")
                    print("   This might be due to:")
                    print("   1. NumPy/PyTorch compatibility issues")
                    print("   2. Missing DETIC dependencies")
                    print("   3. DETIC not properly installed")
                    print("   Try installing DETIC: cd Detic && pip install -e .")
                    import traceback
                    traceback.print_exc()
                    detic_available = False
                    self.detic_available = False
            except ImportError as import_err:
                # Catch import errors when importing DETIC modules
                print(f"⚠️  Failed to import DETIC modules: {import_err}")
                print("   This usually means:")
                print("   1. DETIC not installed: cd Detic && pip install -e .")
                print("   2. NumPy/PyTorch compatibility issues")
                print("   3. Missing dependencies")
                detic_available = False
                self.detic_available = False
            except Exception as e:
                # Catch all other exceptions
                print(f"⚠️  Failed to load DETIC: {type(e).__name__}: {e}")
                print("   Detailed error:")
                import traceback
                traceback.print_exc()
                print("\n   You may need to:")
                print("   1. Install DETIC: cd Detic && pip install -e .")
                print("   2. Install dependencies: cd Detic && pip install -r requirements.txt")
                print("   3. Check NumPy/PyTorch compatibility")
                detic_available = False
                self.detic_available = False
        
        # Load CLIP
        if clip_available:
            try:
                self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
                self.clip_model.eval()
                print("✅ CLIP model loaded (ViT-B/32)")
            except Exception as e:
                print(f"⚠️  Failed to load CLIP: {e}")
                print("   Install with: pip install git+https://github.com/openai/CLIP.git")
                clip_available = False
                self.clip_available = False
        
        # Initialize ByteTrack
        if self.use_tracking:
            try:
                # ByteTrack initialization parameters
                self.tracker = BYTETracker(
                    track_thresh=0.5,
                    track_buffer=30,
                    match_thresh=0.8,
                    frame_rate=10
                )
                print("✅ ByteTrack tracker initialized")
            except Exception as e:
                print(f"⚠️  Failed to initialize ByteTrack: {e}")
                print("   Install with: pip install byte-track")
                self.use_tracking = False
    
    def _expand_prompts(self, object_list: List[str]) -> List[str]:
        """
        Expand object names with CLIP-style prompts
        
        Args:
            object_list: List of object names
            
        Returns:
            Expanded list of prompts
        """
        expanded = []
        for obj_name in object_list:
            # Add variations
            expanded.append(obj_name)
            expanded.append(f"a {obj_name}")
            expanded.append(f"the {obj_name}")
            expanded.append(f"{obj_name} object")
            # Add common synonyms
            if "cup" in obj_name.lower() or "mug" in obj_name.lower():
                expanded.append("mug")
                expanded.append("coffee cup")
            if "machine" in obj_name.lower():
                expanded.append("coffee maker")
                expanded.append("coffee machine")
        return list(set(expanded))  # Remove duplicates
    
    def _filter_with_clip(self, 
                         detections: List[Dict], 
                         object_list: List[str],
                         image: Image.Image) -> List[Dict]:
        """
        Filter detections using CLIP semantic similarity
        
        Args:
            detections: DETIC detections
            object_list: Target object names
            image: PIL Image
            
        Returns:
            Filtered detections with CLIP scores
        """
        if not self.clip_available or self.clip_model is None:
            return detections
        
        # Prepare image for CLIP
        image_tensor = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        # Prepare text prompts
        text_prompts = [f"a photo of {obj}" for obj in object_list]
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Get CLIP embeddings
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_tokens)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Filter detections
        filtered = []
        for det in detections:
            # Extract crop for this detection
            bbox = det['bbox']
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            
            # Crop image
            crop = image.crop((x1, y1, x2, y2))
            crop_tensor = self.clip_preprocess(crop).unsqueeze(0).to(self.device)
            
            # Get crop embedding
            with torch.no_grad():
                crop_features = self.clip_model.encode_image(crop_tensor)
                crop_features = crop_features / crop_features.norm(dim=-1, keepdim=True)
                
                # Compute similarity with all text prompts
                similarities = (crop_features @ text_features.T).cpu().numpy()[0]
                max_sim = similarities.max()
                best_match_idx = similarities.argmax()
            
            # Filter by CLIP threshold
            if max_sim >= self.clip_threshold:
                det['clip_score'] = float(max_sim)
                det['clip_matched_object'] = object_list[best_match_idx]
                filtered.append(det)
        
        return filtered
    
    def _expand_clip_prompts(self, object_list: List[str]) -> List[str]:
        """
        Expand object names to multiple CLIP prompt variations for better detection
        
        Args:
            object_list: List of object names
            
        Returns:
            List of expanded prompts (includes all variations)
        """
        expanded_prompts = []
        for obj_name in object_list:
            # Multiple prompt formats
            expanded_prompts.append(f"a photo of {obj_name}")
            expanded_prompts.append(f"a {obj_name}")
            expanded_prompts.append(f"the {obj_name}")
            expanded_prompts.append(f"a picture of {obj_name}")
            expanded_prompts.append(obj_name)  # Original name
            
            # Handle compound names
            obj_lower = obj_name.lower()
            if "cup" in obj_lower or "mug" in obj_lower:
                expanded_prompts.append("a cup")
                expanded_prompts.append("a mug")
                expanded_prompts.append("a coffee cup")
            if "machine" in obj_lower:
                expanded_prompts.append("a coffee machine")
                expanded_prompts.append("a coffee maker")
            if "table" in obj_lower:
                expanded_prompts.append("a table")
        
        return expanded_prompts
    
    def _detect_with_clip_only(self, rgb_image: Image.Image, object_list: List[str]) -> List[Dict]:
        """
        Fallback detection using CLIP only (when DETIC is not available)
        Uses CLIP to find objects by comparing image regions with text prompts
        
        Improved version with better prompt expansion and sliding window approach
        """
        if not self.clip_available or self.clip_model is None:
            print("⚠️  CLIP not available for fallback detection")
            return []
        
        # Expand prompts for better matching
        expanded_prompts = self._expand_clip_prompts(object_list)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_prompts = []
        for p in expanded_prompts:
            if p.lower() not in seen:
                seen.add(p.lower())
                unique_prompts.append(p)
        
        # Simple grid-based detection using CLIP
        # Divide image into regions and check similarity with object names
        img_array = np.array(rgb_image)
        h, w = img_array.shape[:2]
        
        # Use a finer grid for better detection (increase from 7x7 to 10x10)
        grid_size = 10  # 10x10 grid for better resolution
        cell_h, cell_w = h // grid_size, w // grid_size
        
        detections = []
        # Use expanded prompts
        text_prompts = unique_prompts
        text_tokens = clip.tokenize(text_prompts).to(self.device)
        
        # Get text embeddings for all prompts
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Map prompt indices back to original object names
        prompt_to_obj = {}
        for i, prompt in enumerate(unique_prompts):
            # Find which original object this prompt belongs to
            for obj_idx, obj_name in enumerate(object_list):
                if obj_name.lower() in prompt.lower() or prompt.lower() in obj_name.lower():
                    prompt_to_obj[i] = obj_idx
                    break
            if i not in prompt_to_obj:
                # Default to first object if no match
                prompt_to_obj[i] = 0
        
        # Check each grid cell with sliding window overlap
        step_h, step_w = cell_h // 2, cell_w // 2  # 50% overlap for better coverage
        
        for i in range(0, grid_size * 2 - 1):  # More cells due to overlap
            for j in range(0, grid_size * 2 - 1):
                y1 = min(i * step_h, h - cell_h)
                y2 = min(y1 + cell_h, h)
                x1 = min(j * step_w, w - cell_w)
                x2 = min(x1 + cell_w, w)
                
                # Skip if region is too small
                if (x2 - x1) < 50 or (y2 - y1) < 50:
                    continue
                
                # Extract region
                region = rgb_image.crop((x1, y1, x2, y2))
                region_tensor = self.clip_preprocess(region).unsqueeze(0).to(self.device)
                
                # Get region embedding
                with torch.no_grad():
                    region_features = self.clip_model.encode_image(region_tensor)
                    region_features = region_features / region_features.norm(dim=-1, keepdim=True)
                    
                    # Compute similarity with all prompts
                    similarities = (region_features @ text_features.T).cpu().numpy()[0]
                    max_sim = similarities.max()
                    best_prompt_idx = similarities.argmax()
                    
                    # Map prompt index back to original object
                    obj_idx = prompt_to_obj.get(best_prompt_idx, 0)
                    obj_name = object_list[obj_idx]
                    
                    # Use a slightly lower threshold for CLIP-only mode to improve recall
                    clip_threshold_adjusted = max(0.2, self.clip_threshold * 0.8)
                    
                    # If similarity is high enough, create detection
                    if max_sim >= clip_threshold_adjusted:
                        detection = {
                            'label': obj_name,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'mask': None,
                            'confidence': float(max_sim),
                            'position_3d': None,
                            'class_id': obj_idx,
                            'clip_score': float(max_sim)
                        }
                        detections.append(detection)
        
        # Merge nearby detections of the same class
        if len(detections) > 0:
            merged = []
            used = set()
            for i, det1 in enumerate(detections):
                if i in used:
                    continue
                group = [det1]
                for j, det2 in enumerate(detections[i+1:], i+1):
                    if j in used:
                        continue
                    if det1['label'] == det2['label']:
                        # Check if nearby (IoU > 0.3)
                        bbox1 = det1['bbox']
                        bbox2 = det2['bbox']
                        x1 = max(bbox1[0], bbox2[0])
                        y1 = max(bbox1[1], bbox2[1])
                        x2 = min(bbox1[2], bbox2[2])
                        y2 = min(bbox1[3], bbox2[3])
                        if x2 > x1 and y2 > y1:
                            intersection = (x2 - x1) * (y2 - y1)
                            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
                            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
                            union = area1 + area2 - intersection
                            iou = intersection / union if union > 0 else 0
                            if iou > 0.3:
                                group.append(det2)
                                used.add(j)
                
                # Merge group: use highest confidence, expand bbox
                if len(group) > 1:
                    best = max(group, key=lambda x: x['confidence'])
                    x1 = min(d['bbox'][0] for d in group)
                    y1 = min(d['bbox'][1] for d in group)
                    x2 = max(d['bbox'][2] for d in group)
                    y2 = max(d['bbox'][3] for d in group)
                    best['bbox'] = [x1, y1, x2, y2]
                    merged.append(best)
                else:
                    merged.append(group[0])
                used.add(i)
            
            detections = merged
        
        return detections
    
    def detect_objects(self, rgb_image: Image.Image, object_list: List[str]) -> List[Dict]:
        """
        Detect objects using DETIC + CLIP
        
        Args:
            rgb_image: RGB image (PIL Image)
            object_list: List of object names to detect
            
        Returns:
            List of detections with bbox, mask, confidence, label
        """
        # Check availability using instance variable
        if not hasattr(self, 'detic_available') or not self.detic_available or self.detic_model is None:
            # Fallback to CLIP-only detection if CLIP is available
            if self.clip_available and self.clip_model is not None:
                print("⚠️  DETIC not available, using CLIP-only fallback detection")
                return self._detect_with_clip_only(rgb_image, object_list)
            else:
                print("⚠️  DETIC not available, returning empty detections")
                return []
        
        # Convert PIL to numpy for DETIC
        img_array = np.array(rgb_image)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            # RGB to BGR for DETIC
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Run DETIC
        outputs = self.detic_model(img_array)
        
        # Parse DETIC outputs
        instances = outputs["instances"]
        detections = []
        
        for i in range(len(instances)):
            bbox = instances.pred_boxes[i].tensor.cpu().numpy()[0]  # [x1, y1, x2, y2]
            score = instances.scores[i].cpu().item()
            class_id = instances.pred_classes[i].cpu().item()
            
            # Get mask if available
            mask = None
            if instances.has("pred_masks"):
                mask = instances.pred_masks[i].cpu().numpy().astype(bool)
            
            # Get class name (if available)
            if hasattr(instances, 'pred_class_names'):
                class_name = instances.pred_class_names[i]
            else:
                class_name = f"class_{class_id}"
            
            detection = {
                'label': class_name,
                'bbox': bbox.tolist(),
                'mask': mask,
                'confidence': score,
                'position_3d': None,
                'class_id': class_id
            }
            detections.append(detection)
        
        # Filter with CLIP
        if self.clip_available and self.clip_model is not None:
            detections = self._filter_with_clip(detections, object_list, rgb_image)
        
        # Filter by object_list (if CLIP not available, use simple matching)
        if not self.clip_available:
            # Simple name matching
            filtered = []
            for det in detections:
                det_label_lower = det['label'].lower()
                for obj_name in object_list:
                    if obj_name.lower() in det_label_lower or det_label_lower in obj_name.lower():
                        det['matched_object'] = obj_name
                        filtered.append(det)
                        break
            detections = filtered
        
        return detections
    
    def detect_with_depth(self, 
                         rgb_image: Image.Image, 
                         depth_image: np.ndarray,
                         object_list: List[str], 
                         camera_intrinsics: Dict) -> List[Dict]:
        """
        Detect objects with 3D position estimation using depth
        
        Args:
            rgb_image: RGB image (PIL Image)
            depth_image: Depth image array
            object_list: List of object names to detect
            camera_intrinsics: Camera intrinsic parameters
            
        Returns:
            List of detections with 3D positions
        """
        detections_2d = self.detect_objects(rgb_image, object_list)
        
        # Estimate 3D positions from depth
        fx = camera_intrinsics.get('fx', 914.27)
        fy = camera_intrinsics.get('fy', 913.27)
        cx = camera_intrinsics.get('cx', 647.07)
        cy = camera_intrinsics.get('cy', 356.33)
        
        for detection in detections_2d:
            bbox = detection['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Get depth at center (or use mask if available)
            if detection.get('mask') is not None:
                mask = detection['mask']
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

