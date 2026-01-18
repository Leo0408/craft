#!/usr/bin/env python3
"""
Script to add detection methods A2, REFLECT, and enhanced modules to demo4.ipynb
"""

import json
from pathlib import Path

def create_new_cells():
    """Create new cells for demo4"""
    
    new_cells = []
    
    # Cell 1: Markdown explanation
    cell1 = {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [
            '## 📊 物体检测方法对比：A1 vs A2 vs REFLECT\n',
            '\n',
            '本部分实现三种物体检测方法，并进行对比：\n',
            '\n',
            '### 方法A1：DETIC + CLIP嵌入替换（Zero-shot Learning）⭐\n',
            '- **已实现**：在Cell 9-10中，使用CLIP文本嵌入替换DETIC分类器\n',
            '- **优点**：直接输出自定义类名，无需后处理\n',
            '- **输出**：`[\'purple cup\', \'coffee maker\', ...]`\n',
            '\n',
            '### 方法A2：DETIC + CLIP后处理匹配（两步过程）\n',
            '- **实现**：先检测LVIS类别，再用CLIP匹配到自定义描述\n',
            '- **缺点**：两步过程，可能信息丢失\n',
            '\n',
            '### REFLECT方法：MDETR + CLIP验证\n',
            '- **实现**：MDETR逐个类别检测，CLIP裁剪区域验证（阈值>0.23）\n',
            '- **优点**：可靠性高，降低误检\n',
            '\n',
            '### 📋 后续模块\n',
            '- **深度信息处理**：从分割掩码提取点云\n',
            '- **空间关系计算**：基于点云距离和边界框\n',
            '- **Gripper状态推断**：基于位置推断抓取状态\n'
        ]
    }
    new_cells.append(cell1)
    
    # Cell 2: Method A2 implementation
    cell2_source = '''# ============================================================================
# 方法A2：DETIC + CLIP后处理匹配（两步过程）
# ============================================================================
# 先使用DETIC检测LVIS类别，再用CLIP匹配到自定义描述

from PIL import Image
import numpy as np
import torch
import cv2

class DeticClipPostProcessor:
    """DETIC + CLIP后处理匹配（方案A2）"""
    
    def __init__(self, predictor, metadata):
        self.predictor = predictor
        self.metadata = metadata
        self.detic_threshold = 0.5
        self.clip_threshold = 0.3
        
        # 初始化CLIP
        try:
            import clip
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device="cpu")
            self.clip_available = True
            print("✅ CLIP loaded for post-processing matching")
        except ImportError:
            self.clip_available = False
            self.clip_model = None
            self.clip_preprocess = None
            print("⚠️  CLIP not available, using simple matching only")
    
    def _match_with_clip(self, detections, object_list, rgb_image):
        """使用CLIP匹配检测结果到原始对象描述"""
        if not self.clip_available or self.clip_model is None:
            return detections
        
        # 准备CLIP输入
        image_tensor = self.clip_preprocess(Image.fromarray(rgb_image)).unsqueeze(0).to("cpu")
        
        matched_detections = []
        for det in detections:
            detected_label = det['label']
            
            best_match = None
            best_score = 0
            
            # 尝试匹配到object_list中的每个描述
            for obj_desc in object_list:
                text_inputs = clip.tokenize([f"a photo of a {obj_desc}"]).to("cpu")
                
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(image_tensor)
                    text_features = self.clip_model.encode_text(text_inputs)
                    
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                    
                    similarity = (image_features @ text_features.T).item()
                    
                    if similarity > best_score and similarity > self.clip_threshold:
                        best_score = similarity
                        best_match = obj_desc
            
            if best_match:
                det['matched_object'] = best_match
                det['clip_score'] = best_score
                det['original_label'] = detected_label  # 保存原始LVIS标签
                matched_detections.append(det)
        
        return matched_detections
    
    def detect_objects(self, rgb_image, object_list):
        """DETIC检测 + CLIP后处理匹配"""
        # 转换为BGR格式
        if isinstance(rgb_image, Image.Image):
            rgb_array = np.array(rgb_image)
            img_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR) if len(rgb_array.shape) == 3 else rgb_array
        else:
            rgb_array = rgb_image
            img_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR) if len(rgb_array.shape) == 3 else rgb_array
        
        # Step 1: DETIC检测LVIS类别
        outputs = self.predictor(img_array)
        instances = outputs["instances"]
        valid_instances = instances[instances.scores >= self.detic_threshold]
        
        # 转换检测结果（LVIS类别）
        detections = []
        class_names = self.metadata.get("thing_classes", [])
        
        for i in range(len(valid_instances)):
            bbox = valid_instances.pred_boxes[i].tensor.cpu().numpy()[0]
            score = valid_instances.scores[i].cpu().item()
            class_id = valid_instances.pred_classes[i].cpu().item()
            
            mask = None
            if valid_instances.has("pred_masks"):
                mask = valid_instances.pred_masks[i].cpu().numpy().astype(bool)
            
            class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
            
            detections.append({
                "bbox": bbox.tolist(),
                "confidence": score,
                "label": class_name,  # LVIS类别（如 "cup", "sink"）
                "class_id": class_id,
                "mask": mask,
                "position_3d": None
            })
        
        # Step 2: CLIP匹配到自定义描述
        print(f"🔍 Method A2: Detected {len(detections)} LVIS objects, matching with CLIP...")
        matched_detections = self._match_with_clip(detections, object_list, rgb_array)
        print(f"✅ Method A2: Matched {len(matched_detections)} detections to custom descriptions")
        
        return matched_detections

# 创建方法A2的检测器
detector_a2 = DeticClipPostProcessor(predictor, metadata)
print("✅ Created Method A2 detector (DETIC + CLIP post-processing)")

'''
    cell2 = {
        'cell_type': 'code',
        'metadata': {},
        'source': cell2_source.split('\n')
    }
    new_cells.append(cell2)
    
    return new_cells

if __name__ == '__main__':
    cells = create_new_cells()
    print(f"Created {len(cells)} cells")
    for i, cell in enumerate(cells):
        print(f"  Cell {i+1}: {cell['cell_type']}")
