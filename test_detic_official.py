#!/usr/bin/env python3
"""
简单的DETIC官方示例测试
直接使用DETIC官方的demo代码来测试模型是否正常工作
"""
import sys
import os
import cv2
import numpy as np
from PIL import Image

# 添加DETIC路径
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, 'Detic')

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test

def test_detic_official():
    """使用官方代码测试DETIC"""
    print("=" * 60)
    print("DETIC官方示例测试")
    print("=" * 60)
    
    # 1. 设置配置（完全按照官方demo.py的方式）
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    
    # 使用配置文件
    config_file = "Detic/configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml"
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
        print(f"✅ 加载配置文件: {config_file}")
    else:
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    # 设置权重
    weights_file = "Detic/models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"
    if os.path.exists(weights_file):
        cfg.MODEL.WEIGHTS = weights_file
        print(f"✅ 设置权重: {weights_file}")
    else:
        print(f"❌ 权重文件不存在: {weights_file}")
        return False
    
    # 关键配置（按照官方demo.py）
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'  # 官方方式
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.DEVICE = "cpu"
    
    print("\n✅ 配置完成")
    print(f"   ZEROSHOT_WEIGHT_PATH: {cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH}")
    print(f"   USE_ZEROSHOT_CLS: {cfg.MODEL.ROI_BOX_HEAD.USE_ZEROSHOT_CLS}")
    
    # 2. 创建predictor
    print("\n正在创建DefaultPredictor...")
    try:
        predictor = DefaultPredictor(cfg)
        print("✅ DefaultPredictor创建成功")
    except Exception as e:
        print(f"❌ DefaultPredictor创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 3. 设置LVIS词汇表（官方方式）
    print("\n设置LVIS词汇表...")
    try:
        BUILDIN_CLASSIFIER = {
            'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
        }
        BUILDIN_METADATA_PATH = {
            'lvis': 'lvis_v1_val',
        }
        
        metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH['lvis'])
        classifier_path = BUILDIN_CLASSIFIER['lvis']
        classifier_full_path = os.path.join('Detic', classifier_path)
        
        if os.path.exists(classifier_full_path):
            num_classes = len(metadata.thing_classes)
            reset_cls_test(predictor.model, classifier_full_path, num_classes)
            print(f"✅ 设置LVIS词汇表成功 ({num_classes} 类)")
        else:
            print(f"❌ LVIS分类器文件不存在: {classifier_full_path}")
            return False
    except Exception as e:
        print(f"❌ 设置LVIS词汇表失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 测试检测
    print("\n" + "=" * 60)
    print("测试检测功能")
    print("=" * 60)
    
    # 创建一个简单的测试图像（640x480，随机颜色）
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    # 或者使用一个真实的图像文件（如果存在）
    test_image_path = None  # 可以设置一个测试图像路径
    
    if test_image_path and os.path.exists(test_image_path):
        print(f"使用测试图像: {test_image_path}")
        test_image = cv2.imread(test_image_path)
    else:
        print("使用随机生成的测试图像 (640x480)")
    
    print(f"测试图像尺寸: {test_image.shape}")
    
    # 运行检测
    print("\n运行DETIC检测...")
    try:
        outputs = predictor(test_image)
        instances = outputs["instances"]
        
        print(f"✅ 检测完成")
        print(f"   检测到 {len(instances)} 个实例")
        
        if len(instances) > 0:
            # 显示前5个检测结果
            print("\n前5个检测结果:")
            scores = instances.scores.cpu().numpy()
            classes = instances.pred_classes.cpu().numpy()
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            
            for i in range(min(5, len(instances))):
                bbox = boxes[i]
                width = bbox[2] - bbox[0]
                height = bbox[3] - bbox[1]
                print(f"   {i+1}. bbox=[{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}], "
                      f"size={width:.1f}x{height:.1f}, score={scores[i]:.3f}, class={classes[i]}")
            
            # 检查bbox是否相同
            if len(instances) > 1:
                unique_boxes = len(np.unique(boxes, axis=0))
                print(f"\n   唯一bbox数量: {unique_boxes}/{len(instances)}")
                if unique_boxes == 1:
                    print("   ⚠️  警告: 所有bbox都相同！")
                else:
                    print("   ✅ bbox各不相同")
            
            # 检查bbox大小
            widths = boxes[:, 2] - boxes[:, 0]
            heights = boxes[:, 3] - boxes[:, 1]
            min_size = min(widths.min(), heights.min())
            max_size = max(widths.max(), heights.max())
            print(f"\n   bbox尺寸范围: {min_size:.1f} - {max_size:.1f} 像素")
            
            if min_size < 1:
                print("   ⚠️  警告: 有bbox小于1像素，可能是噪声")
            if max_size < 10:
                print("   ⚠️  警告: 所有bbox都小于10像素，可能有问题")
        else:
            print("   ⚠️  没有检测到任何对象")
        
        return True
        
    except Exception as e:
        print(f"❌ 检测失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_detic_official()
    if success:
        print("\n" + "=" * 60)
        print("✅ 测试完成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 测试失败")
        print("=" * 60)
