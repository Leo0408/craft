# ============================================================
# DETIC自定义词汇表测试Cell（可直接复制到demo2 notebook）
# ============================================================

import os
import sys
import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 保存当前目录
original_dir = os.getcwd()
detic_dir = '/home/fdse/zzy/craft/Detic'

try:
    # 切换到Detic目录
    os.chdir(detic_dir)
    sys.path.insert(0, detic_dir)
    
    # 导入必要的模块
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detic.config import add_detic_config
    from detic.predictor import VisualizationDemo
    from detic.modeling.utils import reset_cls_test
    from detic.modeling.text.text_encoder import build_text_encoder
    
    print("=" * 60)
    print("DETIC自定义词汇表测试")
    print("=" * 60)
    
    # 设置配置
    cfg = get_cfg()
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    
    # 确保FCOS yields proposals
    if hasattr(cfg.MODEL, 'FCOS'):
        cfg.MODEL.FCOS.YIELD_PROPOSAL = True
    
    # 设置自定义词汇表（您的需求）
    custom_vocab = [
        "coffee machine",
        "purple cup",
        "blue cup with handle",
        "cup",
        "table",
        "sink"
    ]
    
    print(f"\n📋 自定义词汇表: {custom_vocab}\n")
    
    # 创建metadata
    metadata = MetadataCatalog.get("__unused")
    metadata.thing_classes = custom_vocab
    
    # 创建demo
    demo = VisualizationDemo(cfg, instance_mode=1)
    
    # 设置自定义词汇表的分类器
    print("🔧 设置自定义词汇表分类器...")
    text_encoder = build_text_encoder(pretrain=True)
    text_encoder.eval()
    prompts = [f"a {name}" for name in custom_vocab]
    classifier = text_encoder(prompts).detach().permute(1, 0).contiguous().cpu()
    reset_cls_test(demo.predictor.model, classifier, len(custom_vocab))
    print("✅ 分类器设置完成\n")
    
    # 加载测试图像
    test_image_path = "test_frame.jpg"
    if not os.path.exists(test_image_path):
        # 尝试使用demo2的第一帧
        if 'frame_data' in globals() and len(frame_data) > 0:
            first_frame_idx = sorted(frame_data.keys())[0]
            first_frame = frame_data[first_frame_idx]
            image = Image.fromarray(first_frame['rgb'])
            image_np = np.array(image)
            print(f"📷 使用demo2的第一帧 (frame {first_frame_idx})")
        else:
            raise FileNotFoundError(f"图像文件不存在: {test_image_path}")
    else:
        image = Image.open(test_image_path).convert("RGB")
        image_np = np.array(image)
        print(f"📷 使用测试图像: {test_image_path}")
    
    print(f"   图像尺寸: {image.size}\n")
    
    # 运行检测
    print("🔍 开始检测...")
    predictions, visualized_output = demo.run_on_image(image_np)
    
    # 显示结果
    instances = predictions['instances']
    print(f"\n✅ 检测完成")
    print(f"检测到 {len(instances)} 个实例\n")
    
    # 显示检测结果详情
    if len(instances) > 0:
        print("📋 检测结果详情:")
        print("-" * 60)
        for i in range(min(20, len(instances))):  # 显示前20个
            class_id = instances.pred_classes[i].item()
            score = instances.scores[i].item()
            bbox = instances.pred_boxes.tensor[i].tolist()
            class_name = custom_vocab[class_id] if class_id < len(custom_vocab) else f"class_{class_id}"
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            print(f"{i+1:2d}. {class_name:25s} | 置信度: {score:.3f} | "
                  f"bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}] | "
                  f"尺寸: {width:.1f}x{height:.1f}")
        if len(instances) > 20:
            print(f"\n... 还有 {len(instances) - 20} 个检测结果")
    else:
        print("⚠️  未检测到任何物体")
        print("\n建议:")
        print("1. 降低置信度阈值 (--confidence-threshold 0.15)")
        print("2. 检查图像中是否包含要检测的物体")
        print("3. 尝试使用更通用的物体名称（如'cup'而不是'purple cup'）")
    
    # 可视化
    print("\n🖼️  显示检测结果...")
    plt.figure(figsize=(15, 10))
    # 将BGR转换为RGB
    vis_image = visualized_output.get_image()[:, :, ::-1]
    plt.imshow(vis_image)
    plt.axis('off')
    plt.title(f"DETIC自定义词汇表检测结果 - {len(instances)} 个实例", 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)

except Exception as e:
    print(f"\n❌ 错误: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 恢复原始目录
    os.chdir(original_dir)
    if detic_dir in sys.path:
        sys.path.remove(detic_dir)

