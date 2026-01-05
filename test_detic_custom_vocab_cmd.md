# DETIC命令行测试自定义词汇表

## 问题1：如何在命令行添加自定义object list？

DETIC demo支持通过`--vocabulary custom`和`--custom_vocabulary`参数来指定自定义词汇表。

### 正确的命令格式

```bash
cd /home/fdse/zzy/craft/Detic

python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_custom_test.jpg \
    --vocabulary custom \
    --custom_vocabulary "coffee machine,purple cup,blue cup with handle,table,sink" \
    --cpu \
    --confidence-threshold 0.2 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth
```

**注意**：
- 参数名是`--custom_vocabulary`（下划线），不是`--custom-vocabulary`（连字符）
- 物体名称用**逗号分隔**，不要有空格
- 使用`--vocabulary custom`来启用自定义词汇表模式

### 完整的测试命令（包含您的需求）

```bash
cd /home/fdse/zzy/craft/Detic

python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input test_frame.jpg \
    --output out_my_objects.jpg \
    --vocabulary custom \
    --custom_vocabulary "coffee machine,purple cup,blue cup with handle,cup,table,sink" \
    --cpu \
    --confidence-threshold 0.2 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth
```

## 问题2：如果命令行不行，Notebook Cell方式

如果命令行测试不方便，可以在notebook中创建一个测试cell：

```python
# ============================================================
# DETIC自定义词汇表测试Cell（在demo2 notebook中）
# ============================================================

import os
import sys
import torch

# 切换到Detic目录
detic_dir = '/home/fdse/zzy/craft/Detic'
original_dir = os.getcwd()

try:
    os.chdir(detic_dir)
    sys.path.insert(0, detic_dir)
    
    # 导入必要的模块
    from detectron2.config import get_cfg
    from detectron2.data import MetadataCatalog
    from detic.config import add_detic_config
    from detic.predictor import VisualizationDemo
    
    # 设置配置
    cfg = get_cfg()
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = "models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = "cpu"
    
    # 设置自定义词汇表
    custom_vocab = [
        "coffee machine",
        "purple cup",
        "blue cup with handle",
        "cup",
        "table",
        "sink"
    ]
    
    # 创建metadata
    metadata = MetadataCatalog.get("__unused")
    metadata.thing_classes = custom_vocab
    
    # 创建demo
    demo = VisualizationDemo(cfg, instance_mode=1)
    
    # 加载测试图像
    from PIL import Image
    import numpy as np
    
    # 使用test_frame.jpg或demo2的第一帧
    test_image_path = "test_frame.jpg"
    if os.path.exists(test_image_path):
        image = Image.open(test_image_path).convert("RGB")
        image_np = np.array(image)
        
        # 运行检测
        print(f"🔍 开始检测，自定义词汇表: {custom_vocab}")
        predictions, visualized_output = demo.run_on_image(image_np)
        
        # 显示结果
        print(f"\n✅ 检测完成")
        print(f"检测到 {len(predictions['instances'])} 个实例\n")
        
        # 显示检测结果
        instances = predictions['instances']
        for i in range(min(10, len(instances))):
            class_id = instances.pred_classes[i].item()
            score = instances.scores[i].item()
            class_name = custom_vocab[class_id] if class_id < len(custom_vocab) else f"class_{class_id}"
            print(f"{i+1}. {class_name}: {score:.3f}")
        
        # 可视化
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        plt.imshow(visualized_output.get_image()[:, :, ::-1])  # BGR to RGB
        plt.axis('off')
        plt.title(f"DETIC自定义词汇表检测结果 - {len(instances)} 个实例", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
    else:
        print(f"❌ 图像文件不存在: {test_image_path}")

finally:
    os.chdir(original_dir)
    if detic_dir in sys.path:
        sys.path.remove(detic_dir)
```

## 问题3：检测到handle和knob是否能说明DETIC模型和配置没问题？

**是的，可以说明！** ✅

### 为什么检测到handle和knob说明模型正常？

1. **模型能够检测物体**：
   - DETIC成功运行，没有报错
   - 输出了检测结果（300个实例）
   - 有合理的置信度分数（0.4-0.5）

2. **配置正确**：
   - 模型权重加载成功
   - 配置合并成功
   - 推理流程正常

3. **LVIS词汇表工作正常**：
   - handle和knob都是LVIS词汇表中的有效类别
   - 说明LVIS词汇表加载和分类器工作正常

### 为什么检测到的是handle/knob而不是coffee machine？

**不是模型问题，而是词汇表问题**：

1. **LVIS词汇表的特性**：
   - LVIS包含1203个类别
   - 包含很多物体的组成部分（handle, bolt, knob等）
   - 这些部分在图像中特征明显，容易检测

2. **特定物体可能不在词汇表中**：
   - "coffee machine"可能在LVIS中不存在
   - 或者使用不同的名称（如"coffee maker"）

3. **检测优先级**：
   - DETIC倾向于检测有明显特征的组成部分
   - 这些部分通常有清晰轮廓和高对比度

### 结论

✅ **DETIC模型和配置是正确的**  
✅ **模型能够正常检测物体**  
✅ **问题在于使用的词汇表（LVIS）不包含您要检测的特定物体**  
✅ **解决方案：使用自定义词汇表**（`--vocabulary custom`）

## 推荐的测试流程

### 步骤1：使用LVIS词汇表测试（确认模型正常）✅ 已完成

```bash
python demo.py --vocabulary lvis --confidence-threshold 0.3 ...
# 结果：检测到handle, knob等 → 说明模型正常
```

### 步骤2：使用自定义词汇表测试（检测您要的物体）

```bash
python demo.py \
    --vocabulary custom \
    --custom_vocabulary "coffee machine,purple cup,blue cup with handle,table,sink" \
    --confidence-threshold 0.2 \
    ...
```

### 步骤3：在代码中使用（demo2 notebook）

使用`DeticClipDetector`，它已经集成了自定义词汇表和CLIP过滤。

