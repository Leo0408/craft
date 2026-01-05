# DETIC 调试测试代码

## 问题诊断

所有DETIC检测的bbox和score完全相同，只有class_id不同。这是一个严重的问题，可能是：
1. 自定义词汇表设置导致的问题
2. DETIC模型配置问题
3. NMS（非极大值抑制）未正常工作

## 测试步骤

### 1. 测试不使用自定义词汇表（使用默认LVIS词汇表）

在notebook中运行以下代码：

```python
# 重新加载模块（如果需要）
import importlib
import sys
if 'perception.detic_clip_detector' in sys.modules:
    importlib.reload(sys.modules['perception.detic_clip_detector'])

from perception.detic_clip_detector import DeticClipDetector

# 确保detector已初始化（如果还没有）
# detector = DeticClipDetector(detic_threshold=0.3, clip_threshold=0.25, enable_tracking=False)

# 测试1: 不使用自定义词汇表（使用默认DETIC词汇表）
print("=" * 60)
print("测试1: 不使用自定义词汇表（使用默认LVIS词汇表）")
print("=" * 60)

detections_no_vocab = detector.detect_objects(
    rgb_pil, 
    object_list, 
    use_custom_vocab=False,  # 禁用自定义词汇表
    debug_mode=True          # 启用详细调试输出
)

print(f"\n✅ 测试1完成: 找到 {len(detections_no_vocab)} 个检测")
print(f"   前5个检测的bbox是否相同: 检查输出中的'Unique bboxes'字段")

# 如果测试1的bbox正常（有多个不同的bbox），说明问题在自定义词汇表设置
# 如果测试1的bbox也不正常，说明问题在DETIC模型配置或模型本身
```

### 2. 测试使用自定义词汇表（对比）

```python
print("\n" + "=" * 60)
print("测试2: 使用自定义词汇表")
print("=" * 60)

detections_with_vocab = detector.detect_objects(
    rgb_pil, 
    object_list, 
    use_custom_vocab=True,   # 启用自定义词汇表
    debug_mode=True          # 启用详细调试输出
)

print(f"\n✅ 测试2完成: 找到 {len(detections_with_vocab)} 个检测")
print(f"   对比测试1和测试2的输出，查看哪个有问题")
```

### 3. 查看模型配置

调试输出会显示：
- `TEST.DETECTIONS_PER_IMAGE`: 每张图像的最大检测数
- `MODEL.ROI_HEADS.SCORE_THRESH_TEST`: 置信度阈值
- `MODEL.ROI_HEADS.NMS_THRESH_TEST`: NMS阈值
- `MODEL.ROI_BOX_HEAD.NUM_CLASSES`: 类别数量

检查这些配置是否合理。

### 4. 检查DETIC原始输出

调试输出会显示：
- 所有bbox是否在tensor中就已经相同（`Unique bboxes`）
- 所有score是否相同（`Unique scores`）

如果`Unique bboxes: 1`，说明问题在DETIC模型本身或配置。
如果`Unique bboxes > 1`但处理后相同，说明问题在我们的处理流程。

## 预期结果

### 正常情况：
- `Unique bboxes`应该 > 1（有很多不同的bbox）
- `Unique scores`应该 > 1（有很多不同的score）
- 检测框应该分布在图像的不同位置

### 异常情况（当前）：
- `Unique bboxes: 1`（所有bbox相同）
- `Unique scores: 1`（所有score相同）
- 所有检测框都在同一个位置

## 可能的解决方案

1. **如果不使用自定义词汇表时正常**：
   - 问题在`reset_cls_test`的实现
   - 可能需要检查CLIP embeddings的生成方式
   - 可能需要调整`safe_reset_cls_test`的参数

2. **如果两种方式都不正常**：
   - 问题在DETIC模型配置
   - 可能需要检查模型权重是否正确加载
   - 可能需要调整NMS阈值或其他配置

3. **如果都不正常，尝试**：
   - 重启Jupyter kernel
   - 重新初始化detector
   - 检查DETIC模型权重文件是否完整
   - 尝试使用DETIC官方的predict.py脚本测试模型是否正常工作

