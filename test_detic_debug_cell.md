# DETIC调试测试代码（Jupyter Notebook版本）

## 问题诊断

所有DETIC检测的bbox和score完全相同，只有class_id不同。

## 测试步骤（在Notebook中运行）

### 步骤1: 重新加载模块

```python
# 重新加载模块以使用最新的代码
import importlib
import sys

# 清除模块缓存
if 'perception.detic_clip_detector' in sys.modules:
    del sys.modules['perception.detic_clip_detector']
if 'craft.perception.detic_clip_detector' in sys.modules:
    del sys.modules['craft.perception.detic_clip_detector']

# 重新导入
from craft.perception.detic_clip_detector import DeticClipDetector

print("✅ 模块已重新加载")
```

### 步骤2: 重新初始化detector（如果已经存在）

```python
# 重新初始化detector以使用最新版本的类
detector = DeticClipDetector(
    detic_threshold=0.3,
    clip_threshold=0.25,
    enable_tracking=False
)

print("✅ Detector已重新初始化")
```

### 步骤3: 测试不使用自定义词汇表

```python
print("=" * 60)
print("测试1: 不使用自定义词汇表（使用默认LVIS词汇表）")
print("=" * 60)

# 确保rgb_pil和object_list已定义
# rgb_pil = Image.fromarray(first_frame['rgb'])  # 如果需要
# object_list = ["coffee machine", "purple cup", "blue cup with handle", "table on the left of sink"]

detections_no_vocab = detector.detect_objects(
    rgb_pil, 
    object_list, 
    use_custom_vocab=False,  # 禁用自定义词汇表
    debug_mode=True          # 启用详细调试输出
)

print(f"\n✅ 测试1完成: 找到 {len(detections_no_vocab)} 个检测")
```

### 步骤4: 测试使用自定义词汇表（对比）

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
```

## 关键检查点

### 调试输出中的重要信息：

1. **模型配置**：
   - `TEST.DETECTIONS_PER_IMAGE`: 每张图像的最大检测数
   - `MODEL.ROI_HEADS.SCORE_THRESH_TEST`: 置信度阈值
   - `MODEL.ROI_HEADS.NMS_THRESH_TEST`: NMS阈值

2. **输出结构检查**（debug_mode=True时）：
   - `Unique bboxes`: 如果 = 1，说明所有bbox都相同（问题所在）
   - `Unique scores`: 如果 = 1，说明所有score都相同

3. **原始检测样本**：
   - 查看前10个检测的bbox是否不同
   - 查看bbox尺寸是否合理

## 预期结果

### 正常情况：
- `Unique bboxes` > 1（有很多不同的bbox）
- `Unique scores` > 1（有很多不同的score）
- 检测框分布在图像的不同位置

### 异常情况（当前）：
- `Unique bboxes: 1`（所有bbox相同）❌
- `Unique scores: 1`（所有score相同）❌

## 可能的原因

1. **如果测试1（不使用自定义词汇表）正常**：
   - 问题在`reset_cls_test`的实现
   - 需要检查自定义词汇表的设置方式

2. **如果测试1也不正常**：
   - 问题在DETIC模型配置
   - 可能需要检查模型权重或配置参数
