# demo2中修改object_list的完整指南

## 您的需求

检测以下物体：
- `'coffee machine'` - 咖啡机
- `'purple cup'` - 紫色杯子
- `'blue cup with handle'` - 带把手的蓝色杯子
- `'table on the left of sink'` - 水槽左边的桌子

## 在demo2中的修改方法

### 方法1：直接使用完整描述（推荐）✅

`DeticClipDetector`已经支持使用完整的描述（包括颜色、属性），CLIP会自动过滤匹配。您只需要修改`object_list`：

```python
# 在demo2 notebook中找到定义object_list的地方，修改为：

object_list = [
    "coffee machine",           # 咖啡机
    "purple cup",               # 紫色杯子（CLIP会匹配颜色）
    "blue cup with handle",     # 带把手的蓝色杯子（CLIP会匹配颜色和属性）
    "cup",                      # 普通杯子（作为备选）
    "table",                    # 桌子
    "sink",                     # 水槽
]

# 然后正常调用检测
detections = detector.detect_objects(rgb_pil, object_list)
```

**注意**：对于`"table on the left of sink"`这种空间关系，需要在检测后进行后处理（见下文）。

### 方法2：检测后处理空间关系

如果您需要检测"table on the left of sink"这种空间关系，可以添加后处理代码：

```python
# 1. 先进行基础检测
object_list = [
    "coffee machine",
    "purple cup",
    "blue cup with handle",
    "cup",
    "table",
    "sink",
]

detections = detector.detect_objects(rgb_pil, object_list)

# 2. 处理空间关系："table on the left of sink"
def find_table_left_of_sink(detections):
    """查找sink左边的table"""
    sinks = [d for d in detections if 'sink' in d.get('label', '').lower()]
    tables = [d for d in detections if 'table' in d.get('label', '').lower()]
    
    results = []
    for table in tables:
        for sink in sinks:
            # 计算中心点x坐标
            table_center_x = (table['bbox'][0] + table['bbox'][2]) / 2
            sink_center_x = (sink['bbox'][0] + sink['bbox'][2]) / 2
            
            # 判断是否在左边（table的x坐标小于sink）
            if table_center_x < sink_center_x:
                results.append({
                    'object': table,
                    'label': 'table on the left of sink',
                    'reference': sink
                })
                print(f"✅ 找到: table在sink的左边")
    
    return results

# 查找空间关系
spatial_detections = find_table_left_of_sink(detections)

# 如果需要，可以将空间关系的检测结果添加到detections中
for spatial_det in spatial_detections:
    # 可以添加一个特殊的标签来表示空间关系
    spatial_det['object']['spatial_label'] = 'table on the left of sink'
```

## 完整代码示例（可直接复制到notebook）

```python
# ============================================================
# demo2中检测自定义物体（带颜色和属性）
# ============================================================

from perception.detic_clip_detector import DeticClipDetector
from PIL import Image

# 创建或获取检测器
if 'detector' not in locals() or detector is None:
    detector = DeticClipDetector(
        detic_threshold=0.3,
        clip_threshold=0.20,  # 稍低的阈值，以便匹配颜色和属性描述
        use_tracking=False
    )

# 定义要检测的物体（使用完整描述，包括颜色和属性）
object_list = [
    "coffee machine",           # 咖啡机
    "purple cup",               # 紫色杯子
    "blue cup with handle",     # 带把手的蓝色杯子
    "cup",                      # 普通杯子（作为备选）
    "table",                    # 桌子
    "sink",                     # 水槽
]

# 使用第一帧进行检测
if 'frame_data' in globals() and len(frame_data) > 0:
    first_frame_idx = sorted(frame_data.keys())[0]
    first_frame = frame_data[first_frame_idx]
    rgb_pil = Image.fromarray(first_frame['rgb'])
    
    # 进行检测
    print("🔍 开始检测...")
    detections = detector.detect_objects(rgb_pil, object_list)
    
    print(f"\n✅ 检测完成: 找到 {len(detections)} 个对象\n")
    
    # 显示检测结果
    for i, det in enumerate(detections, 1):
        label = det.get('label', 'unknown')
        bbox = det.get('bbox', [])
        confidence = det.get('confidence', 0)
        print(f"{i}. {label} (置信度: {confidence:.3f})")
        print(f"   边界框: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print()
    
    # 处理空间关系："table on the left of sink"
    sinks = [d for d in detections if 'sink' in d.get('label', '').lower()]
    tables = [d for d in detections if 'table' in d.get('label', '').lower()]
    
    print("🔍 查找空间关系: table on the left of sink")
    for table in tables:
        for sink in sinks:
            table_center_x = (table['bbox'][0] + table['bbox'][2]) / 2
            sink_center_x = (sink['bbox'][0] + sink['bbox'][2]) / 2
            if table_center_x < sink_center_x:
                print(f"✅ 找到: table在sink的左边")
                print(f"   Table: {table.get('label')} at x={table_center_x:.1f}")
                print(f"   Sink: {sink.get('label')} at x={sink_center_x:.1f}")
    
    # 可视化（如果detector有visualize_detections方法）
    if hasattr(detector, 'visualize_detections'):
        vis_image = detector.visualize_detections(
            rgb_pil,
            detections,
            title=f"DETIC检测结果 - {len(detections)} 个对象"
        )
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15, 10))
        plt.imshow(vis_image)
        plt.axis('off')
        plt.title(f"检测结果: {len(detections)} 个对象", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
```

## 工作原理

### DETIC + CLIP的工作流程

1. **DETIC检测**：使用基础物体名称（cup, table, sink等）进行初步检测
2. **自定义词汇表**：将`object_list`设置为自定义词汇表，DETIC会专门检测这些类别
3. **CLIP过滤**：
   - 对于`"purple cup"`：CLIP会计算检测到的cup与"purple cup"的相似度，只保留紫色杯子
   - 对于`"blue cup with handle"`：CLIP会匹配蓝色且带把手的杯子
   - 只保留相似度高的检测结果（高于`clip_threshold`）
4. **返回结果**：包含匹配的检测框、标签和置信度

### 为什么包含基础名称？

在`object_list`中包含`"cup"`作为备选的原因：
- 如果颜色匹配失败（如杯子不是紫色或蓝色），至少能检测到普通杯子
- 提供更多的候选检测
- CLIP会在后续过滤中区分颜色和属性

## 调整参数建议

### 1. CLIP阈值调整

如果检测不到预期的物体：
- **检测不到**：降低`clip_threshold`（如0.15）
- **误检太多**：提高`clip_threshold`（如0.25）

```python
detector = DeticClipDetector(
    detic_threshold=0.3,
    clip_threshold=0.15,  # 降低阈值以提高召回率
    use_tracking=False
)
```

### 2. 使用多个同义词

对于同一个物体，可以使用多个描述提高检测率：

```python
object_list = [
    "coffee machine",
    "coffee maker",         # 同义词
    "purple cup",
    "blue cup with handle",
    "blue cup",             # 备选（不带handle）
    "cup with handle",      # 备选（不带颜色）
    "cup",                  # 基础名称
    "table",
    "sink",
]
```

## 总结

**修改步骤**：
1. ✅ 找到demo2中定义`object_list`的地方
2. ✅ 修改为包含您要检测的物体（使用完整描述）
3. ✅ 对于空间关系，添加后处理代码
4. ✅ 根据需要调整`clip_threshold`

**关键点**：
- 可以直接使用完整描述（如`"purple cup"`）
- CLIP会自动过滤匹配
- 空间关系需要后处理
- 包含基础名称作为备选可以提高检测率

