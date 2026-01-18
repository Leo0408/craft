# 场景图生成过程诊断 - 为什么出现0 relations

## 问题总结

### 1. 检测到的对象数量不足

**目标检测对象**：`['coffee machine', 'purple cup', 'blue cup with handle', 'table on the left of sink']`

**实际检测到的对象**：只有 `'purple cup'` 和 `'blue cup with handle'`

**缺失对象**：
- ❌ `'coffee machine'` - 没有被检测到
- ❌ `'table on the left of sink'` - 没有被检测到

**影响**：如果检测到了 `coffee machine` 或 `table`，可能会与杯子建立空间关系（如 `on_top_of`），从而增加关系数量。

---

### 2. 距离阈值问题

#### Frame 0 的位置数据

```
purple cup: 
  - centroid: (551.48, 159.37, 1285.25) mm
  - point cloud: 5392 points

blue cup with handle: 
  - centroid: (-360.87, 238.77, 980.63) mm
  - point cloud: 5643 points
```

#### 距离计算

```python
distance = ||purple_cup_pos - blue_cup_pos||
          = sqrt((551.48 - (-360.87))² + (159.37 - 238.77)² + (1285.25 - 980.63)²)
          = sqrt(912.35² + (-79.4)² + 304.62²)
          = sqrt(832,000 + 6,304 + 92,792)
          = sqrt(931,096)
          ≈ 965.13 mm
          ≈ 0.97 m
```

#### 阈值设置（单位：mm）

```python
IN_CONTACT_DISTANCE = 0.1 m = 100 mm
CLOSE_DISTANCE = 0.4 m = 400 mm
ON_TOP_OF_THRESH = 0.05 m = 50 mm (reduced for real-world)
```

#### 距离检查结果

- ✅ `distance (965mm) > in_contact_thresh (100mm)` → 不满足 `in_contact` 条件
- ✅ `distance (965mm) > close_thresh (400mm)` → 不满足 `near` 或 `on_top_of` 条件

---

### 3. 空间关系计算逻辑问题

在 `perception/scene_analyzer.py` 的 `compute_spatial_relations` 方法中，空间关系检查的顺序和条件如下：

#### Priority 1: `inside` 关系

```python
if bbox1 is not None and bbox2 is not None:
    # 检查bbox1是否在bbox2内，或bbox2是否在bbox1内
    inside_12 = self._check_inside(min1, max1, min2, max2, inside_overlap_ratio)
    inside_21 = self._check_inside(min2, max2, min1, max1, inside_overlap_ratio)
    
    if inside_12:
        relations.append((det1['label'], det2['label'], 'inside', 0.9))
        continue  # 跳过其他关系
    elif inside_21:
        relations.append((det2['label'], det1['label'], 'inside', 0.9))
        continue
```

**问题**：两个杯子不可能有 `inside` 关系，所以这个检查会失败。

#### Priority 2: `on_top_of` 关系

```python
# ⚠️ 问题：这个检查在 distance < close_thresh 内部
if distance < close_thresh:
    z_diff = pos1[2] - pos2[2]
    horizontal_dist = np.linalg.norm(pos1[:2] - pos2[:2])
    
    # 要求：
    # 1. distance < close_thresh (400mm)
    # 2. z_diff > on_top_thresh (50mm)
    # 3. horizontal_dist < close_thresh * 0.5 (200mm)
    
    if z_diff > on_top_thresh and horizontal_dist < close_thresh * 0.5:
        relations.append((det1['label'], det2['label'], 'on_top_of', 0.85))
        continue
```

**问题**：
- `distance (965mm) > close_thresh (400mm)` → 不会进入这个检查
- 即使进入，`horizontal_dist (915.80mm) > close_thresh*0.5 (200mm)` → 也不会满足 `on_top_of` 条件

#### Priority 3: `in_contact` 关系

```python
if distance < in_contact_thresh:  # 100mm
    relations.append((det1['label'], det2['label'], 'in_contact', 1.0))
    continue
```

**问题**：`distance (965mm) > in_contact_thresh (100mm)` → 不满足条件

#### Priority 4: `near` 关系

```python
if distance < close_thresh:  # 400mm
    relations.append((det1['label'], det2['label'], 'near', 0.7))
```

**问题**：`distance (965mm) > close_thresh (400mm)` → 不满足条件

---

## 根本原因

1. **检测问题**：只检测到了2个对象，没有检测到 `coffee machine` 和 `table on the left of sink`
   - 如果检测到了这些静态对象，可能会与杯子建立空间关系

2. **距离问题**：两个杯子之间的距离（965mm）超过了所有关系的阈值（最大400mm）
   - 导致不会触发任何空间关系检查

3. **逻辑问题**：空间关系计算逻辑要求距离必须在阈值内才会检查关系
   - 如果距离超过阈值，即使有其他特征（如垂直差），也不会生成关系

---

## 解决方案

### 方案1：增加阈值（推荐）

将 `CLOSE_DISTANCE` 从 `0.4m` 增加到 `1.0m` 或 `1.5m`，以适应真实场景中对象之间的距离。

```python
# 在 perception/scene_analyzer.py 中修改
CLOSE_DISTANCE = 1.5  # 从 0.4 增加到 1.5 m = 1500 mm
```

### 方案2：添加 "far" 或 "visible" 关系

对于距离较远但仍可见的对象对，添加 "far" 或 "visible" 关系：

```python
# 在 compute_spatial_relations 方法中添加
if distance > close_thresh and distance < 2.0 * close_thresh:  # 400mm - 800mm
    relations.append((det1['label'], det2['label'], 'far', 0.5))
```

### 方案3：改进检测逻辑

检查为什么 `coffee machine` 和 `table on the left of sink` 没有被检测到：

1. **降低检测阈值**：如果使用 CLIP 验证，尝试降低 `clip_threshold`
2. **检查对象名称**：确认对象名称是否与检测器期望的格式匹配
3. **添加调试输出**：查看 DETIC 检测到的所有候选对象（包括低置信度的）

### 方案4：使用点云距离

使用 `EnhancedSpatialRelationComputer`（Cell 18），它使用点云距离而不是质心距离，可能更准确。

---

## 调试建议

### 1. 检查检测结果

在场景图生成代码中添加调试输出：

```python
# 在 scene_graph_builder.py 的 process_frame 方法中
detections = self.detector.detect_objects(rgb_pil, object_list)
print(f"🔍 Detection results:")
print(f"  Requested objects: {object_list}")
print(f"  Detected objects: {[d['label'] for d in detections]}")
for det in detections:
    print(f"    - {det['label']}: confidence={det.get('confidence', 0):.3f}")
```

### 2. 检查距离和阈值

在 `compute_spatial_relations` 方法中添加调试输出：

```python
distance = np.linalg.norm(pos1 - pos2)
print(f"  Distance between {det1['label']} and {det2['label']}: {distance:.2f} mm")
print(f"  Thresholds: close={close_thresh:.2f} mm, contact={in_contact_thresh:.2f} mm")
print(f"  Relation checks: near={distance < close_thresh}, contact={distance < in_contact_thresh}")
```

### 3. 检查 bbox3d

确认 `bbox3d` 是否正确传递到 `detections_for_relations`：

```python
for label in pcd_dict.keys():
    if label not in self.bbox3d_dict:
        print(f"  ⚠️  Warning: {label} has no bbox3d")
        continue
    bbox3d = self.bbox3d_dict[label]
    print(f"  {label}: bbox3d={bbox3d}")
```

---

## 下一步行动

1. ✅ 添加调试输出，查看为什么 `coffee machine` 和 `table` 没有被检测到
2. ✅ 检查距离计算和阈值设置
3. ✅ 考虑增加 `CLOSE_DISTANCE` 阈值或添加 "far" 关系
4. ✅ 使用 `EnhancedSpatialRelationComputer` 作为备选方案
