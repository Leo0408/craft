# 混合空间关系判断方法实施总结

## 一、改进概述

基于 CRAFT 和 REFLECT 的对比分析，实现了**混合空间关系判断方法**，结合两者的优势，采用优先级策略判断 `inside` 和 `on_top_of` 关系。

## 二、核心改进

### 2.1 优先级策略

1. **优先级 1：基于 Metadata（置信度 1.0）**
   - 使用 AI2THOR 的 `parentReceptacles` 和 `receptacleObjectIds`
   - 动态判断关系类型（容器 vs 表面）
   - 完全可靠，基于 ground truth 数据

2. **优先级 2：基于位置信息（置信度 0.85）**
   - 使用 3D position 计算 z_diff 和 horizontal_dist
   - 动态表面类型检测（关键词匹配）
   - 阈值：`0.05 < z_diff < 0.5m` 且 `horizontal_dist < 0.2m`

3. **优先级 3：基于点云（置信度 0.75）**
   - 使用点云距离和边界框检查
   - 当 metadata 不可用且位置信息不足时使用
   - 需要点云数据可用

### 2.2 关键函数

#### `determine_spatial_relation_hybrid()`

**位置**：`core/enhanced_scene_graph_utils.py`

**功能**：
- 实现混合方法的核心逻辑
- 返回 `(relation_type, confidence)` 或 `None`
- 支持三种优先级判断方法

**使用示例**：
```python
relation_result = determine_spatial_relation_hybrid(
    obj1, obj2, node1, node2, use_point_cloud=False
)
if relation_result:
    rel_type, confidence = relation_result
    if confidence >= 0.75:
        sg.add_edge(Edge(node1, node2, rel_type))
```

## 三、代码修改

### 3.1 `core/enhanced_scene_graph_utils.py`

**新增内容**：
- `determine_spatial_relation_hybrid()` 函数
- CRAFT 阈值常量（`CRAFT_Z_DIFF_MIN`, `CRAFT_Z_DIFF_MAX`, `CRAFT_HORIZONTAL_DIST_MAX`）
- 改进的 `add_rich_spatial_relations()` 函数（使用混合方法）

**改进点**：
- 动态表面类型判断（避免写死对象类型）
- 置信度分数机制
- 优先级策略实现

### 3.2 `core/enhanced_generate_scene_graph.py`

**修改内容**：
- Step 3：使用混合方法判断 `on_top_of` 关系
- 导入 `determine_spatial_relation_hybrid` 函数
- 改进位置关系判断逻辑

**改进点**：
- 优先使用 metadata（Step 2 已处理）
- 位置信息作为补充（Step 3）
- 点云数据作为后备（Step 4）

### 3.3 `Method.md`

**新增章节**：
- 1.5.3.5 混合空间关系判断方法（Hybrid Spatial Relation Detection）
- 优先级策略说明
- 实现代码示例
- 优势分析

## 四、优势对比

| 特性 | CRAFT 原方法 | REFLECT 原方法 | 混合方法 |
|------|-------------|---------------|---------|
| **准确性** | ⭐⭐⭐ 基于 metadata（高） | ⭐⭐ 基于点云（中等） | ⭐⭐⭐ 优先 metadata（最高） |
| **动态性** | ✅ 完全动态 | ❌ 部分写死 | ✅ 完全动态 |
| **鲁棒性** | ⚠️ 依赖 metadata | ⚠️ 依赖点云 | ✅ 多层回退 |
| **置信度** | ❌ 无 | ❌ 无 | ✅ 有 |
| **可扩展性** | ✅ 高 | ❌ 低 | ✅ 高 |

## 五、使用方式

### 5.1 在 `demo3.ipynb` 中使用

混合方法已经集成到 `generate_scene_graph_from_event_enhanced()` 函数中，无需额外修改：

```python
# Step 2: 生成场景图（已使用混合方法）
initial_sg = generate_scene_graph_from_event_enhanced(
    initial_event,
    task_info,
    timestep=0,
    use_point_cloud=False,  # 仿真环境不使用点云
    use_rich_relations=True
)
```

### 5.2 真实环境使用

在真实环境中，可以启用点云数据：

```python
scene_graph = generate_scene_graph_from_event_enhanced(
    event,
    task_info,
    use_point_cloud=True,  # 启用点云
    use_rich_relations=True
)
```

## 六、预期效果

1. **准确性提升**：
   - 优先使用可靠的 metadata，避免误判
   - 位置信息作为补充，提高覆盖率
   - 点云数据作为后备，增强鲁棒性

2. **动态性保持**：
   - 所有方法都使用动态表面类型判断
   - 不依赖写死的对象类型列表
   - 自动适应新对象类型

3. **置信度机制**：
   - 每个关系都有置信度分数
   - 便于后续处理和筛选
   - 支持基于置信度的决策

## 七、后续优化建议

1. **自适应阈值**：
   - 根据场景类型调整阈值
   - 根据对象类型调整阈值

2. **置信度融合**：
   - 多个方法同时判断时，融合置信度
   - 使用加权平均或最大置信度

3. **时间平滑**：
   - 使用 Environment Memory 平滑关系
   - 减少时间上的跳变

## 八、测试建议

1. **单元测试**：
   - 测试 `determine_spatial_relation_hybrid()` 函数
   - 测试不同优先级场景

2. **集成测试**：
   - 测试 `demo3.ipynb` 中的场景图生成
   - 对比改进前后的结果

3. **性能测试**：
   - 测试混合方法的性能开销
   - 优化计算效率

