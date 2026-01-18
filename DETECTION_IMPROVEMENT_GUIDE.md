# 检测改进指南 - 如何检测更多对象

## 问题：只检测到2个对象

**目标检测对象**：`['coffee machine', 'purple cup', 'blue cup with handle', 'table on the left of sink']`

**实际检测到的对象**：只有 `'purple cup'` 和 `'blue cup with handle'`

**缺失对象**：
- ❌ `'coffee machine'` - 没有被检测到
- ❌ `'table on the left of sink'` - 没有被检测到

---

## 已实施的改进

### 1. 降低检测阈值 ✅

**修改**：将 `DeticPredictorWrapper` 的 `_detic_threshold` 从 `0.5` 降低到 `0.3`

**原因**：较低阈值可以提高召回率，检测到更多对象（可能包括一些置信度较低但正确的检测）

### 2. 添加调试输出 ✅

**修改**：在 `detect_objects` 方法中添加了调试输出，显示：
- 所有候选检测（置信度 >= 0.2）
- 高于阈值的检测（置信度 >= threshold）
- 缺失的对象列表
- 可能的原因和建议

---

## 可能的解决方案

### 方案1：进一步降低阈值（推荐）

如果调试输出显示有低置信度的 `coffee machine` 或 `table` 检测，可以进一步降低阈值：

```python
# 在 Cell 11 中修改
detector.detic_threshold = 0.25  # 或更低，如 0.2
```

**注意**：降低阈值会增加误检，需要在精度和召回率之间平衡。

### 方案2：简化对象名称

对于复杂描述的对象（如 `"table on the left of sink"`），可以：
1. 只检测核心名词：`"table"`
2. 在场景图中用空间关系来表示位置关系

**修改**：在 Cell 10 中修改自定义词汇表：

```python
# 原词汇表
custom_vocab = ['coffee machine', 'purple cup', 'blue cup with handle', 'table on the left of sink']

# 简化后的词汇表
custom_vocab = ['coffee machine', 'purple cup', 'blue cup with handle', 'table', 'sink']
```

然后在场景图生成时，使用空间关系来表示 `"table on the left of sink"`。

### 方案3：使用CLIP补充检测

如果 DETIC 无法检测到某些对象，可以使用 CLIP 进行补充检测：

1. **使用 REFLECT 方法**（Cell 14）：MDETR + CLIP 验证
2. **使用 CLIP 滑动窗口**：在图像的不同区域使用 CLIP 匹配

### 方案4：检查图像中是否真的存在这些对象

可能的原因：
1. **对象不在图像中**：`coffee machine` 或 `table` 可能不在当前帧中
2. **对象被遮挡**：可能被其他对象遮挡
3. **对象太小**：可能太小，无法被检测到
4. **对象名称不匹配**：DETIC 可能使用不同的名称（如 `"coffee maker"` vs `"coffee machine"`）

**检查方法**：
- 查看 RGB 图像，手动确认对象是否存在
- 查看深度图，确认是否有大型平面（可能是 table）

### 方案5：使用多帧融合

对于静态对象（如 `coffee machine` 和 `table`），可以使用多帧融合：
1. 在不同帧中检测这些对象
2. 使用环境记忆（Environment Memory）累积检测结果
3. 即使某些帧中检测不到，也可以通过其他帧的检测结果恢复

---

## 调试步骤

### 步骤1：重新运行检测，查看调试输出

运行 Cell 39（直接检测）或 Cell 30（场景图生成），查看：
1. 所有候选检测列表
2. 缺失对象列表
3. 低置信度检测（如果有）

### 步骤2：根据调试输出调整

如果看到低置信度的 `coffee machine` 或 `table` 检测：
- 进一步降低阈值
- 或手动将这些检测添加到结果中

如果没有看到这些对象：
- 检查图像中是否真的存在
- 尝试简化对象名称
- 使用 CLIP 补充检测

### 步骤3：检查自定义词汇表

确认 Cell 10 中的自定义词汇表设置正确：
```python
custom_vocab = ['coffee machine', 'purple cup', 'blue cup with handle', 'table on the left of sink']
```

**注意**：`"table on the left of sink"` 是一个非常具体的空间关系描述，可能不适合作为对象类别名称。建议：
- 改为 `"table"` 和 `"sink"` 两个独立的类别
- 使用空间关系来表示它们的位置关系

---

## 推荐配置

### 配置1：提高召回率（检测更多对象）

```python
# Cell 10: 简化对象名称
custom_vocab = ['coffee machine', 'purple cup', 'blue cup with handle', 'table', 'sink']

# Cell 11: 降低阈值
detector.detic_threshold = 0.25  # 或 0.2
```

### 配置2：平衡精度和召回率（当前配置）

```python
# Cell 10: 保持原对象名称
custom_vocab = ['coffee machine', 'purple cup', 'blue cup with handle', 'table on the left of sink']

# Cell 11: 中等阈值
detector.detic_threshold = 0.3  # 已修改
```

---

## 下一步行动

1. ✅ **已修改**：降低阈值（0.5 → 0.3）
2. ✅ **已添加**：调试输出
3. **待执行**：重新运行检测，查看调试输出
4. **待决定**：根据调试输出调整策略（进一步降低阈值 vs 简化对象名称 vs 使用CLIP）
