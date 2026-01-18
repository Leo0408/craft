# CRAFT vs REFLECT: Case 1-3 对比分析报告

生成时间: 2026-01-18 00:40:26

---

## 总体摘要

- **总测试用例数**: 8
- **CRAFT 检测到失败**: 0 (0.0%)
- **REFLECT 检测到失败**: 0 (0.0%)
- **两者都检测到**: 0

---


## Case 1: 遮挡导致的物体消失误判

**描述**: 机械臂移动时遮挡mug，Detic/MDETR在2-3帧中未检测到mug

**测试用例数**: 1


**Case 统计**:
- CRAFT 检测: 0/1
- REFLECT 检测: 0/1

### 详细结果

#### 1. warmWater/warmWater-8

**GT失败原因**: Missing step to pour wine out of the mug

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---


## Case 2: 关系抖动引发的错误后置条件

**描述**: mug实际未移动，on_top_of关系在关键帧中短暂丢失

**测试用例数**: 2


**Case 统计**:
- CRAFT 检测: 0/2
- REFLECT 检测: 0/2

### 详细结果

#### 1. makeCoffee/makeCoffee-5

**GT失败原因**: The robot failed to put the mug inside the sink (or on top of the sink basin)

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---

#### 2. makeCoffee/makeCoffee-10

**GT失败原因**: The robot put the mug inside the coffee machine after the coffee machine was turned off, as a result, the mug remained empty.

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---


## Case 3: 感知噪声导致的失败级联

**描述**: 中途1帧感知异常，后续4步全部被误判失败

**测试用例数**: 5


**Case 统计**:
- CRAFT 检测: 0/5
- REFLECT 检测: 0/5

### 详细结果

#### 1. makeCoffee/makeCoffee-1

**GT失败原因**: Dropped Mug

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---

#### 2. makeCoffee/makeCoffee-2

**GT失败原因**: Dropped Mug

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---

#### 3. makeCoffee/makeCoffee-3

**GT失败原因**: The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space.

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---

#### 4. makeCoffee/makeCoffee-4

**GT失败原因**: The mug was already filled with water at the beginning of the task execution, and the robot never emptied it.

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---

#### 5. makeCoffee/makeCoffee-10

**GT失败原因**: The robot put the mug inside the coffee machine after the coffee machine was turned off

**CRAFT 检测结果**:

- 是否失败: False
- 失败类型: None
- 失败时间步: None
- 失败原因: None


**REFLECT 检测结果**:

- ⚠️ 未找到结果

---
