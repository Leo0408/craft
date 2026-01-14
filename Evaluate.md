# CRAFT 评估方法文档

本文档描述了 CRAFT 框架的评估系统，包括评估指标、评估流程和使用方法。

## 目录

1. [评估指标](#评估指标)
2. [失败类型分类](#失败类型分类)
3. [评估流程](#评估流程)
4. [使用方法](#使用方法)
5. [评估参数配置](#评估参数配置)

---

## 评估指标

CRAFT 评估系统提供以下评估指标：

### 1. 检测准确率 (Detection Accuracy)
- **定义**: 正确检测到失败/成功的任务比例
- **计算**: `(TP + TN) / (TP + TN + FP + FN)`
- **说明**: 评估系统是否能正确判断任务是否失败

### 2. 失败类型准确率 (Failure Type Accuracy)
- **定义**: 正确识别失败类型的任务比例
- **计算**: 仅在检测到失败的任务中计算
- **说明**: 评估系统是否能正确分类失败类型（F1-F5）

### 3. 时间步准确率 (Timestep Accuracy)
- **定义**: 正确识别失败时间步的任务比例
- **计算**: 预测时间步与真实时间步匹配（允许容差）
- **容差**: 默认5秒（对于时间字符串格式如"00:51"）
- **说明**: 评估系统是否能准确定位失败发生的时间

### 4. 根因归因准确率 (Attribution Accuracy)
- **定义**: 正确识别失败根因的任务比例
- **计算**: 失败原因与真实原因匹配
- **说明**: 评估系统是否能准确解释失败的根本原因

### 5. 误报率 (False Positive Rate, FPR)
- **定义**: 将成功任务误判为失败的比例
- **计算**: `FP / (FP + TN)`

### 6. 漏报率 (False Negative Rate, FNR)
- **定义**: 将失败任务误判为成功的比例
- **计算**: `FN / (FN + TP)`

### 7. 各失败类型的精确率、召回率和F1分数
- **精确率 (Precision)**: `TP / (TP + FP)`
- **召回率 (Recall)**: `TP / (TP + FN)`
- **F1分数**: `2 * (Precision * Recall) / (Precision + Recall)`

---

## 失败类型分类

CRAFT 使用5种标准失败类型（F1-F5）：

### F1: Precondition Violation / Missing required step
- **定义**: 前置条件违反或缺少必要步骤
- **关键词**: precondition, missing, did not, forgot to, failed to, without, before, required step, prerequisite
- **示例**: "The robot failed to put the mug inside the coffee machine because the coffee machine was not empty (precondition violation)."

### F2: Postcondition Violation / Action had no effect
- **定义**: 后置条件违反或动作没有产生预期效果
- **关键词**: postcondition, not satisfied, not inside, not on top, not holding, action had no effect, no effect
- **示例**: "The robot failed to put the mug inside the coffee machine because the mug was not placed inside after the action (postcondition violation)."

### F3: Physical Constraint / Physically impossible
- **定义**: 物理约束违反或物理上不可能
- **关键词**: occupied, already inside, already in, space is occupied, blocking, blocked, in the way, obstructing, physically impossible, cannot be put, limited space, capacity
- **示例**: "The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space."

### F4: State Inconsistency / State drift
- **定义**: 状态不一致或状态漂移
- **关键词**: state, inconsistency, drift, must be toggled, must be open, must be empty, must be filled, state violation, wrong state
- **示例**: "The robot failed to toggle on the stove burner because the burner state was inconsistent with the expected state."

### F5: Perception Uncertainty / Uncertain due to occlusion
- **定义**: 感知不确定性或由于遮挡导致的不确定
- **关键词**: wrong, incorrect, mistakenly, mistake, misidentified, not found, occlusion, uncertain, ambiguous, wrong burner, wrong object, perception, uncertainty
- **示例**: "The robot failed to pick up the pot because it was uncertain about the pot's location due to occlusion."

---

## 评估流程

### Step 1: 数据准备
1. 加载任务数据（task_info, events, actions）
2. 加载或准备 Ground Truth（真实标签）
   - `has_failure`: 是否有失败（bool）
   - `failure_type`: 失败类型（F1-F5）
   - `failure_step`: 失败时间步（整数或"mm:ss"格式）
   - `failure_reason`: 失败原因（一句话格式）

### Step 2: 执行 CRAFT 失败检测
1. 生成场景图
2. 生成约束
3. 检查约束违反
4. LLM 分析根因
5. 输出结构化检测结果：
   - `has_detected_failure`: 是否检测到失败
   - `detected_failure_type`: 检测到的失败类型（F1-F5）
   - `detected_failure_step`: 检测到的失败时间步
   - `detected_failure_reason`: 检测到的失败原因（一句话格式）

### Step 3: 评估单个任务
使用 `CRAFTEvaluator.evaluate_single_task()` 方法：
- 比较检测结果与 Ground Truth
- 计算各项指标
- 返回评估结果

### Step 4: 计算统计指标
使用 `CRAFTEvaluator.calculate_statistics()` 方法：
- 汇总所有任务的评估结果
- 计算总体统计指标
- 计算各失败类型的精确率、召回率、F1分数

### Step 5: 生成评估报告
使用 `CRAFTEvaluator.generate_report()` 方法：
- 生成 JSON 格式报告
- 生成 Markdown 格式报告

---

## 使用方法

### 基本使用（单个任务评估）

```python
from demo3 import CRAFTEvaluator

# 初始化评估器
evaluator = CRAFTEvaluator()

# 评估单个任务
result = evaluator.evaluate_single_task(
    task_name="makeCoffee",
    task_info=task_info,
    actions=actions,
    violations=violations,
    real_errors=real_errors,
    root_violation=root_violation,
    skipped_constraints=skipped_constraints,
    ground_truth={
        'has_failure': True,
        'failure_type': 'F3',
        'failure_step': '00:51',
        'failure_reason': 'The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space.'
    }
)

# 计算统计指标
stats = evaluator.calculate_statistics()

# 生成报告
evaluator.generate_report(output_dir='evaluation_results')
```

### 批量评估（多个任务）

```python
# 批量评估配置
ENABLE_BATCH_EVALUATION = True
GPT_MODEL = 'gpt-4'  # 或 'gpt-3.5-turbo'
TASK_INSTANCE_INDEX = [0]  # 或 [0, 1] 或 'all'
SIM_DATA_ROOT = '../reflect/reflect_dataset/sim_data'

# 执行批量评估
results = evaluate_all_sim_datasets(
    sim_data_root=SIM_DATA_ROOT,
    gpt_model=GPT_MODEL,
    instance_filter=TASK_INSTANCE_INDEX,
    verbose=True
)
```

### Case 对比分析

```python
# 定义测试用例
CASE_DEFINITIONS = {
    'Case 1': {
        'name': 'warmWater',
        'cases': ['warmWater-8'],
        'gt_failure_reason': 'The robot failed to put the pot inside the sink because...'
    },
    # ...
}

# 执行对比分析
comparison_results = {}
for case_name, case_def in CASE_DEFINITIONS.items():
    # 加载数据集
    # 执行 CRAFT 流程
    # 执行 REFLECT 流程（如果可用）
    # 对比结果
    pass

# 生成对比报告
generate_case_comparison_report(comparison_results, summary_stats)
```

---

## 评估参数配置

### 时间步对齐参数

- **容差 (Tolerance)**: 默认5秒
  - 用于时间字符串格式（如"00:51"）的匹配
  - 对于整数时间步，使用1秒容差

- **时间格式转换**:
  - 支持将 action step 索引转换为时间格式（mm:ss）
  - 需要提供 events 和 actions 参数

### LLM 分析参数

- **模型选择**: `gpt-4` 或 `gpt-3.5-turbo`
- **API Key**: 从环境变量或全局变量获取
  - `POLOAPI_API_KEY` 或 `OPENAI_API_KEY`
- **最大 tokens**: 默认1000

### 失败类型判断参数

- **关键词匹配**: 基于约束描述和违反原因的关键词匹配
- **优先级**: 
  1. 首先根据原始失败类型（PRECONDITION/POSTCONDITION）判断
  2. 然后根据关键词匹配
  3. 默认返回 F5（感知不确定性）

---

## 评估 Cell 说明

### Step 7: CRAFT 评估系统

**Cell 40**: `CRAFTEvaluator` 类定义
- `evaluate_single_task()`: 评估单个任务
- `calculate_statistics()`: 计算统计指标
- `generate_report()`: 生成评估报告

**Cell 41**: 使用示例
- 演示如何评估当前任务
- 从 Step 5 的输出中获取检测结果

**Cell 42**: 完整评估
- 计算统计指标
- 生成 JSON 和 Markdown 报告

### Step 7.1: Case 1-3 测试用例对比分析

**Cell 43**: Case 定义和配置
- 定义测试用例
- 配置 API Key 和 GPT 模型

**Cell 44**: 执行对比分析
- 加载实际数据集
- 执行完整 CRAFT 流程
- 对比 CRAFT 和 REFLECT 结果

**Cell 45**: 生成对比报告
- 显示总体对比数据摘要
- 生成详细的 Markdown 报告

---

## 输出格式

### 结构化检测结果

```python
{
    'has_detected_failure': bool,
    'detected_failure_type': 'F1' | 'F2' | 'F3' | 'F4' | 'F5',
    'detected_failure_step': int | str,  # action step 或 "mm:ss"
    'detected_failure_reason': str,  # 一句话格式
    'detected_failure_attribution': {
        'step': int,
        'action': str,
        'constraint_description': str,
        'failure_type_raw': str
    }
}
```

### 评估结果

```python
{
    'task_name': str,
    'has_detected_failure': bool,
    'detected_failure_type': str,
    'detected_failure_step': int | str,
    'detected_failure_reason': str,
    'ground_truth': {
        'has_failure': bool,
        'failure_type': str,
        'failure_step': int | str,
        'failure_reason': str
    },
    'evaluation': {
        'detection_correct': bool,
        'failure_type_correct': bool,
        'timestep_correct': bool,
        'attribution_correct': bool
    }
}
```

### 统计指标

```python
{
    'total_tasks': int,
    'tasks_with_gt': int,
    'metrics': {
        'detection_accuracy': float,
        'failure_type_accuracy': float,
        'timestep_accuracy': float,
        'attribution_accuracy': float,
        'false_positive_rate': float,
        'false_negative_rate': float
    },
    'failure_type_metrics': {
        'F1': {'precision': float, 'recall': float, 'f1': float, 'tp': int, 'fp': int, 'fn': int},
        # ... F2-F5
    }
}
```

---

## 注意事项

1. **时间步对齐**: 
   - Ground Truth 中的时间步可能是"mm:ss"格式（如"00:51"）
   - CRAFT 检测结果中的时间步可能是 action step 索引（整数）
   - 系统会自动尝试转换和匹配

2. **失败原因格式**:
   - LLM 会生成一句话格式的失败原因
   - 格式: "The robot failed to [action] because [reason]."
   - 参考 Ground Truth 的格式

3. **失败类型判断**:
   - 优先根据原始失败类型（PRECONDITION/POSTCONDITION）判断
   - 然后根据关键词匹配
   - 如果无法匹配，默认返回 F5

4. **API Key 配置**:
   - 优先从环境变量获取
   - 如果环境变量不存在，从全局变量获取
   - 如果都不存在，使用默认值（在 Cell 13 中定义）

---

## 相关文件

- `demo3.ipynb`: 主评估 notebook
- `evaluation_results/`: 评估结果输出目录
  - `craft_evaluation_results.json`: JSON 格式报告
  - `craft_evaluation_results.md`: Markdown 格式报告
  - `batch/`: 批量评估结果
  - `case_comparison_report.md`: Case 对比分析报告

---

## 更新日志

- **2026-01-13**: 
  - 添加5种标准失败类型（F1-F5）
  - 修改LLM分析，输出一句话格式的失败原因
  - 添加时间步对齐功能，支持xx:xx格式
  - 创建评估方法文档
