# Progressive Explanation 评判标准

## 问题：Progressive Explanation 由什么评判？

### 当前实现

Progressive Explanation 由 `FailureAnalyzer.analyze_failure()` 方法生成，评判标准包括：

1. **约束违反**（优先级最高）
   - 如果检测到约束违反，基于约束违反生成解释
   - 输入：`failed_constraints`（包含约束、动作、原因）
   - 使用 LLM 分析约束违反的根因和因果链

2. **动作失败**（向后兼容）
   - 如果有失败的动作，分析动作失败原因
   - 输入：`task_executor.get_failed_actions()`
   - 使用 LLM 的 `explain_execution_failure()` 方法

3. **规划失败**（如果没有动作失败）
   - 检查任务目标是否达成
   - 使用 LLM 的 `explain_planning_failure()` 方法

### 优化后的评判标准

优化后，Progressive Explanation 的评判优先级：

1. **约束违反**（主要评判标准）
   - 基于约束违反生成根因分析
   - 基于约束违反生成因果链
   - 基于约束违反生成详细分析和修正建议

2. **动作失败**（辅助信息）
   - 如果动作失败，提供动作级别的解释

3. **场景图变化**（上下文信息）
   - 初始状态和最终状态的对比
   - 帮助理解失败发生的上下文

### LLM Prompt 结构

```python
# Root Cause Analysis
prompt = f"""
Task: {task_name}
Constraint Violations: {constraint_descriptions}
Initial State: {initial_state}
Final State: {final_state}

Analyze the root cause of these constraint violations.
"""

# Causal Chain
prompt = f"""
Task: {task_name}
Constraint Violations: {constraint_descriptions}

Create a causal chain explaining the sequence of events.
"""

# Detailed Analysis
prompt = f"""
Task: {task_name}
Constraint Violations: {constraint_descriptions}

Provide detailed analysis with corrective recommendations.
"""
```

### 与 REFLECT 的对比

| 方面 | REFLECT | CRAFT++ (优化后) |
|------|---------|------------------|
| 检测方式 | 子目标渐进验证 | 约束违反检测 |
| 评判标准 | 子目标是否达成 | 约束是否满足 |
| 解释生成 | 基于子目标失败 | 基于约束违反 |
| 优势 | 细粒度检测 | 可执行逻辑验证 |

### 总结

Progressive Explanation 的评判标准：
1. **主要**：约束违反（precondition/postcondition/invariant/goal）
2. **辅助**：动作失败状态
3. **上下文**：场景图变化

优化后，CRAFT 能够像 REFLECT 一样检测到"咖啡机已有杯子却试图放入"的错误，但使用可执行约束而非子目标验证。
