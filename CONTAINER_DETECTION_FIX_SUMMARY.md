# 容器占用检测优化总结

## 问题分析

### REFLECT 检测到的错误
```
The failure at 00:51 occurred because the robot attempted to place the mug inside 
the coffee machine while there was already a cup inside it. The robot should have 
removed the existing cup from the coffee machine before attempting to place the mug inside.
```

### CRAFT 为什么检测不出来？

1. **约束生成问题** ❌
   - 生成的约束都是关于初始状态的
   - **缺少针对 `put_in` 动作的 precondition**：容器必须为空
   - 没有 occupancy constraint（容器占用约束）

2. **验证时机问题** ❌
   - 只在最终状态验证约束
   - **没有在动作执行时验证 precondition**
   - 即使生成了"容器必须为空"的约束，也没有在 put_in 动作前验证

3. **验证逻辑问题** ❌
   - 当前的 `evaluate_constraint` 函数过于简单
   - 只检查了 `empty` 关键字，没有检查容器内是否有其他对象

## 已实施的优化

### 1. ✅ 改进约束生成 Prompt

更新了 `reasoning/llm_prompter.py`：
- 明确要求生成容器占用约束
- 添加了 occupancy constraint 的示例
- 强调 put_in 动作需要"容器必须为空"的 precondition

### 2. ✅ 改进 ConstraintEvaluator

更新了 `reasoning/constraint_evaluator.py`：
- 添加了 `_check_empty()` 方法专门检查容器是否为空
- 检查场景图中是否有对象在容器内（通过 edges）
- 返回详细的违反原因（列出容器内的对象）

### 3. ✅ 改进 FailureAnalyzer

更新了 `reasoning/failure_analyzer.py`：
- 支持基于约束违反生成解释
- 添加了 `root_cause`, `causal_chain`, `detailed_analysis` 字段
- 优先级：约束违反 > 动作失败 > 规划失败

## 需要手动更新的部分

### demo1.ipynb Step 6

需要在 Step 6 中添加时序验证逻辑，参考 `TIMING_VALIDATION_UPDATE.md`：

1. **在动作执行时验证约束**
   - 动作前验证 precondition
   - 动作后验证 postcondition
   - 使用 `ConstraintEvaluator` 评估 AST 表达式

2. **检查约束与动作的相关性**
   - `_is_constraint_related_to_action()` 函数
   - 判断约束是否与特定动作相关

3. **收集违反的约束**
   - 记录违反的约束、动作、原因
   - 传递给 FailureAnalyzer 生成解释

### demo1.ipynb Step 7

更新 Step 7 使用新的 FailureAnalyzer API：

```python
explanation = failure_analyzer.analyze_failure(
    initial_scene_graph=initial_sg,
    final_scene_graph=final_sg,
    failed_constraints=failed_constraints,  # 传递约束违反
    task_info=task_info_craft
)
```

## 预期效果

优化后，CRAFT 应该能够：

### 1. 生成容器占用约束

```json
{
  "id": "C8",
  "type": "pre",
  "description": "Coffee machine must be empty before inserting mug",
  "condition_expr": "(empty coffee_machine)",
  "severity": "hard",
  "eval_time": "pre"
}
```

### 2. 在 put_in 动作前检测到违反

```
--- Action 9: put_in ---
  ❌ Precondition violated: Coffee machine must be empty before inserting mug...
     Reason: Container 'CoffeeMachine' is not empty: Cup inside
```

### 3. 生成详细的解释

```
Root Cause: The robot attempted to place the mug inside the coffee machine 
while there was already a cup inside it. The precondition "Coffee machine 
must be empty before inserting mug" was violated because the container was 
not empty.

Causal Chain:
1. Initial state: Cup is inside coffee machine
2. Robot attempts put_in(mug, coffee_machine) at step 9
3. Precondition check: (empty coffee_machine) → FALSE
4. Constraint violation detected: Container 'CoffeeMachine' is not empty: Cup inside
5. Action should be blocked or cup should be removed first
```

## Progressive Explanation 评判标准

### 当前实现

Progressive Explanation 由 `FailureAnalyzer.analyze_failure()` 生成，评判标准：

1. **约束违反**（优先级最高）
   - 基于约束违反生成根因分析
   - 基于约束违反生成因果链
   - 基于约束违反生成详细分析

2. **动作失败**（向后兼容）
   - 如果有失败的动作，分析动作失败原因

3. **规划失败**（如果没有动作失败）
   - 检查任务目标是否达成

### 与 REFLECT 的对比

| 方面 | REFLECT | CRAFT++ (优化后) |
|------|---------|------------------|
| 检测方式 | 子目标渐进验证 | 约束违反检测 |
| 评判标准 | 子目标是否达成 | 约束是否满足 |
| 解释生成 | 基于子目标失败 | 基于约束违反 |
| 优势 | 细粒度检测 | 可执行逻辑验证 |

## 文件更新清单

1. ✅ `reasoning/llm_prompter.py` - 更新约束生成 prompt
2. ✅ `reasoning/constraint_evaluator.py` - 添加容器占用检查
3. ✅ `reasoning/failure_analyzer.py` - 支持基于约束违反的解释
4. 📝 `demo1.ipynb` Step 6 - 需要手动添加时序验证（参考 `TIMING_VALIDATION_UPDATE.md`）
5. 📝 `demo1.ipynb` Step 7 - 需要手动更新 FailureAnalyzer 调用

## 下一步

1. 在 `demo1.ipynb` 中应用 `TIMING_VALIDATION_UPDATE.md` 中的代码
2. 测试容器占用检测
3. 验证 Progressive Explanation 生成

---

**优化完成时间**：2024年
**相关文档**：
- `CONTAINER_OCCUPANCY_FIX.md` - 详细问题分析和方案
- `TIMING_VALIDATION_UPDATE.md` - Step 6 更新代码
- `PROGRESSIVE_EXPLANATION_ANSWER.md` - Progressive Explanation 评判标准

