# 约束验证改进说明

## 问题诊断

根据GPT的分析，之前的约束验证存在以下问题：
1. **全部显示SATISFIED不符合实际情况** - 许多约束在当前状态下应该是VIOLATED
2. **没有区分约束类型** - 没有区分precondition/postcondition/invariant/goal
3. **评估时点错误** - 在初始化阶段错误地评估了postcondition和goal
4. **默认返回True** - 无法解析时默认返回True，导致漏报

## 已完成的改进

### 1. 添加约束类型识别 (`_parse_constraints`)

现在会自动识别约束类型：
- **precondition**: "before", "must be opened", "must be empty"
- **postcondition**: "after", "must be moved from X to Y"
- **goal**: "to complete", "final", "success"
- **invariant**: "must not", "must always"

### 2. 改进验证方法 (`validate_constraint`)

- **返回类型**: 从 `bool` 改为 `Tuple[bool, str]`，返回状态和原因
- **评估时点检查**: 根据约束类型检查评估时点是否匹配
- **默认行为**: 从返回True改为返回False（更安全，避免漏报）

### 3. 改进所有检查函数

所有 `_check_*` 函数现在都：
- 返回 `Tuple[bool, str]` 而不是 `bool`
- 提供详细的违反原因
- 正确处理边界情况

### 4. 添加调试输出

在demo notebook中添加了：
- 当前世界状态快照（所有节点和关系）
- 约束类型显示
- 详细的违反原因

## 预期改进效果

### 之前的问题场景：
```
场景: coffee machine (closed), blue cup inside machine, purple cup on table
约束验证结果: 全部 ✅ SATISFIED (错误!)
```

### 改进后的预期结果：
```
场景: coffee machine (closed), blue cup inside machine, purple cup on table

约束1: "coffee machine must be opened" (precondition)
  → ❌ VIOLATED: coffee machine has state 'closed' but required state is 'open'

约束2: "blue cup must be moved from inside coffee machine to table" (postcondition)
  → ❌ VIOLATED: blue cup is still at source location (inside coffee machine)
  → 注意: 这是postcondition，在初始化时检查会显示VIOLATED（因为动作未执行）

约束3: "purple cup must be moved from table" (postcondition)
  → ❌ VIOLATED: purple cup is still at source location (table)

约束4: "filled mug must be on countertop" (goal)
  → ❌ VIOLATED: Goal constraint checked before task completion
```

## 使用方法

### 在代码中使用：

```python
# 验证约束（在当前状态）
is_valid, reason = constraint_generator.validate_constraint(
    constraint, 
    scene_graph, 
    evaluation_time="now"  # 或 "pre", "post", "final"
)

if not is_valid:
    print(f"Constraint violated: {reason}")
```

### 在demo notebook中：

约束验证现在会显示：
1. 当前世界状态快照
2. 每个约束的类型
3. 验证状态（SATISFIED/VIOLATED）
4. 如果违反，显示详细原因

## 关键改进点

1. **约束类型识别**: 自动从描述中推断类型
2. **时点检查**: 确保在正确的时点评估约束
3. **详细反馈**: 提供违反原因，便于调试
4. **更严格的默认**: 无法解析时返回False而不是True
5. **调试支持**: 输出世界状态快照和评估过程

## 下一步建议

1. **单元测试**: 为每个约束类型创建测试用例
2. **LLM Prompt改进**: 让LLM在生成约束时明确标注类型
3. **表达式解析**: 考虑使用更严格的逻辑表达式解析
4. **状态记忆**: 使用环境记忆平滑观测后再评估




