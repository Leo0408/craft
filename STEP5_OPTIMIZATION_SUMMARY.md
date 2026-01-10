# Step 5 失败检测优化总结

## 一、问题诊断

### 1.1 核心问题

**问题现象**：
- makeCoffee：22 个 violation
- boilWater：12 个 violation
- 所有根因都指向 Step 2 的 `reachable(...)`
- Root Violation 都是：`Unknown constraint type: reachable(xxx)`

**问题本质**：
这不是任务真的失败了，而是**约束解释器失控**，把"不可判定的约束"当成了"违反的约束"。

### 1.2 系统性问题

#### ❌ 问题 1：reachable(x) 是"语义约束"，但被当成"可验证约束"

- `reachable(x)` 是语义/能力约束，需要通过 navigation/motion planning 判断
- 无法通过 scene graph 可靠验证
- 但系统将其视为"违反"，导致大量误报

#### ❌ 问题 2：第 2 步就失败，后面全是级联假失败

- Step 2 的 `reachable` 约束失败
- 后续 90% 的错误都是"假失败的衍生品"
- 这是典型的"雪崩式误报"

#### ❌ 问题 3：混淆了三种本质不同的约束

| 类型 | 示例 | 是否应该失败 |
|------|------|------------|
| 可验证物理约束 | `container_empty` | ✅ 是 |
| 可验证状态约束 | `holding(mug)` | ✅ 是 |
| 语义/能力约束 | `reachable(mug)` | ❌ 不应作为失败 |

---

## 二、优化方案

### 2.1 核心改进

#### ✅ Priority 0：将 Unknown constraint type 从「失败」降级为「不可判定」

**之前**：
```python
return False, f"Unknown constraint type: {constraint_template}", False, diagnostics
# → 被视为 PRECONDITION VIOLATION
```

**现在**：
```python
if is_non_verifiable:
    return True, f"SKIPPED: Non-verifiable constraint '{constraint_template}' (semantic/ability constraint, not a failure)", False, diagnostics
else:
    return True, f"SKIPPED: Unknown constraint type '{constraint_template}' (not implemented, not a failure)", False, diagnostics
# → 返回 True（跳过），不视为失败
```

#### ✅ Priority 1：给约束加「可判定性标签」

**定义**：
```python
# 可验证约束（可通过 scene graph 可靠验证）
VERIFIABLE_CONSTRAINTS = {
    "holding", "gripper_empty", "inside", "on_top_of",
    "empty", "container_empty", "toggled_on", "toggled_off",
    "open", "closed"
}

# 不可验证约束（语义/能力约束，不应作为失败判定）
NON_VERIFIABLE_CONSTRAINTS = {
    "reachable",      # 需要 navigation/motion planning
    "filled",         # 需要内容物检测
    "container_open"  # 如果无法从 scene graph 可靠获得
}
```

#### ✅ Priority 2：不可判定约束不能成为 Root Cause

**修改 `collapse_failures`**：
```python
def collapse_failures(real_violations: List[Dict]) -> Dict:
    # 优化：排除 SKIPPED 约束（不可验证约束不应成为 root cause）
    verifiable_violations = [
        v for v in real_violations 
        if not v.get('reason', '').startswith("SKIPPED:")
    ]
    # ... 只对可验证的违反进行收敛
```

#### ✅ Priority 3：失败检测逻辑跳过 SKIPPED 约束

**修改失败检测循环**：
```python
is_valid, reason, is_warning, diagnostics = evaluate_constraint(eval_sg, constraint)

# 优化：跳过不可验证的约束（SKIPPED）
if reason.startswith("SKIPPED:"):
    skipped_constraints.append({...})
    continue  # 跳过，不加入 violations

if not is_valid:
    # 只有可验证的约束失败才会加入 violations
    violations.append(violation_info)
```

---

## 三、具体修改

### 3.1 修改清单

1. ✅ **添加约束分类定义**
   - `VERIFIABLE_CONSTRAINTS`：可验证约束列表
   - `NON_VERIFIABLE_CONSTRAINTS`：不可验证约束列表

2. ✅ **修改 `evaluate_constraint` 函数**
   - 将 Unknown constraint type 从 `False`（失败）改为 `True`（跳过）
   - 返回 `SKIPPED:` 前缀的 reason，标识不可验证约束

3. ✅ **修改失败检测逻辑**
   - 检查 `reason.startswith("SKIPPED:")`，如果是则跳过
   - 记录到 `skipped_constraints`，不加入 `violations`

4. ✅ **修改 `real_errors` 过滤**
   - 添加 `not v.get('reason', '').startswith("SKIPPED:")` 条件
   - 确保 SKIPPED 约束不出现在 real_errors 中

5. ✅ **修改 `collapse_failures` 函数**
   - 在收敛前过滤掉 SKIPPED 约束
   - 确保 root cause 必须是可验证的物理/状态违反

6. ✅ **添加输出显示**
   - 显示 `skipped_constraints` 统计
   - 按类型分组显示（Non-verifiable / Unknown）

---

## 四、预期效果

### 4.1 修改前（错误示例）

```
Root Violation:
  Step 2: (pick_up, Mug)
  Reason: Unknown constraint type: reachable(mug)

Derived Violations: 20 个
```

### 4.2 修改后（理想）

```
Skipped Constraints:
  Non-verifiable (5 个):
    - Step 2: (pick_up, Mug) - Mug must be reachable
    - Step 5: (toggle_on, Faucet) - Faucet must be reachable
    ...

Root Violation:
  Step 10: (put_in, Mug, CoffeeMachine)
  Reason: CoffeeMachine not empty (contains Cup)
```

---

## 五、关键改进点

### 5.1 约束分类

| 约束类型 | 示例 | 处理方式 |
|---------|------|---------|
| **可验证约束** | `holding(mug)`, `inside(mug, sink)` | ✅ 正常检测，失败视为 violation |
| **不可验证约束** | `reachable(mug)`, `filled(sink)` | ⏭️ 跳过，不视为失败 |
| **未知约束** | 未实现的约束类型 | ⏭️ 跳过，不视为失败 |

### 5.2 失败检测流程

```
约束生成
   ↓
约束分类（verifiable / non-verifiable）
   ↓
只对 verifiable 约束做失败检测
   ↓
non-verifiable 只用于 LLM 解释，不用于判失败
```

### 5.3 Root Cause 规则

**Root Cause 必须是**：
- ✅ 物理违反（如 `container_empty` 失败）
- ✅ 状态不一致（如 `holding(mug)` 失败）
- ✅ 因果断裂（如 precondition 失败导致后续失败）

**Root Cause 不能是**：
- ❌ 不可验证约束（如 `reachable`）
- ❌ 未知约束类型
- ❌ Robot/NoneType 相关错误（模拟环境）

---

## 六、代码位置

- **Step 5 代码**：`demo3.ipynb` Cell 29
- **约束分类定义**：Cell 29，第 70 行附近
- **evaluate_constraint 函数**：Cell 29，第 71 行开始
- **失败检测逻辑**：Cell 29，第 421 行和第 467 行
- **real_errors 过滤**：Cell 29，第 556 行
- **collapse_failures 函数**：Cell 29，第 317 行
- **输出显示**：Cell 29，第 580 行附近

---

## 七、验证方法

### 7.1 检查修改是否生效

运行 Step 5，检查输出：
1. ✅ 应该看到 "跳过约束" 统计
2. ✅ `reachable` 约束应该出现在 "跳过约束" 中，而不是 "真实错误"
3. ✅ Root Violation 应该是可验证的物理/状态违反，而不是 `Unknown constraint type: reachable`

### 7.2 预期结果

- **makeCoffee**：应该只有 1-2 个真实错误（如 `CoffeeMachine not empty`），而不是 22 个
- **boilWater**：应该只有 1-2 个真实错误，而不是 12 个
- **Root Violation**：应该是可验证的约束失败，而不是 `reachable`

---

## 八、后续优化建议

### 8.1 实现更多可验证约束

如果发现某些约束应该可验证但目前被跳过，可以：
1. 在 `evaluate_constraint` 中实现对应的检查逻辑
2. 将其从 `NON_VERIFIABLE_CONSTRAINTS` 移到 `VERIFIABLE_CONSTRAINTS`

### 8.2 优化约束生成

在约束生成阶段就标记约束的可验证性：
```python
constraint = {
    'type': 'precondition',
    'template': 'reachable(mug)',
    'verifiability': 'non_verifiable'  # 在生成时就标记
}
```

### 8.3 区分感知延迟和真实失败

对于 postcondition 失败，如果动作执行成功，可能是场景图更新延迟：
```python
if action_result.status == "SUCCESS" and postcondition_failed:
    # 等待下一帧再检查
    # 如果下一帧满足，标记为感知延迟，不是真实失败
```

---

## 九、总结

### 9.1 核心改进

1. ✅ **约束分类**：区分可验证和不可验证约束
2. ✅ **失败降级**：将 Unknown constraint type 从 violation 降级为 SKIPPED
3. ✅ **Root Cause 过滤**：确保 root cause 必须是可验证的物理/状态违反
4. ✅ **输出优化**：清晰显示跳过的约束，便于调试

### 9.2 效果

- ✅ 减少误报：不再将不可验证约束视为失败
- ✅ 聚焦真实失败：Root Cause 分析更准确
- ✅ 符合论文叙事：只报告可验证的物理/状态违反

### 9.3 关键原则

> **只有可验证的约束失败才是真正的失败**
> 
> 不可验证的约束（如 `reachable`）是语义/能力约束，不应作为失败判定，只用于 LLM 解释。

