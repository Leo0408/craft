# Demo4 对齐 Demo3 计划

## 任务概述

1. ✅ 检查 scene graph 生成是否正确
2. 📋 对齐后续模块到 demo3.ipynb：
   - Step 9: 约束生成（Action-aware）
   - Step 10: 约束编译
   - Step 11: 失败检测（置信度感知）
3. 📊 实现评估部分（1-2）项

---

## 1. Scene Graph 生成状态检查

### 当前状态
- ✅ 生成了 9 个 memory-smoothed scene graphs
- ⚠️ 每帧只有 2 个对象（purple cup, blue cup with handle）
- ❌ 空间关系为 0（所有帧都是 0 relations）

### 问题分析
1. **对象检测不足**：`coffee machine` 和 `table on the left of sink` 没有被检测到
2. **空间关系为 0**：即使只有 2 个对象，距离 965mm < 1500mm 阈值，应该能检测到 `near` 关系
   - 可能原因：代码未重新加载（需要重启 kernel）
   - 已添加调试输出，需要重新运行查看

### 已实施的改进
1. ✅ 降低检测阈值（0.5 → 0.3）
2. ✅ 增加 CLOSE_DISTANCE 阈值（0.4m → 1.5m）
3. ✅ 添加调试输出（显示距离、阈值、关系计算过程）

---

## 2. 模块对齐计划

### Step 9: 约束生成（对齐 demo3.ipynb Step 3）

**Demo3 方式**：
- 使用 `constraint_generator.generate_constraints_for_action()` 方法
- 逐个动作生成约束，生成时就绑定 `action_index`
- 支持 LLM 和模板两种方法

**Demo4 当前方式**：
- 使用 `constraint_generator.generate_constraints_from_templates()`
- 模板方法，但没有正确绑定动作索引

**需要修改**：
1. 使用 `generate_constraints_for_action()` 替代 `generate_constraints_from_templates()`
2. 逐个动作生成约束，确保绑定 `action_index`

### Step 10: 约束编译（对齐 demo3.ipynb Step 4）

**Demo3 方式**：
- 使用 `compiled_constraints`，每个约束包含 `condition_expr`
- 编译逻辑在 `ConstraintGenerator.compile_constraints()` 中

**Demo4 当前方式**：
- 有编译逻辑，但比较简单
- 需要对齐到 demo3 的编译方式

### Step 11: 失败检测（对齐 demo3.ipynb Step 5）

**Demo3 方式**：
- 使用 `evaluate_constraint_with_confidence()` 函数
- 支持置信度感知验证
- 使用环境记忆平滑的场景图
- 输出结构化失败检测结果

**Demo4 当前方式**：
- 有基本框架，但需要对齐细节

---

## 3. 评估部分（1-2）项

根据用户附加的 Cell 32 第一行：
```
# Step 7.2: 批量评估（详细日志版本，不使用 LLM）
# 输出格式：Ground truth + 约束条件 log + 根因分析
```

**评估项（1-2）**应该是指：
1. **Ground truth** - 真实标签对比
2. **约束条件 log** - 约束违反日志

**实现内容**：
1. 从 task_info 中获取 ground truth（如果有）
2. 生成约束条件 log（所有约束的验证结果）
3. 对比检测结果与 ground truth
4. 输出评估指标（检测准确率、失败类型准确率等）

---

## 实施步骤

1. ✅ 已添加场景图调试输出
2. ⏳ 对齐约束生成模块（Step 9）
3. ⏳ 对齐约束编译模块（Step 10）
4. ⏳ 对齐失败检测模块（Step 11）
5. ⏳ 实现评估部分（1-2）项
