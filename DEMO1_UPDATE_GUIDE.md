# demo1.ipynb 更新指南

本文档说明如何在 `demo1.ipynb` 中应用 CRAFT++ 优化方案。

---

## 一、更新 Step 4: 约束生成

### 当前代码位置
在 `demo1.ipynb` 中找到 `STEP 4: CONSTRAINT GENERATION` cell。

### 更新后的代码

在约束生成后，添加以下代码来显示结构化 JSON 和 AST：

```python
# STEP 4: CONSTRAINT GENERATION (Enhanced)
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CONSTRAINT GENERATION")
print("="*80)

# ... 原有的约束生成代码 ...

print(f"\n✅ Generated {len(constraints_craft)} constraints")

# 显示约束的详细信息（优化新增）
print(f"\n📋 Generated Constraints (with AST):")
for i, constraint in enumerate(constraints_craft, 1):
    constraint_type = constraint.get('type', 'unknown')
    description = constraint.get('description', 'N/A')
    condition_expr = constraint.get('condition_expr', '')
    severity = constraint.get('severity', 'hard')
    eval_time = constraint.get('eval_time', 'now')
    
    type_icon = "🔒" if constraint_type == "precondition" else "✅" if constraint_type == "postcondition" else "⚠️"
    print(f"\n   {i}. {type_icon} [{constraint_type}]")
    print(f"      ID: {constraint.get('id', 'N/A')}")
    print(f"      Description: {description}")
    if condition_expr:
        print(f"      AST: {condition_expr}")
    else:
        print(f"      AST: (not generated, will compile from description)")
    print(f"      Severity: {severity}, Eval Time: {eval_time}")

# 保存约束为 JSON（用于验证）
import json
constraints_json = {
    "constraints": [
        {
            "id": c.get('id', f'C{i}'),
            "type": c.get('type', 'precondition'),
            "description": c.get('description', ''),
            "condition_expr": c.get('condition_expr', ''),
            "severity": c.get('severity', 'hard'),
            "eval_time": c.get('eval_time', 'now')
        }
        for i, c in enumerate(constraints_craft, 1)
    ]
}
print(f"\n💾 Constraints JSON (first constraint example):")
print(json.dumps(constraints_json["constraints"][0] if constraints_json["constraints"] else {}, indent=2))
```

---

## 二、更新 Step 5: 约束编译

### 当前代码位置
在 `demo1.ipynb` 中找到 `STEP 5: CONSTRAINT CODE GENERATION` cell。

### 更新后的代码

确保所有约束都有有效的 `condition_expr`：

```python
# STEP 5: CONSTRAINT CODE GENERATION (AST/DSL) - Enhanced
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CONSTRAINT CODE GENERATION (AST/DSL)")
print("="*80)

from craft.reasoning import ConstraintGenerator

# Compile constraints to executable AST/DSL expressions
compiled_constraints = []

print(f"\n📝 Compiling {len(constraints_craft)} constraints to AST...")

for constraint in constraints_craft:
    condition_expr = constraint.get('condition_expr', '')
    
    # If LLM already generated condition_expr, use it
    if condition_expr:
        compiled_expr = condition_expr
        print(f"   ✅ Constraint {constraint.get('id', 'N/A')}: Using LLM-generated AST: {compiled_expr}")
    else:
        # Otherwise, compile from description
        generator = ConstraintGenerator(None)  # We only need compile_constraint method
        compiled_expr = generator.compile_constraint(constraint)
        if compiled_expr:
            print(f"   ✅ Constraint {constraint.get('id', 'N/A')}: Compiled AST: {compiled_expr}")
        else:
            print(f"   ⚠️  Constraint {constraint.get('id', 'N/A')}: Could not compile to AST")
            compiled_expr = None
    
    compiled_constraints.append({
        'constraint': constraint,
        'condition_expr': compiled_expr
    })

print(f"\n✅ Compiled {len([c for c in compiled_constraints if c['condition_expr']])} constraints with valid AST")
```

---

## 三、更新 Step 6: 失败检测（关键更新）

### 当前代码位置
在 `demo1.ipynb` 中找到 `STEP 6: CODE-BASED FAILURE DETECTION` cell。

### 完全替换为以下代码

```python
# STEP 6: CODE-BASED FAILURE DETECTION (Enhanced with Timing Validation & Atom-level Trace)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CODE-BASED FAILURE DETECTION")
print("="*80)

from craft.reasoning import ConstraintEvaluator

# Initialize constraint evaluator
evaluator = ConstraintEvaluator(min_confidence_threshold=0.7)

# Validate constraints with timing awareness
if len(scene_graphs_craft) > 0 and len(compiled_constraints) > 0:
    print(f"\n🔍 Validating constraints with timing validation and atom-level trace...")
    
    violated_constraints = []
    satisfied_constraints = []
    uncertain_constraints = []
    validation_results = []
    
    # Validate constraints at each action (timing-aware)
    for action_idx, action_result in enumerate(action_results):
        action_name = action_result.get('action', 'N/A')
        action_status = action_result.get('status', 'N/A')
        
        # Get scene graph for this action
        if action_idx < len(scene_graphs_craft):
            scene_graph = scene_graphs_craft[action_idx]
        else:
            scene_graph = scene_graphs_craft[-1]  # Use last scene graph
        
        # Determine evaluation time based on action
        if action_idx == 0:
            evaluation_time = "now"  # Initial state
        elif action_status == "SUCCESS":
            evaluation_time = "post"  # After successful action
        else:
            evaluation_time = "pre"  # Before action (if failed, check preconditions)
        
        print(f"\n   Action {action_idx + 1}: {action_name} ({action_status})")
        print(f"   Evaluation Time: {evaluation_time}")
        
        # Validate each constraint
        for comp_const in compiled_constraints:
            constraint = comp_const['constraint']
            constraint_id = constraint.get('id', 'UNKNOWN')
            constraint_type = constraint.get('type', 'postcondition')
            
            # Skip if constraint doesn't have valid AST
            if not comp_const.get('condition_expr'):
                continue
            
            # Validate constraint with timing awareness
            result = evaluator.validate_constraint(
                constraint=constraint,
                scene_graph=scene_graph,
                evaluation_time=evaluation_time
            )
            
            validation_results.append({
                'action_idx': action_idx,
                'action_name': action_name,
                'evaluation_time': evaluation_time,
                'result': result
            })
            
            # Display result with atom-level trace
            status = result['status']
            if status == 'VIOLATED':
                violated_constraints.append(result)
                print(f"\n      ❌ [{constraint_id}] {constraint_type}: VIOLATED")
                print(f"         Reason: {result['reason']}")
                print(f"         Confidence: {result['confidence']:.2f}")
                print(f"         AST: {result.get('condition_expr', 'N/A')}")
                
                # Display atom-level trace
                if result.get('atom_traces'):
                    print(f"         Atom Traces:")
                    for trace in result['atom_traces']:
                        print(f"           - {trace.atom_expr}: {trace.value} (conf={trace.confidence:.2f}, source={trace.source})")
                        print(f"             Reason: {trace.reason}")
            elif status == 'SATISFIED':
                satisfied_constraints.append(result)
                print(f"      ✅ [{constraint_id}] {constraint_type}: SATISFIED (conf={result['confidence']:.2f})")
            elif status == 'UNCERTAIN':
                uncertain_constraints.append(result)
                print(f"      ⚠️  [{constraint_id}] {constraint_type}: UNCERTAIN (conf={result['confidence']:.2f})")
            elif status == 'SKIP':
                print(f"      ⏭️  [{constraint_id}] {constraint_type}: SKIP ({result['reason']})")
    
    # Final summary
    print(f"\n" + "="*80)
    print(f"📊 VALIDATION SUMMARY")
    print(f"="*80)
    print(f"   Total Constraints Validated: {len(validation_results)}")
    print(f"   ✅ Satisfied: {len(satisfied_constraints)}")
    print(f"   ❌ Violated: {len(violated_constraints)}")
    print(f"   ⚠️  Uncertain: {len(uncertain_constraints)}")
    
    if violated_constraints:
        print(f"\n   🚨 FAILURE DETECTED!")
        print(f"   Violated Constraints:")
        for vc in violated_constraints:
            print(f"      - {vc['id']}: {vc['reason']}")
    else:
        print(f"\n   ✅ All constraints satisfied!")
    
    # Display detailed atom-level trace for first violated constraint (if any)
    if violated_constraints:
        print(f"\n" + "-"*80)
        print(f"📋 DETAILED ATOM-LEVEL TRACE (First Violated Constraint)")
        print(f"-"*80)
        first_violated = violated_constraints[0]
        print(f"Constraint ID: {first_violated['id']}")
        print(f"Status: {first_violated['status']}")
        print(f"Confidence: {first_violated['confidence']:.2f}")
        print(f"Reason: {first_violated['reason']}")
        print(f"AST: {first_violated.get('condition_expr', 'N/A')}")
        print(f"\nAtom Traces:")
        for i, trace in enumerate(first_violated.get('atom_traces', []), 1):
            print(f"  {i}. {trace.atom_expr}")
            print(f"     Value: {trace.value}")
            print(f"     Confidence: {trace.confidence:.2f}")
            print(f"     Source: {trace.source}")
            print(f"     Reason: {trace.reason}")
else:
    print(f"\n⚠️  No scene graphs or constraints available for validation")
    if len(scene_graphs_craft) == 0:
        print(f"   - No scene graphs available")
    if len(compiled_constraints) == 0:
        print(f"   - No compiled constraints available")
```

---

## 四、更新 Step 3: 场景图生成（可选，填充完整属性）

### 当前代码位置
在 `demo1.ipynb` 中找到 `STEP 3: SCENE GRAPH GENERATION` cell。

### 在 Node 创建时添加完整属性

找到创建 Node 的代码（通常在循环中），更新为：

```python
# 在创建 Node 时添加完整属性
position = obj.get("position", {})
pos_tuple = None
if position:
    pos_tuple = (position.get('x', 0), position.get('y', 0), position.get('z', 0))

# 计算 bbox（简化版本，实际应从 AI2THOR 获取）
bbox = None
if position:
    # 假设对象大小为 0.1m x 0.1m x 0.1m（实际应从 metadata 获取）
    bbox = {
        "min": [position.get('x', 0) - 0.05, position.get('y', 0) - 0.05, position.get('z', 0) - 0.05],
        "max": [position.get('x', 0) + 0.05, position.get('y', 0) + 0.05, position.get('z', 0) + 0.05]
    }

# 创建 pose（简化版本）
pose = None
if position:
    rotation = obj.get("rotation", {})
    pose = {
        "position": [position.get('x', 0), position.get('y', 0), position.get('z', 0)],
        "rotation": [rotation.get('x', 0), rotation.get('y', 0), rotation.get('z', 0)]
    }

# 获取置信度（从 AI2THOR metadata，如果可用）
confidence = obj.get("confidence", 1.0)  # AI2THOR 通常不提供，默认为 1.0

# 获取时间戳
import time
last_seen_ts = time.time()  # 或从 event 获取时间戳

# 创建 Node 时包含所有属性
node = Node(
    name=obj_name,
    object_type=obj_type,
    state=state,
    position=pos_tuple,
    bbox=bbox,
    pose=pose,
    confidence=confidence,
    last_seen_ts=last_seen_ts,
    velocity=None  # 需要计算，暂时为 None
)
```

---

## 五、验证更新

### 5.1 运行更新后的代码

1. 按顺序运行所有 cells
2. 检查 Step 4 输出是否显示结构化 JSON 和 AST
3. 检查 Step 6 输出是否显示：
   - 时序验证（pre/post/now/final）
   - Atom-level trace
   - 置信度信息

### 5.2 预期输出示例

**Step 4 输出**：
```
📋 Generated Constraints (with AST):
   1. 🔒 [precondition]
      ID: C1
      Description: Coffee machine must be empty before inserting mug
      AST: (empty coffee_machine)
      Severity: hard, Eval Time: pre
```

**Step 6 输出**：
```
   Action 1: navigate_to_obj, Mug (SUCCESS)
   Evaluation Time: now
      ✅ [C1] precondition: SATISFIED (conf=1.00)
      ⏭️  [C2] postcondition: SKIP (evaluation_time=now, expected 'post' or 'final')
   
   Action 2: pick_up, Mug (SUCCESS)
   Evaluation Time: post
      ✅ [C1] precondition: SATISFIED (conf=1.00)
      ✅ [C2] postcondition: SATISFIED (conf=0.95)
         Atom Traces:
           - (inside mug sink): True (conf=0.95, source=edge_relation)
             Reason: Edge found: inside with confidence 0.95
```

---

## 六、注意事项

1. **导入 ConstraintEvaluator**：确保在 Step 6 之前导入 `ConstraintEvaluator`
2. **JSON 格式**：如果 LLM 没有生成 JSON，代码会回退到文本解析
3. **AST 编译**：如果 LLM 没有生成 `condition_expr`，代码会尝试从 description 编译
4. **时序验证**：确保 `evaluation_time` 与约束类型匹配

---

## 七、故障排除

### 问题 1: ConstraintEvaluator 未找到
**解决**：确保导入 `from craft.reasoning import ConstraintEvaluator`

### 问题 2: Atom-level trace 为空
**解决**：检查 `evaluate` 方法是否设置了 `return_trace=True`

### 问题 3: 时序验证总是 SKIP
**解决**：检查 `evaluation_time` 参数是否与约束类型匹配

---

## 八、参考

- `CRAFT_PLUS_PLUS_OPTIMIZATION_GUIDE.md`: 完整优化方案说明
- `Method.md`: CRAFT++ 框架设计
- `method_add.md`: 优化方案详细说明

