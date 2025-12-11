# 时序验证更新方案

## 问题

当前 CRAFT 框架只在最终状态验证约束，无法检测动作执行时的违反。例如：
- REFLECT 检测到：咖啡机里已有杯子，但机器人仍试图放入杯子
- CRAFT 检测不到：因为没有在 put_in 动作前验证"容器必须为空"的约束

## 解决方案

在 Step 6 失败检测中添加动作执行时的约束验证。

### 更新 demo1.ipynb Step 6

```python
# STEP 6: CODE-BASED FAILURE DETECTION (WITH TIMING VALIDATION)
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CODE-BASED FAILURE DETECTION")
print("="*80)

from craft.reasoning import ConstraintEvaluator

evaluator = ConstraintEvaluator()

# Validate constraints with timing
if len(scene_graphs_craft) > 0 and len(compiled_constraints) > 0:
    print(f"\nValidating constraints with timing (pre/post action checks)...")
    
    violated_constraints = []
    validation_results = []
    
    # Validate constraints at each action
    for action_idx, action_result in enumerate(action_results):
        action_name = action_result.get('action_name', '')
        action_status = action_result.get('status', '')
        
        print(f"\n--- Action {action_idx + 1}: {action_name} ---")
        
        # Get scene graphs before and after action
        scene_graph_before = scene_graphs_craft[action_idx] if action_idx < len(scene_graphs_craft) else scene_graphs_craft[0]
        scene_graph_after = scene_graphs_craft[action_idx + 1] if action_idx + 1 < len(scene_graphs_craft) else scene_graphs_craft[-1]
        
        # Check preconditions before action
        for comp_const in compiled_constraints:
            constraint = comp_const['constraint']
            condition_expr = comp_const.get('condition_expr', '')
            
            # Check if constraint is related to this action
            if _is_constraint_related_to_action(constraint, action_name):
                constraint_type = constraint.get('type', '')
                eval_time = constraint.get('eval_time', 'now')
                
                # Validate precondition before action
                if constraint_type == 'precondition' and eval_time == 'pre':
                    is_valid, reason, conf = evaluator.evaluate(
                        condition_expr,
                        scene_graph_before
                    )
                    
                    validation_results.append({
                        'constraint': constraint,
                        'action': action_name,
                        'action_idx': action_idx,
                        'eval_time': 'pre',
                        'is_valid': is_valid,
                        'reason': reason,
                        'confidence': conf
                    })
                    
                    if not is_valid:
                        print(f"  ❌ Precondition violated: {constraint.get('description', '')[:60]}...")
                        print(f"     Reason: {reason}")
                        violated_constraints.append({
                            'constraint': constraint,
                            'action': action_name,
                            'action_idx': action_idx,
                            'reason': reason,
                            'eval_time': 'pre'
                        })
                    else:
                        print(f"  ✅ Precondition satisfied: {constraint.get('description', '')[:60]}...")
        
        # Check postconditions after action (only if action succeeded)
        if action_status == 'SUCCESS':
            for comp_const in compiled_constraints:
                constraint = comp_const['constraint']
                condition_expr = comp_const.get('condition_expr', '')
                
                if _is_constraint_related_to_action(constraint, action_name):
                    constraint_type = constraint.get('type', '')
                    eval_time = constraint.get('eval_time', 'now')
                    
                    # Validate postcondition after action
                    if constraint_type == 'postcondition' and eval_time == 'post':
                        is_valid, reason, conf = evaluator.evaluate(
                            condition_expr,
                            scene_graph_after
                        )
                        
                        validation_results.append({
                            'constraint': constraint,
                            'action': action_name,
                            'action_idx': action_idx,
                            'eval_time': 'post',
                            'is_valid': is_valid,
                            'reason': reason,
                            'confidence': conf
                        })
                        
                        if not is_valid:
                            print(f"  ❌ Postcondition violated: {constraint.get('description', '')[:60]}...")
                            print(f"     Reason: {reason}")
                            violated_constraints.append({
                                'constraint': constraint,
                                'action': action_name,
                                'action_idx': action_idx,
                                'reason': reason,
                                'eval_time': 'post'
                            })
                        else:
                            print(f"  ✅ Postcondition satisfied: {constraint.get('description', '')[:60]}...")
    
    # Also validate final state constraints
    final_scene_graph = scene_graphs_craft[-1]
    print(f"\n--- Final State Validation ---")
    
    for comp_const in compiled_constraints:
        constraint = comp_const['constraint']
        condition_expr = comp_const.get('condition_expr', '')
        constraint_type = constraint.get('type', '')
        eval_time = constraint.get('eval_time', 'now')
        
        # Validate goal constraints at final state
        if constraint_type == 'goal' or eval_time == 'final':
            is_valid, reason, conf = evaluator.evaluate(
                condition_expr,
                final_scene_graph
            )
            
            validation_results.append({
                'constraint': constraint,
                'action': 'final',
                'action_idx': len(action_results),
                'eval_time': 'final',
                'is_valid': is_valid,
                'reason': reason,
                'confidence': conf
            })
            
            if not is_valid:
                print(f"  ❌ Goal constraint violated: {constraint.get('description', '')[:60]}...")
                violated_constraints.append({
                    'constraint': constraint,
                    'action': 'final',
                    'action_idx': len(action_results),
                    'reason': reason,
                    'eval_time': 'final'
                })
    
    # Summary
    print(f"\n" + "="*80)
    print(f"✅ Validated {len(validation_results)} constraint checks")
    print(f"   Violated: {len(violated_constraints)}")
    print(f"   Satisfied: {len(validation_results) - len(violated_constraints)}")
    
    if violated_constraints:
        print(f"\n❌ Violated Constraints:")
        for i, vc in enumerate(violated_constraints, 1):
            constraint = vc['constraint']
            action = vc['action']
            reason = vc['reason']
            print(f"   {i}. [{vc['eval_time']}] {action}: {constraint.get('description', 'N/A')[:60]}...")
            print(f"      Reason: {reason}")
    else:
        print(f"\n✅ All constraints satisfied!")
    
    failed_constraints = violated_constraints
    violated_count = len(violated_constraints)
else:
    print(f"\n⚠️  Cannot validate: need scene graphs and compiled constraints")
    validation_results = []
    violated_constraints = []
    failed_constraints = []
    violated_count = 0


def _is_constraint_related_to_action(constraint: Dict, action_name: str) -> bool:
    """Check if constraint is related to a specific action"""
    description = constraint.get('description', '').lower()
    condition_expr = constraint.get('condition_expr', '').lower()
    
    # Check constraint description or expression for action-related keywords
    if action_name == 'put_in':
        # put_in related constraints should mention container or machine
        return ('machine' in description or 'container' in description or 
                'coffee' in description or 'empty' in description or
                'machine' in condition_expr or 'container' in condition_expr)
    elif action_name == 'put_on':
        return ('on' in description or 'top' in description or 
                'surface' in description or 'on_top_of' in condition_expr)
    elif action_name == 'pick_up':
        return ('pick' in description or 'hold' in description or 
                'grab' in description)
    elif action_name in ['toggle_on', 'toggle_off']:
        return ('open' in description or 'close' in description or 
                'toggle' in description or 'switch' in description)
    elif action_name == 'navigate_to_obj':
        return False  # Navigation constraints are usually not action-specific
    
    # Default: check if action name appears in description
    return action_name.lower() in description
```

## Progressive Explanation 更新

更新 Step 7 使用约束违反生成解释：

```python
# STEP 7: PROGRESSIVE EXPLANATION
# ============================================================================
print("\n" + "="*80)
print("STEP 7: PROGRESSIVE EXPLANATION")
print("="*80)

# Generate progressive explanation using FailureAnalyzer
if violated_count > 0 or len(action_failures) > 0:
    print(f"\nGenerating progressive explanation for failures...")
    
    # Initialize failure analyzer
    failure_analyzer = FailureAnalyzer(llm_prompter)
    
    # Perform failure analysis with constraint violations
    if len(scene_graphs_craft) > 0:
        initial_sg = scene_graphs_craft[0]
        final_sg = scene_graphs_craft[-1]
        
        explanation = failure_analyzer.analyze_failure(
            initial_scene_graph=initial_sg,
            final_scene_graph=final_sg,
            failed_constraints=failed_constraints if failed_constraints else None,
            task_info=task_info_craft
        )
        
        print(f"\n📊 Progressive Explanation:")
        print(f"\n🔍 Root Cause:")
        print(f"   {explanation.get('root_cause', 'N/A')}")
        print(f"\n🔗 Causal Chain:")
        print(f"   {explanation.get('causal_chain', 'N/A')}")
        print(f"\n📝 Detailed Analysis:")
        print(f"   {explanation.get('detailed_analysis', 'N/A')}")
    else:
        print(f"\n⚠️  Cannot generate explanation: need scene graphs")
else:
    print(f"\n✅ No failures detected - all constraints satisfied and actions successful!")
```

## 预期效果

优化后，CRAFT 应该能够：

1. **在 put_in 动作前检测到违反**：
   ```
   --- Action 9: put_in ---
     ❌ Precondition violated: Coffee machine must be empty before inserting mug...
        Reason: Container 'CoffeeMachine' is not empty: Cup inside
   ```

2. **生成详细的解释**：
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

