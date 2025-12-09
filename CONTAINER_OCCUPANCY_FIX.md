# 容器占用检测优化方案

## 问题分析

### REFLECT 检测到的错误
```
The failure at 00:51 occurred because the robot attempted to place the mug inside 
the coffee machine while there was already a cup inside it. The robot should have 
removed the existing cup from the coffee machine before attempting to place the mug inside.
```

### CRAFT 为什么检测不出来？

1. **约束生成问题**：
   - 生成的约束都是关于初始状态的（Mug must be inside Sink, CoffeeMachine must be on top of CounterTop）
   - **缺少针对 `put_in` 动作的 precondition**：容器必须为空
   - 没有 occupancy constraint（容器占用约束）

2. **验证时机问题**：
   - 只在最终状态验证约束
   - **没有在动作执行时验证 precondition**
   - 即使生成了"容器必须为空"的约束，也没有在 put_in 动作前验证

3. **验证逻辑问题**：
   - 当前的 `evaluate_constraint` 函数过于简单
   - 只检查了 `empty` 关键字，没有检查容器内是否有其他对象

## 优化方案

### 1. 改进约束生成 Prompt

在 LLM prompt 中明确要求生成容器占用约束：

```python
'constraint-generator': {
    'template-user': '''...
Generate constraints covering:
1. Preconditions (must be satisfied before actions)
   - For put_in actions: container must be empty
   - For put_on actions: surface must be clear
   - For toggle actions: object must be accessible
2. Postconditions (must be satisfied after actions)
3. Invariants (must always be satisfied)
4. Goal constraints (final success conditions)
5. Causal chains (e.g., fill → has_water → heat)
6. **Occupancy constraints** (containers must be empty before insertion)

Example occupancy constraint:
{
  "id": "C8",
  "type": "pre",
  "description": "Coffee machine must be empty before inserting mug",
  "condition_expr": "(empty coffee_machine)",
  "severity": "hard",
  "eval_time": "pre",
  "action": "put_in"  // Associated action
}
...'''
}
```

### 2. 添加 Occupancy Constraint 检查

在 `constraint_generator.py` 中添加专门的容器占用检查：

```python
def _check_container_empty(self, description: str, scene_graph: SceneGraph) -> Tuple[bool, str]:
    """Check if container is empty (precondition for put_in operations)"""
    # Find container name
    container_name = None
    words = description.split()
    
    for word in words:
        if 'machine' in word.lower() or 'container' in word.lower():
            idx = words.index(word)
            if idx > 0:
                container_name = f"{words[idx-1]} {word}"
            else:
                container_name = word
            break
    
    if not container_name:
        if 'coffee machine' in description.lower():
            container_name = 'coffee machine'
        else:
            return True, "Cannot identify container"
    
    # Check if any object is inside this container
    container_node = scene_graph.get_node(container_name)
    if not container_node:
        return False, f"Container '{container_name}' not found in scene graph"
    
    # Check all edges to see if anything is inside the container
    items_inside = []
    for (start_name, end_name), edge in scene_graph.edges.items():
        if edge.end.name == container_name and edge.edge_type in ['inside', 'in']:
            items_inside.append(edge.start.name)
    
    if items_inside:
        return False, f"Container '{container_name}' is not empty: {', '.join(items_inside)} inside"
    else:
        return True, f"Container '{container_name}' is empty"
```

### 3. 添加动作相关的约束验证

在 Step 6 失败检测中，添加动作执行时的验证：

```python
# 在动作执行时验证相关约束
for action_result in action_results:
    action_name = action_result.get('action_name', '')
    action_idx = action_result.get('action_idx', 0)
    
    # 获取动作前的场景图
    if action_idx > 0:
        scene_graph_before = scene_graphs_craft[action_idx - 1]
    else:
        scene_graph_before = scene_graphs_craft[0]
    
    # 获取动作后的场景图
    scene_graph_after = scene_graphs_craft[action_idx] if action_idx < len(scene_graphs_craft) else scene_graphs_craft[-1]
    
    # 验证与该动作相关的 precondition
    for constraint in constraints_craft:
        # 检查约束是否与该动作相关
        if self._is_constraint_related_to_action(constraint, action_name):
            if constraint['type'] == 'precondition' and constraint.get('eval_time') == 'pre':
                is_valid, reason, conf = evaluator.evaluate(
                    constraint['condition_expr'],
                    scene_graph_before
                )
                if not is_valid:
                    print(f"❌ Precondition violated before {action_name}: {reason}")
                    violated_constraints.append({
                        'constraint': constraint,
                        'action': action_name,
                        'action_idx': action_idx,
                        'reason': reason,
                        'eval_time': 'pre'
                    })
    
    # 验证与该动作相关的 postcondition
    for constraint in constraints_craft:
        if self._is_constraint_related_to_action(constraint, action_name):
            if constraint['type'] == 'postcondition' and constraint.get('eval_time') == 'post':
                is_valid, reason, conf = evaluator.evaluate(
                    constraint['condition_expr'],
                    scene_graph_after
                )
                if not is_valid:
                    print(f"❌ Postcondition violated after {action_name}: {reason}")
                    violated_constraints.append({
                        'constraint': constraint,
                        'action': action_name,
                        'action_idx': action_idx,
                        'reason': reason,
                        'eval_time': 'post'
                    })

def _is_constraint_related_to_action(self, constraint: Dict, action_name: str) -> bool:
    """Check if constraint is related to a specific action"""
    description = constraint.get('description', '').lower()
    condition_expr = constraint.get('condition_expr', '').lower()
    
    # 检查约束描述或表达式中是否包含动作相关的对象
    if action_name == 'put_in':
        # put_in 相关的约束应该提到容器
        return 'machine' in description or 'container' in description or 'coffee' in description
    elif action_name == 'put_on':
        return 'on' in description or 'top' in description
    elif action_name == 'pick_up':
        return 'pick' in description or 'hold' in description
    # ... 其他动作
    
    return False
```

### 4. 改进 ConstraintEvaluator 支持容器占用检查

在 `constraint_evaluator.py` 中添加：

```python
def _check_empty(self, expr: str, scene_graph: SceneGraph) -> Tuple[bool, str, float]:
    """Check if container is empty: (empty container)"""
    parts = expr.split()
    if len(parts) < 2:
        return False, "Insufficient arguments for empty check", 0.0
    
    container_name = parts[1].lower()
    
    # Find container node
    container_node = None
    for node in scene_graph.nodes:
        if container_name in node.name.lower() or node.name.lower() in container_name:
            container_node = node
            break
    
    if not container_node:
        return False, f"Container '{container_name}' not found", 0.0
    
    # Check if any object is inside this container
    items_inside = []
    for (start_name, end_name), edge in scene_graph.edges.items():
        if edge.end.name == container_node.name and edge.edge_type in ['inside', 'in']:
            items_inside.append(edge.start.name)
    
    if items_inside:
        return False, f"Container '{container_node.name}' is not empty: {', '.join(items_inside)} inside", 1.0
    else:
        return True, f"Container '{container_node.name}' is empty", 1.0
```

### 5. 改进 Progressive Explanation

当前问题：Progressive explanation 只在有 failed_actions 时才生成。

优化方案：

```python
def analyze_failure(self, task_executor: TaskExecutor,
                   initial_scene_graph: SceneGraph,
                   final_scene_graph: SceneGraph,
                   failed_constraints: Optional[List[Dict]] = None,
                   action_results: Optional[List[Dict]] = None) -> Dict:
    """
    Analyze failures including constraint violations
    
    Args:
        failed_constraints: List of violated constraints with action context
        action_results: List of action execution results
    """
    analysis = {
        'root_cause': None,
        'causal_chain': None,
        'detailed_analysis': None,
        'violated_constraints': failed_constraints or [],
        'failed_actions': []
    }
    
    # 如果有违反的约束，基于约束生成解释
    if failed_constraints:
        # 构建约束违反的描述
        constraint_descriptions = []
        for vc in failed_constraints:
            constraint = vc.get('constraint', {})
            action = vc.get('action', 'unknown')
            reason = vc.get('reason', '')
            constraint_descriptions.append(
                f"Before {action}: {constraint.get('description', '')} - {reason}"
            )
        
        # 使用 LLM 生成根因分析
        prompt = f"""
Task: {task_executor.task_info.get('name', '')}
Violated Constraints:
{chr(10).join(constraint_descriptions)}

Initial State: {initial_scene_graph.to_text()}
Final State: {final_scene_graph.to_text()}

Analyze the root cause and provide a causal chain explaining why these constraints were violated.
"""
        
        analysis['root_cause'] = self.llm_prompter.query(...)
        analysis['causal_chain'] = self.llm_prompter.query(...)
    
    # 如果有失败的动作，分析动作失败
    failed_actions = [r for r in (action_results or []) if r.get('status') == 'FAILED']
    if failed_actions:
        # ... 现有的动作失败分析
    
    return analysis
```

## 实施步骤

1. ✅ 更新 LLM prompt 要求生成容器占用约束
2. ✅ 添加 `_check_container_empty` 方法
3. ✅ 在 Step 6 添加动作执行时的约束验证
4. ✅ 改进 ConstraintEvaluator 支持容器占用检查
5. ✅ 改进 FailureAnalyzer 基于约束违反生成解释

## 预期效果

优化后，CRAFT 应该能够：

1. **生成容器占用约束**：
   ```json
   {
     "id": "C8",
     "type": "pre",
     "description": "Coffee machine must be empty before inserting mug",
     "condition_expr": "(empty coffee_machine)",
     "eval_time": "pre",
     "action": "put_in"
   }
   ```

2. **在 put_in 动作前检测到违反**：
   ```
   ❌ Precondition violated before put_in: Container 'CoffeeMachine' is not empty: Cup inside
   ```

3. **生成详细的解释**：
   ```
   Root Cause: The robot attempted to place the mug inside the coffee machine 
   while there was already a cup inside it. The precondition "Coffee machine 
   must be empty before inserting mug" was violated.
   
   Causal Chain:
   1. Initial state: Cup is inside coffee machine
   2. Robot attempts put_in(mug, coffee_machine)
   3. Precondition check: (empty coffee_machine) → FALSE
   4. Constraint violation detected
   5. Action should be blocked or cup should be removed first
   ```

