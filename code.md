# CRAFT 框架核心代码参考 (CRAFT Core Implementation Reference)

本文档整理了 CRAFT 框架中四个核心阶段的重要代码实现与伪代码逻辑：场景图生成、动作感知约束生成、约束编译以及失败检测。

---

## 1. 场景图生成 (Scene Graph Generation)

### 1.1 核心数据结构 (Pseudocode)
```python
class Node:
    name: str
    object_type: str
    attributes: Dict[str, Any]  # isPickedUp, isOpen, isFilled, isToggled

class Edge:
    start_node: Node
    end_node: Node
    edge_type: str              # inside, on_top_of, holding

class SceneGraph:
    nodes: List[Node]
    edges: Dict[(str, str), Edge]
```

### 1.2 任务相关子图裁剪 (Task-Relevant Subgraph Extraction)
为了减少计算复杂度，我们从完整场景图中提取与任务相关的最小子图。

```python
def extract_task_relevant_subgraph(full_scene_graph: SceneGraph, task_info: Dict) -> SceneGraph:
    # 1. 提取任务涉及的所有对象 (from actions & success_condition)
    relevant_object_names = extract_task_relevant_objects(task_info)
    
    subgraph = SceneGraph()
    # 2. 匹配并添加相关节点
    for node in full_scene_graph.nodes:
        if node.name in relevant_object_names:
            subgraph.add_node(node)
            
    # 3. 添加相关边（保留至少一个端点在子图中的边，以维持环境上下文）
    for edge in full_scene_graph.edges.values():
        if edge.start_node in subgraph or edge.end_node in subgraph:
            subgraph.add_edge(edge)
            
    return subgraph
```

---

## 2. 动作感知约束生成 (Action-aware Constraint Generation)

### 2.1 核心思想
**断言**：因果失败发生在动作层。约束生成必须以动作序列为中心，基于动作语义模板实例化 Pre/Post 条件。

### 2.2 逻辑流 (Python Implementation)
```python
# Step 3: Action-aware Generation
for action in action_sequence:
    # 1. 调用 LLM 匹配模板 (holding, empty, is_on, inside, toggled)
    raw_constraints = llm_prompter.generate_constraints(scene_graph, action)
    
    # 2. 动作绑定与语义匹配 (Action Binding)
    for constraint in raw_constraints:
        # 基于对象名和动作关键词，将约束绑定到具体的 Step 索引
        if 'holding' in constraint.template:
            # 绑定到 pick_up 之后的第一个交互动作
            bound_step = find_first_interaction_step(action_sequence, object_name)
```

---

## 3. 约束编译 (Constraint Compilation)

将结构化的逻辑模板映射为可执行的 Python 表达式。

### 3.1 核心映射逻辑 (Python Implementation)
```python
def compile_constraint(template, node, target):
    if 'holding' in template:
        return "node.attributes.get('isPickedUp', False)"
    
    elif 'empty' in template:
        # 检查容器内是否有任何 'inside' 边
        return "len([e for e in sg.edges.values() if e.end.name == node.name and e.edge_type == 'inside']) == 0"
    
    elif 'is_on' in template:
        return "has_edge(node.name, target_name, 'on_top_of')"
    
    elif 'toggled' in template:
        return "node.attributes.get('isToggled', False)"
        
    return "True"  # Default fallback
```

---

## 4. 失败检测与归因 (Failure Detection & Attribution)

### 4.1 核心算法：动作级因果验证
CRAFT 遵循“动作级顺序校验”原则：一旦 Precondition 失败，立即判定任务由于该动作的物理非法性而终止。

```python
Algorithm ActionLevelDetection(actions, compiled_constraints):
    for step_idx, action in enumerate(actions):
        # A. 校验动作前置条件 (Preconditions)
        pre_constraints = get_constraints(step_idx, type='PRE')
        for constr in pre_constraints:
            if not evaluate(constr, scene_graph_before_action):
                return Failure(type='PRECONDITION_VIOLATION', step=step_idx, reason=constr.desc)
                
        # B. 校验动作后置条件 (Postconditions)
        post_constraints = get_constraints(step_idx, type='POST')
        for constr in post_constraints:
            if not evaluate(constr, scene_graph_after_action):
                return Failure(type='POSTCONDITION_VIOLATION', step=step_idx)
                
    # C. 校验全局目标 (Goal)
    if not evaluate(goal_constraints, final_scene_graph):
        return Failure(type='GOAL_NOT_ACHIEVED')
        
    return Success()
```

### 4.2 评估引擎 (Evaluation Engine)
使用 `eval()` 结合动态上下文执行编译后的代码。

```python
def evaluate_constraint(condition_expr, scene_graph, node):
    context = {
        'node': node,
        'scene_graph': scene_graph,
        'has_edge': lambda s, e, t: (s, e) in scene_graph.edges,
        'len': len
    }
    return eval(condition_expr, context)
```

