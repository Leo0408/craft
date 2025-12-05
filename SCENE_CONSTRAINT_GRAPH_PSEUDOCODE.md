# 场景约束图构建伪代码

## 1. 场景图构建 (Scene Graph Construction)

```
算法: BuildSceneGraph
输入: 
    - detections: List[Detection]  // 对象检测结果
    - spatial_relations: List[Relation]  // 空间关系
    - task_info: Dict  // 任务信息
输出: SceneGraph  // 场景图

BEGIN
    // 初始化场景图
    scene_graph = SceneGraph()
    
    // 步骤1: 从检测结果创建节点
    FOR EACH detection IN detections DO
        node = Node(
            name = detection.object_name,
            object_type = detection.object_type,
            state = detection.state,  // e.g., "empty", "filled", "closed"
            position = detection.position,
            attributes = detection.attributes
        )
        scene_graph.add_node(node)
    END FOR
    
    // 步骤2: 从空间关系创建边
    FOR EACH relation IN spatial_relations DO
        start_node = scene_graph.get_node(relation.obj1)
        end_node = scene_graph.get_node(relation.obj2)
        
        IF start_node != NULL AND end_node != NULL THEN
            edge = Edge(
                start = start_node,
                end = end_node,
                edge_type = relation.relation_type,  // e.g., "on_top_of", "inside", "near"
                confidence = relation.confidence
            )
            scene_graph.add_edge(edge)
        END IF
    END FOR
    
    // 步骤3: 关联任务信息
    scene_graph.task = task_info
    
    RETURN scene_graph
END
```

## 2. 约束生成 (Constraint Generation)

```
算法: GenerateConstraints
输入:
    - scene_graph: SceneGraph  // 当前场景图
    - task_info: Dict  // 任务信息 {name, success_condition}
    - goal: String  // 任务目标描述
输出: List[Constraint]  // 约束列表

BEGIN
    // 步骤1: 将场景图转换为文本描述
    scene_text = scene_graph.to_text()
    // 例如: "table, purple cup (empty), blue cup (empty), coffee machine (closed). 
    //        purple cup (empty) is on_top_of table. 
    //        blue cup (empty) is inside coffee machine (closed)."
    
    // 步骤2: 构建LLM提示
    prompt = FormatPrompt(
        task = task_info.name,
        scene_graph = scene_text,
        goal = goal
    )
    
    // 步骤3: 调用LLM生成约束
    llm_response = LLMQuery(
        system_prompt = "You are a constraint generator for robot tasks...",
        user_prompt = prompt,
        max_tokens = 800
    )
    
    // 步骤4: 解析LLM响应
    constraints = ParseConstraints(llm_response)
    
    RETURN constraints
END

// 约束解析子算法
算法: ParseConstraints
输入: llm_response: String  // LLM响应文本
输出: List[Constraint]  // 结构化约束列表

BEGIN
    constraints = []
    lines = Split(llm_response, '\n')
    
    FOR EACH line IN lines DO
        line = Trim(line)
        
        // 跳过空行和注释
        IF line == "" OR StartsWith(line, '#') THEN
            CONTINUE
        END IF
        
        // 移除编号 (如 "1. ", "2. ")
        IF IsDigit(line[0]) THEN
            line = Split(line, '.')[1]  // 取编号后的部分
        END IF
        
        // 解析约束格式: description (condition)
        IF Contains(line, '(') AND Contains(line, ')') THEN
            description = Split(line, '(')[0].Trim()
            condition = ExtractBetween(line, '(', ')')
            
            constraint = {
                description: description,
                condition: condition,
                raw: line
            }
            constraints.append(constraint)
        ELSE
            // 无条件的约束
            constraint = {
                description: line,
                condition: NULL,
                raw: line
            }
            constraints.append(constraint)
        END IF
    END FOR
    
    RETURN constraints
END
```

## 3. 约束验证 (Constraint Validation)

```
算法: ValidateConstraint
输入:
    - constraint: Dict  // 约束字典 {description, condition, raw}
    - scene_graph: SceneGraph  // 当前场景图
输出: Boolean  // True表示满足约束，False表示违反约束

BEGIN
    description = constraint.description.ToLower()
    condition = constraint.condition
    
    // 如果没有条件，默认满足
    IF condition == NULL THEN
        RETURN True
    END IF
    
    // 模式1: 移动约束 "X must be moved from A to B"
    IF Contains(description, "must be moved from") AND Contains(description, "to") THEN
        RETURN CheckMovementConstraint(description, scene_graph)
    END IF
    
    // 模式2: 位置约束 "X must be on/in A"
    IF Contains(description, "must be") AND 
       (Contains(description, "on") OR Contains(description, "inside") OR Contains(description, "in")) THEN
        RETURN CheckLocationConstraint(description, scene_graph)
    END IF
    
    // 模式3: 状态约束 "X must be open/closed/empty/filled"
    IF Contains(description, "must be") AND 
       (Contains(description, "open") OR Contains(description, "closed") OR 
        Contains(description, "empty") OR Contains(description, "filled")) THEN
        RETURN CheckStateConstraint(description, scene_graph)
    END IF
    
    // 模式4: 负约束 "X must not be ..."
    IF Contains(description, "must not") THEN
        RETURN CheckNegativeConstraint(description, scene_graph)
    END IF
    
    // 模式5: 容器空约束 (前置条件)
    IF Contains(description, "empty") AND 
       (Contains(description, "container") OR Contains(description, "machine")) THEN
        RETURN CheckContainerEmpty(description, scene_graph)
    END IF
    
    // 默认: 无法解析时保守地返回True
    RETURN True
END

// 移动约束检查子算法
算法: CheckMovementConstraint
输入:
    - description: String  // 约束描述
    - scene_graph: SceneGraph  // 场景图
输出: Boolean

BEGIN
    // 解析: "blue cup must be moved from inside coffee machine to table"
    words = Split(description, ' ')
    
    // 提取对象名、源位置、目标位置
    must_idx = IndexOf(words, "must")
    from_idx = IndexOf(words, "from")
    to_idx = IndexOf(words, "to")
    
    obj_name = Join(words[0:must_idx], ' ')  // "blue cup"
    source_location = Join(words[from_idx+1:to_idx], ' ')  // "inside coffee machine"
    dest_location = Join(words[to_idx+1:], ' ')  // "table"
    
    // 检查对象是否在源位置 (应该不在)
    is_at_source = CheckObjectLocation(obj_name, source_location, scene_graph)
    
    // 检查对象是否在目标位置 (应该在)
    is_at_dest = CheckObjectLocation(obj_name, dest_location, scene_graph)
    
    // 约束满足: 不在源位置 AND 在目标位置
    RETURN (NOT is_at_source) AND is_at_dest
END

// 对象位置检查子算法
算法: CheckObjectLocation
输入:
    - obj_name: String  // 对象名称
    - location: String  // 位置描述
    - scene_graph: SceneGraph  // 场景图
输出: Boolean

BEGIN
    obj_node = scene_graph.get_node(obj_name)
    IF obj_node == NULL THEN
        RETURN False
    END IF
    
    location_lower = location.ToLower()
    
    // 查找位置节点
    location_node = NULL
    FOR EACH node IN scene_graph.nodes DO
        IF node.name.ToLower() IN location_lower OR location_lower IN node.name.ToLower() THEN
            location_node = node
            BREAK
        END IF
    END FOR
    
    IF location_node == NULL THEN
        RETURN False
    END IF
    
    // 确定期望的关系类型
    expected_relations = []
    IF Contains(location_lower, "inside") OR Contains(location_lower, "in") THEN
        expected_relations = ["inside", "in"]
    ELSE IF Contains(location_lower, "on") OR Contains(location_lower, "top") THEN
        expected_relations = ["on_top_of", "on", "on top of"]
    ELSE IF Contains(location_lower, "near") THEN
        expected_relations = ["near"]
    ELSE IF Contains(location_lower, "contact") THEN
        expected_relations = ["in_contact", "contact"]
    ELSE
        expected_relations = ["on", "inside", "in", "on_top_of", "near", "in_contact"]
    END IF
    
    // 检查边 (双向)
    edge_key1 = (obj_node.name, location_node.name)
    edge_key2 = (location_node.name, obj_node.name)
    
    FOR EACH edge_key IN [edge_key1, edge_key2] DO
        IF edge_key IN scene_graph.edges THEN
            edge = scene_graph.edges[edge_key]
            edge_type_lower = edge.edge_type.ToLower()
            
            // 检查关系类型是否匹配
            FOR EACH expected_rel IN expected_relations DO
                IF expected_rel IN edge_type_lower THEN
                    RETURN True
                END IF
            END FOR
        END IF
    END FOR
    
    RETURN False
END

// 容器空约束检查子算法 (前置条件)
算法: CheckContainerEmpty
输入:
    - description: String  // 约束描述
    - scene_graph: SceneGraph  // 场景图
输出: Boolean

BEGIN
    // 提取容器名称 (如 "coffee machine")
    container_name = ExtractContainerName(description)
    
    container_node = scene_graph.get_node(container_name)
    IF container_node == NULL THEN
        RETURN True  // 容器不存在，假设为空
    END IF
    
    // 检查是否有对象在容器内
    FOR EACH (start_name, end_name), edge IN scene_graph.edges DO
        IF edge.end.name == container_name AND 
           edge.edge_type IN ["inside", "in"] THEN
            // 有对象在容器内，约束违反
            RETURN False
        END IF
    END FOR
    
    // 容器为空，约束满足
    RETURN True
END
```

## 4. 完整流程 (Complete Workflow)

```
算法: BuildConstraintGraph
输入:
    - detections: List[Detection]  // 对象检测
    - spatial_relations: List[Relation]  // 空间关系
    - task_info: Dict  // 任务信息
输出: ConstraintGraph  // 约束图

BEGIN
    // 步骤1: 构建场景图
    scene_graph = BuildSceneGraph(detections, spatial_relations, task_info)
    
    // 步骤2: 生成约束
    constraints = GenerateConstraints(
        scene_graph = scene_graph,
        task_info = task_info,
        goal = task_info.success_condition
    )
    
    // 步骤3: 验证约束
    constraint_status = []
    FOR EACH constraint IN constraints DO
        is_satisfied = ValidateConstraint(constraint, scene_graph)
        constraint_status.append({
            constraint: constraint,
            satisfied: is_satisfied,
            status: IF is_satisfied THEN "SATISFIED" ELSE "VIOLATED"
        })
    END FOR
    
    // 步骤4: 构建约束图
    constraint_graph = {
        scene_graph: scene_graph,
        constraints: constraints,
        constraint_status: constraint_status,
        violated_constraints: Filter(constraint_status, status == "VIOLATED"),
        satisfied_constraints: Filter(constraint_status, status == "SATISFIED")
    }
    
    RETURN constraint_graph
END
```

## 5. 约束类型说明

### 5.1 移动约束 (Movement Constraint)
```
格式: "X must be moved from A to B"
含义: 对象X应该从位置A移动到位置B
验证: 检查X不在A且X在B
```

### 5.2 位置约束 (Location Constraint)
```
格式: "X must be on/in A"
含义: 对象X应该在位置A
验证: 检查X与A之间存在相应的空间关系
```

### 5.3 状态约束 (State Constraint)
```
格式: "X must be open/closed/empty/filled"
含义: 对象X应该处于特定状态
验证: 检查X的state属性
```

### 5.4 负约束 (Negative Constraint)
```
格式: "X must not be in A"
含义: 对象X不应该在位置A
验证: 检查X不在A
```

### 5.5 前置条件约束 (Precondition Constraint)
```
格式: "Container must be empty"
含义: 容器必须为空 (用于put_in等操作的前置条件)
验证: 检查容器内是否有其他对象
```

## 6. 示例

```
输入场景:
    - 对象: table, purple cup (empty), blue cup (empty), coffee machine (closed)
    - 关系: purple cup on_top_of table, blue cup inside coffee machine

生成约束:
    1. "purple cup must be moved from table to coffee machine"
    2. "coffee machine must be opened"
    3. "blue cup must be moved from inside coffee machine to table"
    4. "coffee machine must be empty" (前置条件)

验证结果:
    约束1: ✅ SATISFIED (purple cup在table上，不在coffee machine中)
    约束2: ❌ VIOLATED (coffee machine状态是closed，不是open)
    约束3: ❌ VIOLATED (blue cup仍在coffee machine中，未移动到table)
    约束4: ❌ VIOLATED (coffee machine中有blue cup，不满足空容器前置条件)
```

