"""
这是一个可以在 demo3.ipynb 中运行的 cell，用于详细分析 Step 3-5
显示两个实例的完整信息，包括实际运行 CRAFT 流程的结果
"""
# 这个代码应该作为 notebook cell 运行，因为它需要访问 notebook 中的变量和函数

print("="*100)
print("Step 3-5 详细分析：两个实例的完整信息")
print("="*100)
print()

# 定义两个实例
instances = [
    ("makeCoffee", "makeCoffee-1"),
    ("boilWater", "boilWater-1"),
]

for task_name, instance_name in instances:
    print("\n" + "="*100)
    print(f"📊 分析实例: {task_name}/{instance_name}")
    print("="*100)
    print()
    
    # 构建数据集路径
    dataset_path = f"../reflect/reflect_dataset/sim_data/{task_name}/{instance_name}"
    
    # 检查路径是否存在
    import os
    if not os.path.exists(dataset_path):
        print(f"⚠️ 数据集路径不存在: {dataset_path}")
        continue
    
    # 加载数据
    try:
        # 使用 notebook 中的 load_task_data 函数
        if 'load_task_data' in globals():
            task_data_dict = load_task_data(dataset_path)
            events = task_data_dict.get('events', [])
            
            # 加载 task.json
            import json
            with open(os.path.join(dataset_path, 'task.json'), 'r') as f:
                task_info = json.load(f)
            
            actions = task_info.get('actions', [])
        else:
            print("⚠️ load_task_data 函数不可用，请先运行 Step 1")
            continue
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        continue
    
    print(f"✅ 数据加载成功:")
    print(f"   - 动作数量: {len(actions)}")
    print(f"   - 事件数量: {len(events)}")
    print()
    
    # 1. 显示每个动作对应的帧
    print("="*100)
    print("1️⃣ 动作和帧的对应关系")
    print("="*100)
    print()
    
    print(f"{'动作索引':<8} {'动作':<45} {'对应帧':<10} {'帧说明':<30}")
    print("-"*100)
    
    for action_idx, action in enumerate(actions):
        frame_idx = action_idx
        frame_desc = f"Frame {frame_idx} (动作执行前)"
        print(f"{action_idx:<8} {str(action):<45} {frame_idx:<10} {frame_desc:<30}")
    
    print()
    
    # 2. 显示每个动作的约束（如果已经生成）
    print("="*100)
    print("2️⃣ 每个动作的 Pre/Post 约束")
    print("="*100)
    print()
    
    # 尝试从 notebook 中获取约束
    if 'task_data' in globals() and task_name in task_data:
        data = task_data[task_name]
        if 'constraints' in data:
            constraints = data['constraints']
            
            # 按动作分组约束
            constraints_by_action = {}
            for constraint in constraints:
                action_idx = constraint.get('action_index')
                if action_idx is not None:
                    if action_idx not in constraints_by_action:
                        constraints_by_action[action_idx] = {'pre': [], 'post': []}
                    
                    constraint_type = constraint.get('type', '').lower()
                    if 'pre' in constraint_type:
                        constraints_by_action[action_idx]['pre'].append(constraint)
                    elif 'post' in constraint_type:
                        constraints_by_action[action_idx]['post'].append(constraint)
            
            # 显示每个动作的约束
            for action_idx, action in enumerate(actions):
                if action_idx in constraints_by_action:
                    pre_constraints = constraints_by_action[action_idx]['pre']
                    post_constraints = constraints_by_action[action_idx]['post']
                    
                    if pre_constraints or post_constraints:
                        print(f"📍 Action {action_idx + 1}: {action}")
                        if pre_constraints:
                            print(f"   Pre 约束 ({len(pre_constraints)} 个):")
                            for i, c in enumerate(pre_constraints[:3], 1):
                                desc = c.get('description', c.get('template', 'N/A'))
                                print(f"      {i}. {desc[:70]}")
                            if len(pre_constraints) > 3:
                                print(f"      ... 还有 {len(pre_constraints) - 3} 个")
                        if post_constraints:
                            print(f"   Post 约束 ({len(post_constraints)} 个):")
                            for i, c in enumerate(post_constraints[:3], 1):
                                desc = c.get('description', c.get('template', 'N/A'))
                                print(f"      {i}. {desc[:70]}")
                            if len(post_constraints) > 3:
                                print(f"      ... 还有 {len(post_constraints) - 3} 个")
                        print()
        else:
            print("⚠️ 约束未生成，请先运行 Step 3-4")
    else:
        print("⚠️ 约束未生成，请先运行 Step 3-4")
    
    print()
    
    # 3. 显示失败检测检测到的帧（如果已经运行 Step 5）
    print("="*100)
    print("3️⃣ 失败检测检测到的帧和情况")
    print("="*100)
    print()
    
    # 尝试从 notebook 中获取失败检测结果
    if 'violations' in globals() or 'real_errors' in globals():
        violations = globals().get('violations', [])
        real_errors = globals().get('real_errors', [])
        root_violation = globals().get('root_violation', None)
        
        print(f"检测到的违反数量: {len(violations)}")
        print(f"真实错误数量: {len(real_errors)}")
        print()
        
        if real_errors:
            print("真实错误详情:")
            print(f"{'Step':<8} {'Frame':<10} {'类型':<25} {'动作':<45}")
            print("-"*100)
            
            for error in real_errors:
                step = error.get('step', 'N/A')
                frame = error.get('frame', 'N/A')
                failure_type = error.get('failure_type', 'N/A')
                action = error.get('action', 'N/A')
                print(f"{step:<8} {frame:<10} {failure_type[:25]:<25} {str(action)[:45]:<45}")
            
            print()
            
            if root_violation:
                print("根因违反:")
                print(f"  Step: {root_violation.get('step', 'N/A')}")
                print(f"  Frame: {root_violation.get('frame', 'N/A')}")
                print(f"  类型: {root_violation.get('failure_type', 'N/A')}")
                print(f"  动作: {root_violation.get('action', 'N/A')}")
                reason = root_violation.get('reason', 'N/A')
                print(f"  原因: {reason[:80]}...")
        else:
            print("✅ 未检测到真实错误")
    else:
        print("⚠️ 失败检测结果不可用，请先运行 Step 5")
        print()
        print("每个动作可能检测到的帧:")
        print(f"{'动作索引':<8} {'动作':<45} {'Pre帧':<10} {'Post帧范围':<20} {'窗口大小':<10}")
        print("-"*100)
        
        for action_idx, action in enumerate(actions):
            pre_frame = action_idx
            action_lower = str(action).lower()
            
            # 确定时间窗口大小
            if 'toggle' in action_lower:
                K = 5
            elif 'put_in' in action_lower or 'put_on' in action_lower:
                K = 8
            elif 'pick_up' in action_lower:
                K = 3
            else:
                K = 5
            
            post_start = action_idx + 1
            post_end = min(post_start + K, len(events))
            post_range = f"{post_start}-{post_end-1}" if post_end > post_start else "N/A"
            
            print(f"{action_idx:<8} {str(action):<45} {pre_frame:<10} {post_range:<20} {K:<10}")
    
    print()
    print("="*100)

print("\n✅ 分析完成！")
