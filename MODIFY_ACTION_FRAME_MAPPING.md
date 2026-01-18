# 修改动作-帧映射指南

## 目标
修改代码以使用实际帧范围，而不是假设 1:1 对应（Frame i 对应 Action i）。

## 修改步骤

### 1. 修改 `load_task_data` 函数（Cell 1）

在 `return` 语句之前添加：

```python
# 建立动作-帧映射（使用实际帧范围，而不是 1:1 对应）
try:
    from core.action_frame_mapper import build_action_frame_mapping
    action_frame_map = build_action_frame_mapping(events, actions, task_info)
    print(f"✅ 建立了动作-帧映射: {len(action_frame_map)} 个动作")
    if action_frame_map:
        # 显示前几个映射
        for i, (action_idx, (start, end)) in enumerate(list(action_frame_map.items())[:3]):
            action_str = actions[action_idx] if action_idx < len(actions) else 'N/A'
            print(f"   Action {action_idx}: {action_str} -> Frames {start}-{end}")
except Exception as e:
    print(f"⚠️ 建立动作-帧映射失败: {e}，将使用默认的 1:1 对应")
    action_frame_map = {}
```

在 `return` 语句中添加 `action_frame_map`：

```python
return {
    'events': events,
    'task_info': task_info,
    'action_results': action_results,
    'action_frame_map': action_frame_map,  # 添加动作-帧映射
    'data_path': str(data_dir)
}
```

### 2. 修改 Step 5 的 Precondition 检查（Cell 31）

找到以下代码：

```python
# 获取动作执行前的场景图
if action_idx == 0:
    eval_sg = initial_sg
    frame_info = "初始场景图"
elif action_idx < len(events):
    # 使用 events[action_idx] 的场景图（动作执行前）
    from craft.core.enhanced_generate_scene_graph import generate_scene_graph_from_event_enhanced
    eval_sg = generate_scene_graph_from_event_enhanced(
        events[action_idx],
        task_info,
        timestep=action_idx,
        action=action
    )
    eval_sg = add_virtual_robot_node(eval_sg)
    frame_info = f"帧 {action_idx}"
```

修改为：

```python
# 获取动作执行前的场景图（使用实际帧范围）
from core.action_frame_mapper import get_precondition_frame

# 获取 action_frame_map（从 task_data 中）
action_frame_map = task_data.get('action_frame_map', {}) if 'task_data' in globals() else {}

if action_idx == 0:
    eval_sg = initial_sg
    frame_info = "初始场景图"
else:
    # 使用实际帧范围获取 precondition 检查帧
    pre_frame = get_precondition_frame(action_idx, action_frame_map, events)
    if pre_frame < len(events):
        from craft.core.enhanced_generate_scene_graph import generate_scene_graph_from_event_enhanced
        eval_sg = generate_scene_graph_from_event_enhanced(
            events[pre_frame],
            task_info,
            timestep=pre_frame,
            action=action
        )
        eval_sg = add_virtual_robot_node(eval_sg)
        frame_info = f"帧 {pre_frame} (动作 {action_idx} 开始前的帧)"
    else:
        eval_sg = initial_sg
        frame_info = "初始场景图（回退）"
```

### 3. 修改 Step 5 的 Postcondition 检查（Cell 31）

找到以下代码：

```python
# Postcondition Temporal Window 检查
# 确定检查的帧范围：从 action_idx + 1 到 action_idx + 1 + K
start_frame = action_idx + 1
end_frame = min(start_frame + temporal_window_size, len(events))
```

修改为：

```python
# Postcondition Temporal Window 检查（使用实际帧范围）
from core.action_frame_mapper import get_postcondition_start_frame

# 获取 action_frame_map（从 task_data 中）
action_frame_map = task_data.get('action_frame_map', {}) if 'task_data' in globals() else {}

# 使用实际帧范围获取 postcondition 检查起始帧
start_frame = get_postcondition_start_frame(action_idx, action_frame_map, events)
end_frame = min(start_frame + temporal_window_size, len(events))
```

### 4. 确保 action_frame_map 在 Step 5 中可用

在 Step 5 的开始部分添加：

```python
# 获取 action_frame_map（如果可用）
action_frame_map = {}
if 'task_data' in globals():
    # 尝试从当前任务数据中获取
    current_task_data = None
    for task_name, data in task_data.items():
        if 'action_frame_map' in data:
            action_frame_map = data['action_frame_map']
            current_task_data = data
            break
    
    # 如果没找到，尝试从 events 和 actions 重新构建
    if not action_frame_map and current_task_data:
        from core.action_frame_mapper import build_action_frame_mapping
        events = current_task_data.get('events', [])
        actions = current_task_data.get('task_info', {}).get('actions', [])
        task_info = current_task_data.get('task_info', {})
        if events and actions:
            action_frame_map = build_action_frame_mapping(events, actions, task_info)
            print(f"✅ 在 Step 5 中重新建立了动作-帧映射: {len(action_frame_map)} 个动作")
```

## 验证

修改后，运行代码应该看到：
1. 在数据加载时显示动作-帧映射信息
2. Precondition 检查使用动作开始前的实际帧
3. Postcondition 检查使用动作结束后的下一帧（而不是下一个动作的帧）

## 注意事项

- 如果 `action_frame_map` 为空，代码会回退到默认的 1:1 对应
- 确保 `core/action_frame_mapper.py` 文件已创建并可用
- 修改后需要重新运行 Step 1 和 Step 5
