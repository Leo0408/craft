"""
详细分析 Step 3-5 中两个实例的：
1. 每个动作具体在几帧
2. 用到的动作和 pre/post 约束
3. 关键帧中的物体情况
4. 失败检测检测到的帧和情况
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def load_task_data(data_path: str) -> Dict:
    """加载任务数据"""
    data_dir = Path(data_path)
    
    # 加载 task.json
    task_json_path = data_dir / "task.json"
    if not task_json_path.exists():
        raise FileNotFoundError(f"task.json not found: {task_json_path}")
    
    with open(task_json_path, 'r') as f:
        task_info = json.load(f)
    
    # 加载 events
    events_dir = data_dir / "events"
    if not events_dir.exists():
        raise FileNotFoundError(f"events directory not found: {events_dir}")
    
    events = []
    event_files = sorted([f for f in os.listdir(events_dir) if f.endswith('.pickle')])
    
    for event_file in event_files:
        event_path = events_dir / event_file
        try:
            import pickle
            with open(event_path, 'rb') as f:
                event = pickle.load(f)
                events.append(event)
        except Exception as e:
            print(f"⚠️ 加载 {event_file} 失败: {e}")
            continue
    
    return {
        'task_info': task_info,
        'events': events
    }

def analyze_instance(instance_path: str, instance_name: str):
    """分析单个实例"""
    print("="*100)
    print(f"📊 分析实例: {instance_name}")
    print("="*100)
    print()
    
    # 加载数据
    try:
        data = load_task_data(instance_path)
        task_info = data['task_info']
        events = data['events']
        actions = task_info.get('actions', [])
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        return
    
    print(f"✅ 数据加载成功:")
    print(f"   - 动作数量: {len(actions)}")
    print(f"   - 事件数量: {len(events)}")
    print()
    
    # 1. 显示每个动作对应的帧
    print("="*100)
    print("1️⃣ 动作和帧的对应关系")
    print("="*100)
    print()
    
    print(f"{'动作索引':<8} {'动作':<40} {'对应帧':<10} {'帧说明':<30}")
    print("-"*100)
    
    for action_idx, action in enumerate(actions):
        frame_idx = action_idx  # 通常 1:1 对应
        frame_desc = f"Frame {frame_idx} (动作执行前)"
        print(f"{action_idx:<8} {str(action):<40} {frame_idx:<10} {frame_desc:<30}")
    
    print()
    
    # 2. 显示每个动作的 pre/post 约束（需要运行 Step 3-4）
    print("="*100)
    print("2️⃣ 每个动作的 Pre/Post 约束")
    print("="*100)
    print()
    print("⚠️ 注意：需要运行 Step 3-4 来生成约束")
    print("   这里显示约束的结构和检查逻辑")
    print()
    
    print("Precondition 检查逻辑:")
    print("  - 使用帧: events[action_idx] (当前动作对应的帧，动作执行前)")
    print("  - 如果 action_idx == 0: 使用初始场景图")
    print()
    
    print("Postcondition 检查逻辑:")
    print("  - 起始帧: events[action_idx + 1]")
    print("  - 时间窗口大小 K:")
    print("    * toggle 动作: 5 帧")
    print("    * put_in / put_on 动作: 8 帧")
    print("    * pick_up 动作: 3 帧")
    print("    * 默认: 5 帧")
    print("  - 结束帧: min(action_idx + 1 + K, len(events))")
    print()
    
    # 3. 显示关键帧中的物体情况
    print("="*100)
    print("3️⃣ 关键帧中的物体情况")
    print("="*100)
    print()
    
    # 显示前几个关键帧的物体情况
    for action_idx, action in enumerate(actions[:min(5, len(actions))]):
        if action_idx < len(events):
            event = events[action_idx]
            print(f"📍 Frame {action_idx} - Action {action_idx + 1}: {action}")
            print("-"*100)
            
            # 提取物体信息
            if isinstance(event, dict):
                # 尝试提取物体列表
                objects = event.get('objects', [])
                if objects:
                    print(f"   物体数量: {len(objects)}")
                    # 显示前5个物体
                    for i, obj in enumerate(objects[:5]):
                        obj_type = obj.get('objectType', 'Unknown')
                        obj_id = obj.get('objectId', 'Unknown')
                        print(f"     {i+1}. {obj_type} ({obj_id[:30]}...)")
                    if len(objects) > 5:
                        print(f"     ... 还有 {len(objects) - 5} 个物体")
                else:
                    print("   ⚠️ 无法提取物体信息（事件格式可能不同）")
            else:
                print("   ⚠️ 事件不是字典格式")
            print()
    
    # 4. 显示失败检测检测到的帧（需要运行 Step 5）
    print("="*100)
    print("4️⃣ 失败检测检测到的帧和情况")
    print("="*100)
    print()
    print("⚠️ 注意：需要运行 Step 5 来执行失败检测")
    print("   这里显示失败检测的逻辑和可能检测到的帧")
    print()
    
    print("失败检测逻辑:")
    print("  - Precondition 违反:")
    print("    * 检查帧: events[action_idx] (动作执行前)")
    print("    * 如果违反，记录: step=action_idx+1, frame=action_idx")
    print()
    print("  - Postcondition 违反:")
    print("    * 检查帧: events[action_idx + 1] 到 events[action_idx + 1 + K] (时间窗口)")
    print("    * 如果窗口内所有帧都不满足，记录: step=action_idx+1, frame=end_frame-1")
    print("    * 记录时间窗口: [start_frame, end_frame-1]")
    print()
    
    # 显示每个动作可能检测到的帧
    print("每个动作可能检测到的帧:")
    print(f"{'动作索引':<8} {'动作':<40} {'Pre帧':<10} {'Post帧范围':<20}")
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
        
        print(f"{action_idx:<8} {str(action):<40} {pre_frame:<10} {post_range:<20}")
    
    print()
    print("="*100)

def main():
    """主函数"""
    print("="*100)
    print("Step 3-5 详细分析工具")
    print("="*100)
    print()
    
    # 定义两个实例
    sim_data_root = "../reflect/reflect_dataset/sim_data"
    
    instances = [
        ("makeCoffee/makeCoffee-1", "makeCoffee-1"),
        ("boilWater/boilWater-1", "boilWater-1"),
    ]
    
    for instance_path, instance_name in instances:
        full_path = os.path.join(sim_data_root, instance_path)
        if os.path.exists(full_path):
            analyze_instance(full_path, instance_name)
            print()
        else:
            print(f"⚠️ 实例路径不存在: {full_path}")
            print()

if __name__ == "__main__":
    main()
