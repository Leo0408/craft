"""
详细分析 Step 3-5 中两个实例的完整信息：
1. 每个动作具体在几帧
2. 用到的动作和 pre/post 约束（实际生成）
3. 关键帧中的物体情况
4. 失败检测检测到的帧和情况（实际检测结果）
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

def extract_objects_from_event(event):
    """从事件中提取物体信息"""
    objects = []
    if isinstance(event, dict):
        # 尝试多种可能的键
        if 'objects' in event:
            objects = event['objects']
        elif 'metadata' in event and 'objects' in event['metadata']:
            objects = event['metadata']['objects']
        elif 'object_list' in event:
            objects = event['object_list']
    return objects

def analyze_instance_detailed(instance_path: str, instance_name: str):
    """详细分析单个实例，包括实际运行 CRAFT 流程"""
    print("="*100)
    print(f"📊 详细分析实例: {instance_name}")
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
    
    print(f"{'动作索引':<8} {'动作':<45} {'对应帧':<10} {'帧说明':<30}")
    print("-"*100)
    
    for action_idx, action in enumerate(actions):
        frame_idx = action_idx  # 通常 1:1 对应
        frame_desc = f"Frame {frame_idx} (动作执行前)"
        print(f"{action_idx:<8} {str(action):<45} {frame_idx:<10} {frame_desc:<30}")
    
    print()
    
    # 2. 尝试生成约束（需要导入相关模块）
    print("="*100)
    print("2️⃣ 每个动作的 Pre/Post 约束（实际生成）")
    print("="*100)
    print()
    
    try:
        from craft.reasoning.constraint_generator import ConstraintGenerator
        from craft.reasoning.llm_prompter import LLMPrompter
        
        # 尝试获取 API Key
        api_key = os.environ.get('POLOAPI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            api_key = globals().get('API_KEY', "sk-UQFLOuq6vZEiBEeNVSRiRkgNErTitcdVQjAMwrlH08f870H2")
        
        llm_prompter = LLMPrompter(
            gpt_version="gpt-4",
            api_key=api_key,
            base_url="https://poloai.top/v1"
        )
        constraint_generator = ConstraintGenerator(llm_prompter)
        
        print("✅ 约束生成器初始化成功")
        print()
        
        # 为每个动作生成约束
        all_constraints = []
        for action_idx, action in enumerate(actions):
            try:
                action_constraints = constraint_generator.generate_constraints_for_action(
                    action=action,
                    action_index=action_idx
                )
                if action_constraints:
                    all_constraints.extend(action_constraints)
                    
                    # 显示该动作的约束
                    pre_constraints = [c for c in action_constraints if 'pre' in c.get('type', '').lower()]
                    post_constraints = [c for c in action_constraints if 'post' in c.get('type', '').lower()]
                    
                    print(f"📍 Action {action_idx + 1}: {action}")
                    print(f"   Pre 约束: {len(pre_constraints)} 个")
                    for i, c in enumerate(pre_constraints[:3], 1):  # 只显示前3个
                        desc = c.get('description', c.get('template', 'N/A'))
                        print(f"      {i}. {desc[:70]}")
                    if len(pre_constraints) > 3:
                        print(f"      ... 还有 {len(pre_constraints) - 3} 个 Pre 约束")
                    
                    print(f"   Post 约束: {len(post_constraints)} 个")
                    for i, c in enumerate(post_constraints[:3], 1):  # 只显示前3个
                        desc = c.get('description', c.get('template', 'N/A'))
                        print(f"      {i}. {desc[:70]}")
                    if len(post_constraints) > 3:
                        print(f"      ... 还有 {len(post_constraints) - 3} 个 Post 约束")
                    print()
            except Exception as e:
                print(f"⚠️ Action {action_idx + 1} 约束生成失败: {e}")
                continue
        
        print(f"✅ 总共生成了 {len(all_constraints)} 个约束")
        print()
        
    except Exception as e:
        print(f"⚠️ 无法生成约束: {e}")
        print("   将显示约束检查逻辑")
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
            objects = extract_objects_from_event(event)
            if objects:
                print(f"   物体数量: {len(objects)}")
                # 显示前10个物体
                for i, obj in enumerate(objects[:10]):
                    if isinstance(obj, dict):
                        obj_type = obj.get('objectType', obj.get('type', 'Unknown'))
                        obj_id = obj.get('objectId', obj.get('id', 'Unknown'))
                        print(f"     {i+1}. {obj_type} ({str(obj_id)[:40]}...)")
                    else:
                        print(f"     {i+1}. {str(obj)[:60]}")
                if len(objects) > 10:
                    print(f"     ... 还有 {len(objects) - 10} 个物体")
            else:
                print("   ⚠️ 无法提取物体信息（事件格式可能不同）")
                # 尝试显示事件的其他信息
                if isinstance(event, dict):
                    print(f"   事件键: {list(event.keys())[:10]}")
            print()
    
    # 4. 显示失败检测检测到的帧（需要运行 Step 5）
    print("="*100)
    print("4️⃣ 失败检测检测到的帧和情况")
    print("="*100)
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
    print("⚠️ 注意：要查看实际检测到的失败，需要运行 Step 5 的完整失败检测流程")
    print("   实际检测结果会显示在 Step 5 的输出中")
    print()
    print("="*100)

def main():
    """主函数"""
    print("="*100)
    print("Step 3-5 详细分析工具（包含实际 CRAFT 流程）")
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
            analyze_instance_detailed(full_path, instance_name)
            print()
        else:
            print(f"⚠️ 实例路径不存在: {full_path}")
            print()

if __name__ == "__main__":
    main()
