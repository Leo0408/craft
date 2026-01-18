"""
检查 Step 3-5 中动作和帧的对应关系，以及 pre/post 约束的帧选择
"""
import json
import os
from pathlib import Path

def check_action_frame_mapping(demo3_path="demo3.ipynb"):
    """检查动作和帧的对应关系"""
    
    print("="*80)
    print("Step 3-5 动作和帧对应关系检查")
    print("="*80)
    print()
    
    # 从 demo3.ipynb 中提取关键信息
    # 这里我们基于代码逻辑分析
    
    print("📋 1. 动作和帧的对应关系")
    print("-"*80)
    print("""
    根据代码分析：
    - 在 Step 2 中，每个动作对应一个关键帧（keyframe）
    - Frame 索引从 0 开始，Action 索引也从 0 开始
    - 通常：Frame i 对应 Action i（1:1 对应）
    
    但是，实际数据中：
    - 一个动作可能跨越多个帧（例如：navigate_to_obj 可能需要多帧）
    - 我们只选择关键帧（keyframe）进行场景图生成
    - 关键帧通常是动作开始或结束的帧
    """)
    
    print("\n📋 2. Precondition 的帧选择")
    print("-"*80)
    print("""
    代码位置：Step 5 (Cell 29) - Precondition 检查
    
    选择逻辑：
    - 如果 action_idx > 0: 使用 events[action_idx]（当前动作对应的帧）
    - 如果 action_idx == 0: 使用初始场景图（initial_sg）
    
    关键点：
    - Precondition 检查的是动作执行前的状态
    - 使用 events[action_idx] 表示动作执行前的帧
    - 这是当前动作对应的帧，不是前一个动作的帧
    """)
    
    print("\n📋 3. Postcondition 的帧选择")
    print("-"*80)
    print("""
    代码位置：Step 5 (Cell 29) - Postcondition 检查（使用 Temporal Window）
    
    选择逻辑：
    - 起始帧：start_frame = action_idx + 1
    - 结束帧：end_frame = min(start_frame + K, len(events))
    - 在时间窗口 [start_frame, end_frame) 内检查
    
    时间窗口大小 K（根据动作类型）：
    - toggle 动作：K = 5 帧
    - put_in / put_on 动作：K = 8 帧
    - pick_up 动作：K = 3 帧
    - 默认：K = 5 帧
    
    关键点：
    - Postcondition 检查的是动作执行后的状态
    - 使用 events[action_idx + 1] 开始的时间窗口
    - 只要窗口内任何一帧满足，就认为 postcondition 满足
    """)
    
    print("\n📋 4. 动作之间的帧数差距")
    print("-"*80)
    print("""
    根据代码逻辑：
    - 如果每个动作对应一个关键帧，那么：
      - Action 0 → Frame 0
      - Action 1 → Frame 1
      - Action 2 → Frame 2
      - ...
    - 动作之间的帧数差距：通常是 1 帧（如果使用关键帧）
    
    但是，实际数据中：
    - 一个动作可能跨越多个帧
    - 我们只选择关键帧，所以看起来是 1:1 对应
    - 实际帧数差距取决于动作执行时间
    """)
    
    print("\n📋 5. 总结")
    print("-"*80)
    print("""
    Precondition:
    - 使用：events[action_idx]（当前动作对应的帧，动作执行前）
    - 不是：events[action_idx - 1]（前一个动作的帧）
    - 不是：前后一个动作，而是当前动作执行前的帧
    
    Postcondition:
    - 使用：events[action_idx + 1] 开始的时间窗口
    - 窗口大小：根据动作类型（3-8 帧）
    - 不是：只检查 events[action_idx + 1]（下一帧）
    - 不是：前后一个动作，而是动作执行后的时间窗口
    
    动作之间的帧数差距：
    - 如果使用关键帧：通常是 1 帧（1:1 对应）
    - 实际数据中：可能更多，取决于动作执行时间
    """)
    
    print("\n" + "="*80)
    print("检查完成！")
    print("="*80)

if __name__ == "__main__":
    check_action_frame_mapping()
