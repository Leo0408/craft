"""
动作和帧的映射工具
用于建立动作和帧的实际对应关系，而不是假设 1:1 对应
"""
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path


def build_action_frame_mapping(events: List, actions: List, task_info: Dict) -> Dict[int, Tuple[int, int]]:
    """
    建立动作和帧的对应关系
    
    Args:
        events: 事件列表
        actions: 动作列表
        task_info: 任务信息
        
    Returns:
        Dict[int, Tuple[int, int]]: 动作索引 -> (起始帧, 结束帧) 的映射
    """
    if not events or not actions:
        return {}
    
    # 方法1: 尝试从 event metadata 中获取 action 信息
    action_frame_map = {}
    current_action_idx = 0
    
    for frame_idx, event in enumerate(events):
        # 检查 event 中是否有 action 信息
        action_from_event = None
        if hasattr(event, 'metadata') and event.metadata:
            metadata = event.metadata
            # 检查各种可能的 action 字段
            if 'action' in metadata:
                action_from_event = metadata['action']
            elif 'lastAction' in metadata:
                action_from_event = metadata['lastAction']
        
        # 如果找到了 action 信息，尝试匹配
        if action_from_event:
            # 尝试匹配到 actions 列表中的动作
            for i, action in enumerate(actions):
                if str(action) == str(action_from_event) or str(action_from_event) in str(action):
                    if i not in action_frame_map:
                        action_frame_map[i] = {'start': frame_idx, 'end': frame_idx}
                    else:
                        action_frame_map[i]['end'] = frame_idx
                    break
    
    # 方法2: 如果没有从 metadata 中找到，使用平均分配策略
    if not action_frame_map:
        total_frames = len(events)
        total_actions = len(actions)
        
        # 计算每个动作平均占用的帧数
        frames_per_action = total_frames / total_actions
        
        for action_idx in range(total_actions):
            start_frame = int(action_idx * frames_per_action)
            end_frame = int((action_idx + 1) * frames_per_action) - 1
            if action_idx == total_actions - 1:
                end_frame = total_frames - 1  # 最后一个动作到最后一帧
            
            action_frame_map[action_idx] = {'start': start_frame, 'end': end_frame}
    
    # 转换为 (start, end) 元组格式
    result = {}
    for action_idx, frame_range in action_frame_map.items():
        result[action_idx] = (frame_range['start'], frame_range['end'])
    
    return result


def get_action_frames(action_idx: int, action_frame_map: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    """
    获取动作对应的帧范围
    
    Args:
        action_idx: 动作索引
        action_frame_map: 动作-帧映射字典
        
    Returns:
        Tuple[int, int]: (起始帧, 结束帧)
    """
    if action_idx in action_frame_map:
        return action_frame_map[action_idx]
    else:
        # 如果没有映射，返回 None（表示使用默认的 1:1 对应）
        return None


def get_precondition_frame(action_idx: int, action_frame_map: Dict[int, Tuple[int, int]], 
                          events: List) -> int:
    """
    获取 precondition 检查应该使用的帧
    
    Args:
        action_idx: 动作索引
        action_frame_map: 动作-帧映射字典
        events: 事件列表
        
    Returns:
        int: 帧索引
    """
    if action_idx in action_frame_map:
        start_frame, _ = action_frame_map[action_idx]
        # Precondition 检查使用动作开始前的帧
        if start_frame > 0:
            return start_frame - 1
        else:
            return 0  # 第一帧，使用初始场景图
    else:
        # 如果没有映射，使用默认的 1:1 对应
        return action_idx if action_idx > 0 else 0


def get_postcondition_start_frame(action_idx: int, action_frame_map: Dict[int, Tuple[int, int]], 
                                  events: List) -> int:
    """
    获取 postcondition 检查应该使用的起始帧
    
    修改：postcondition 检查从动作实际帧的起点开始，而不是从下一帧开始
    例如：如果动作的实际帧是 20-24，postcondition 检查从 20 开始
    
    Args:
        action_idx: 动作索引
        action_frame_map: 动作-帧映射字典
        events: 事件列表
        
    Returns:
        int: 起始帧索引（动作实际帧的起点）
    """
    if action_idx in action_frame_map:
        start_frame, _ = action_frame_map[action_idx]
        # Postcondition 检查使用动作实际帧的起点（从动作开始帧检查）
        return start_frame
    else:
        # 如果没有映射，使用默认的 1:1 对应
        return action_idx
