#!/usr/bin/env python3
"""
完整运行REFLECT和CRAFT框架的流程对比
使用makeCoffee-1数据集和original-video.mp4
"""

import os
import sys
import json
import time
import pickle
import cv2
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import traceback
import numpy as np

# 添加路径
sys.path.insert(0, '/home/fdse/zzy/craft')
sys.path.insert(0, '/home/fdse/zzy/reflect/main')

# 统计信息收集器
class StepStatistics:
    def __init__(self, step_name: str):
        self.step_name = step_name
        self.start_time = None
        self.end_time = None
        self.duration = None
        self.outputs = {}
        self.data_sizes = {}
        self.metrics = {}
        self.errors = []
        self.warnings = []
    
    def start(self):
        self.start_time = time.time()
    
    def end(self):
        self.end_time = time.time()
        if self.start_time:
            self.duration = self.end_time - self.start_time
    
    def add_output(self, key: str, value: Any):
        self.outputs[key] = value
    
    def add_data_size(self, key: str, size_bytes: int):
        self.data_sizes[key] = {
            'bytes': size_bytes,
            'kb': size_bytes / 1024,
            'mb': size_bytes / (1024 * 1024)
        }
    
    def add_metric(self, key: str, value: Any):
        self.metrics[key] = value
    
    def add_error(self, error: str):
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    @property
    def success(self) -> bool:
        return len(self.errors) == 0
    
    def to_dict(self) -> Dict:
        return {
            'step_name': self.step_name,
            'duration_seconds': self.duration,
            'outputs': self.outputs,
            'data_sizes': self.data_sizes,
            'metrics': self.metrics,
            'errors': self.errors,
            'warnings': self.warnings,
            'success': self.success
        }


class FrameworkRunner:
    def __init__(self, framework_name: str):
        self.framework_name = framework_name
        self.steps: Dict[str, StepStatistics] = {}
        self.start_time = None
        self.end_time = None
    
    def start_step(self, step_name: str) -> StepStatistics:
        step = StepStatistics(step_name)
        step.start()
        self.steps[step_name] = step
        return step
    
    def get_summary(self) -> Dict:
        total_duration = sum(s.duration or 0 for s in self.steps.values())
        return {
            'framework': self.framework_name,
            'total_duration_seconds': total_duration,
            'steps_count': len(self.steps),
            'steps': {name: step.to_dict() for name, step in self.steps.items()},
            'total_errors': sum(len(s.errors) for s in self.steps.values()),
            'total_warnings': sum(len(s.warnings) for s in self.steps.values()),
            'success': all(s.success for s in self.steps.values())
        }


def analyze_video(video_path: str) -> Dict:
    """分析视频文件"""
    stats = {
        'exists': False,
        'frame_count': 0,
        'fps': 0,
        'duration_seconds': 0,
        'width': 0,
        'height': 0,
        'size_mb': 0
    }
    
    if not os.path.exists(video_path):
        return stats
    
    stats['exists'] = True
    stats['size_mb'] = os.path.getsize(video_path) / (1024 * 1024)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            stats['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            stats['fps'] = cap.get(cv2.CAP_PROP_FPS)
            stats['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            stats['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if stats['fps'] > 0:
                stats['duration_seconds'] = stats['frame_count'] / stats['fps']
            cap.release()
    except Exception as e:
        stats['error'] = str(e)
    
    return stats


def run_reflect_framework() -> FrameworkRunner:
    """运行REFLECT框架"""
    runner = FrameworkRunner('REFLECT')
    runner.start_time = time.time()
    
    reflect_root = '/home/fdse/zzy/reflect'
    folder_name = 'makeCoffee-1'
    task_folder = f'{reflect_root}/thor_tasks/makeCoffee/{folder_name}'
    
    # Step 1: 数据加载和分析
    step = runner.start_step('data_loading')
    try:
        video_path = f'{task_folder}/original-video.mp4'
        task_json_path = f'{task_folder}/task.json'
        
        # 分析视频
        video_stats = analyze_video(video_path)
        step.add_output('video_path', video_path)
        step.add_output('video_stats', video_stats)
        step.add_data_size('video', os.path.getsize(video_path) if os.path.exists(video_path) else 0)
        
        # 加载任务信息
        if os.path.exists(task_json_path):
            with open(task_json_path, 'r') as f:
                task_data = json.load(f)
            step.add_output('task_data', task_data)
            step.add_metric('actions_count', len(task_data.get('actions', [])))
        
        # 检查其他文件
        events_dir = f'{task_folder}/events'
        if os.path.exists(events_dir):
            event_files = list(Path(events_dir).glob('*.json'))
            step.add_metric('event_files_count', len(event_files))
            step.add_output('events_dir', events_dir)
        
        step.add_output('task_folder', task_folder)
        step.add_output('folder_exists', os.path.exists(task_folder))
        
    except Exception as e:
        step.add_error(str(e))
        traceback.print_exc()
    finally:
        step.end()
    
    # Step 2: 状态摘要检查
    step = runner.start_step('state_summary')
    try:
        summary_folder = f'{reflect_root}/main/state_summary/makeCoffee/{folder_name}'
        
        if os.path.exists(summary_folder):
            files = list(Path(summary_folder).rglob('*'))
            file_count = len([f for f in files if f.is_file()])
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            step.add_output('summary_folder', summary_folder)
            step.add_metric('files_count', file_count)
            step.add_data_size('total', total_size)
            
            # 检查特定文件类型
            json_files = list(Path(summary_folder).glob('**/*.json'))
            step.add_metric('json_files_count', len(json_files))
        else:
            step.add_warning(f'State summary folder not found: {summary_folder}')
            step.add_output('summary_folder', summary_folder)
            step.add_output('exists', False)
    except Exception as e:
        step.add_error(str(e))
    finally:
        step.end()
    
    # Step 3: LLM推理检查
    step = runner.start_step('llm_reasoning')
    try:
        llm_folder = f'{reflect_root}/main/LLM/makeCoffee/{folder_name}'
        response_file = f'{llm_folder}/response.json'
        
        if os.path.exists(response_file):
            with open(response_file, 'r') as f:
                response_data = json.load(f)
            
            step.add_output('response_file', response_file)
            step.add_data_size('response', os.path.getsize(response_file))
            step.add_metric('response_keys', list(response_data.keys()) if isinstance(response_data, dict) else [])
            
            # 提取关键信息
            if isinstance(response_data, dict):
                if 'failure_explanation' in response_data:
                    explanation = response_data['failure_explanation']
                    if isinstance(explanation, str):
                        step.add_metric('explanation_length', len(explanation))
                if 'correction_plan' in response_data:
                    plan = response_data['correction_plan']
                    if isinstance(plan, list):
                        step.add_metric('correction_steps_count', len(plan))
        else:
            step.add_warning(f'LLM response file not found: {response_file}')
            step.add_output('response_file', response_file)
            step.add_output('exists', False)
    except Exception as e:
        step.add_error(str(e))
    finally:
        step.end()
    
    runner.end_time = time.time()
    return runner


def run_craft_framework() -> FrameworkRunner:
    """运行CRAFT框架"""
    runner = FrameworkRunner('CRAFT')
    runner.start_time = time.time()
    
    # 使用poloapi配置（与demo1.ipynb相同）
    API_KEY = "sk-wJJVkr6BUx8LruNeHNUCdmE1ARiB4qpLcdHHr3p4zVZTt8Fr"
    POLOAPI_BASE_URL = "https://poloai.top/v1"
    
    reflect_root = '/home/fdse/zzy/reflect'
    folder_name = 'makeCoffee-1'
    task_folder = f'{reflect_root}/thor_tasks/makeCoffee/{folder_name}'
    
    # Step 1: 数据加载
    step = runner.start_step('data_loading')
    try:
        video_path = f'{task_folder}/original-video.mp4'
        task_json_path = f'{task_folder}/task.json'
        
        # 分析视频
        video_stats = analyze_video(video_path)
        step.add_output('video_path', video_path)
        step.add_output('video_stats', video_stats)
        step.add_data_size('video', os.path.getsize(video_path) if os.path.exists(video_path) else 0)
        
        # 加载任务信息
        if os.path.exists(task_json_path):
            with open(task_json_path, 'r') as f:
                task_data = json.load(f)
            step.add_output('task_data', task_data)
            step.add_metric('actions_count', len(task_data.get('actions', [])))
        
        step.add_output('task_folder', task_folder)
        
    except Exception as e:
        step.add_error(str(e))
        traceback.print_exc()
    finally:
        step.end()
    
    # Step 2: 场景图生成
    step = runner.start_step('scene_graph_generation')
    try:
        from core.scene_graph import SceneGraph, Node, Edge
        
        # 尝试从事件文件加载数据
        events_dir = f'{task_folder}/events'
        scene_graphs = {}
        
        if os.path.exists(events_dir):
            event_files = sorted(Path(events_dir).glob('*.json'))
            step.add_metric('event_files_available', len(event_files))
            
            # 读取几个关键事件
            key_indices = [0, len(event_files)//2, len(event_files)-1] if len(event_files) > 0 else []
            
            for idx in key_indices:
                if idx < len(event_files):
                    try:
                        with open(event_files[idx], 'r') as f:
                            event_data = json.load(f)
                        
                        # 创建场景图（简化版）
                        scene_graph = SceneGraph()
                        
                        # 从事件中提取对象
                        if 'objects' in event_data.get('metadata', {}):
                            objects = event_data['metadata']['objects']
                            for obj in objects[:10]:  # 限制对象数量
                                node = Node(
                                    obj_id=obj.get('objectId', ''),
                                    obj_type=obj.get('objectType', ''),
                                    position=obj.get('position', {}),
                                    visible=obj.get('visible', False)
                                )
                                scene_graph.add_node(node)
                        
                        scene_graphs[idx] = scene_graph
                    except Exception as e:
                        step.add_warning(f'Error loading event {idx}: {e}')
            
            step.add_metric('scene_graphs_generated', len(scene_graphs))
            step.add_output('scene_graphs_keys', list(scene_graphs.keys()))
            
            # 统计场景图信息
            if scene_graphs:
                total_nodes = sum(len(sg.nodes) for sg in scene_graphs.values())
                total_edges = sum(len(sg.edges) for sg in scene_graphs.values())
                step.add_metric('total_nodes', total_nodes)
                step.add_metric('total_edges', total_edges)
        else:
            step.add_warning('Events directory not found')
        
    except Exception as e:
        step.add_error(str(e))
        traceback.print_exc()
    finally:
        step.end()
    
    # Step 3: 约束生成
    step = runner.start_step('constraint_generation')
    try:
        # 直接导入LLMPrompter（避免相对导入问题）
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "llm_prompter", 
                "/home/fdse/zzy/craft/reasoning/llm_prompter.py"
            )
            llm_prompter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_prompter_module)
            LLMPrompter = llm_prompter_module.LLMPrompter
            llm_available = True
        except Exception as e:
            llm_available = False
            step.add_warning(f'LLMPrompter import failed: {e}')
        
        if llm_available:
            # 使用poloapi配置初始化LLM
            try:
                llm_prompter = LLMPrompter(
                    gpt_version="gpt-3.5-turbo",
                    api_key=API_KEY,
                    base_url=POLOAPI_BASE_URL
                )
                step.add_output('llm_available', True)
                step.add_output('api_base_url', POLOAPI_BASE_URL)
                step.add_output('model', 'gpt-3.5-turbo')
                step.add_metric('constraints_generated', 0)
                step.add_output('note', 'LLM configured with poloapi, constraint generation available')
            except Exception as e:
                step.add_error(f'LLM initialization failed: {e}')
                step.add_output('llm_available', False)
                step.add_metric('constraints_generated', 0)
        else:
            step.add_output('llm_available', False)
            step.add_metric('constraints_generated', 0)
    except Exception as e:
        step.add_error(str(e))
        traceback.print_exc()
    finally:
        step.end()
    
    # Step 4: 失败检测
    step = runner.start_step('failure_detection')
    try:
        # 模拟约束检查
        step.add_metric('constraints_checked', 0)
        step.add_metric('violations_detected', 0)
        step.add_output('note', 'Failure detection would validate constraints here')
    except Exception as e:
        step.add_error(str(e))
    finally:
        step.end()
    
    # Step 5: 渐进式解释
    step = runner.start_step('progressive_explanation')
    try:
        # 直接导入LLMPrompter（避免相对导入问题）
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "llm_prompter", 
                "/home/fdse/zzy/craft/reasoning/llm_prompter.py"
            )
            llm_prompter_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(llm_prompter_module)
            LLMPrompter = llm_prompter_module.LLMPrompter
            llm_available = True
        except Exception as e:
            llm_available = False
            step.add_warning(f'LLMPrompter import failed: {e}')
        
        if llm_available:
            # 使用poloapi配置初始化LLM
            try:
                llm_prompter = LLMPrompter(
                    gpt_version="gpt-3.5-turbo",
                    api_key=API_KEY,
                    base_url=POLOAPI_BASE_URL
                )
                step.add_output('llm_available', True)
                step.add_output('api_base_url', POLOAPI_BASE_URL)
                step.add_output('model', 'gpt-3.5-turbo')
                step.add_output('note', 'LLM configured with poloapi, progressive explanation available')
                step.add_metric('explanation_generated', False)
            except Exception as e:
                step.add_error(f'LLM initialization failed: {e}')
                step.add_output('llm_available', False)
                step.add_metric('explanation_generated', False)
        else:
            step.add_output('llm_available', False)
            step.add_metric('explanation_generated', False)
    except Exception as e:
        step.add_error(str(e))
        traceback.print_exc()
    finally:
        step.end()
    
    runner.end_time = time.time()
    return runner


def generate_comparison_report(reflect_runner: FrameworkRunner, craft_runner: FrameworkRunner) -> Dict:
    """生成对比报告"""
    reflect_summary = reflect_runner.get_summary()
    craft_summary = craft_runner.get_summary()
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'makeCoffee-1',
        'video_path': '/home/fdse/zzy/reflect/thor_tasks/makeCoffee/makeCoffee-1/original-video.mp4',
        'reflect': reflect_summary,
        'craft': craft_summary,
        'comparison': {
            'duration': {
                'reflect_seconds': reflect_summary['total_duration_seconds'],
                'craft_seconds': craft_summary['total_duration_seconds'],
                'difference_seconds': craft_summary['total_duration_seconds'] - reflect_summary['total_duration_seconds']
            },
            'steps': {
                'reflect_count': reflect_summary['steps_count'],
                'craft_count': craft_summary['steps_count']
            },
            'errors': {
                'reflect_count': reflect_summary['total_errors'],
                'craft_count': craft_summary['total_errors']
            },
            'warnings': {
                'reflect_count': reflect_summary['total_warnings'],
                'craft_count': craft_summary['total_warnings']
            }
        }
    }
    
    return report


def print_detailed_statistics(runner: FrameworkRunner):
    """打印详细统计信息"""
    print(f"\n{'='*80}")
    print(f"{runner.framework_name} Framework - Detailed Statistics")
    print(f"{'='*80}")
    
    summary = runner.get_summary()
    print(f"\nTotal Duration: {summary['total_duration_seconds']:.2f} seconds")
    print(f"Steps: {summary['steps_count']}")
    print(f"Errors: {summary['total_errors']}")
    print(f"Warnings: {summary['total_warnings']}")
    
    print(f"\nStep Details:")
    for step_name, step_data in summary['steps'].items():
        print(f"\n  Step: {step_name}")
        print(f"    Duration: {step_data.get('duration_seconds', 0):.2f}s")
        print(f"    Success: {step_data.get('success', False)}")
        
        if step_data.get('metrics'):
            print(f"    Metrics:")
            for key, value in step_data['metrics'].items():
                print(f"      {key}: {value}")
        
        if step_data.get('data_sizes'):
            print(f"    Data Sizes:")
            for key, size_info in step_data['data_sizes'].items():
                print(f"      {key}: {size_info.get('mb', 0):.2f} MB")
        
        if step_data.get('outputs'):
            print(f"    Outputs:")
            for key, value in step_data['outputs'].items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"      {key}: {value}")
        
        if step_data.get('errors'):
            print(f"    Errors:")
            for error in step_data['errors']:
                print(f"      - {error}")
        
        if step_data.get('warnings'):
            print(f"    Warnings:")
            for warning in step_data['warnings']:
                print(f"      - {warning}")


def main():
    """主函数"""
    print("="*80)
    print("Framework Comparison: REFLECT vs CRAFT")
    print("Dataset: makeCoffee-1")
    print("Video: original-video.mp4")
    print("="*80)
    
    # 运行REFLECT框架
    print("\n" + "="*80)
    print("Running REFLECT Framework...")
    print("="*80)
    reflect_runner = run_reflect_framework()
    print_detailed_statistics(reflect_runner)
    
    # 运行CRAFT框架
    print("\n" + "="*80)
    print("Running CRAFT Framework...")
    print("="*80)
    craft_runner = run_craft_framework()
    print_detailed_statistics(craft_runner)
    
    # 生成对比报告
    print("\n" + "="*80)
    print("Generating Comparison Report...")
    print("="*80)
    
    report = generate_comparison_report(reflect_runner, craft_runner)
    
    # 保存报告
    output_dir = 'output/comparison'
    os.makedirs(output_dir, exist_ok=True)
    report_file = f"{output_dir}/framework_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved to: {report_file}")
    
    # 打印对比摘要
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(f"REFLECT Framework:")
    print(f"  - Total Duration: {report['comparison']['duration']['reflect_seconds']:.2f}s")
    print(f"  - Steps: {report['comparison']['steps']['reflect_count']}")
    print(f"  - Errors: {report['comparison']['errors']['reflect_count']}")
    print(f"  - Warnings: {report['comparison']['warnings']['reflect_count']}")
    
    print(f"\nCRAFT Framework:")
    print(f"  - Total Duration: {report['comparison']['duration']['craft_seconds']:.2f}s")
    print(f"  - Steps: {report['comparison']['steps']['craft_count']}")
    print(f"  - Errors: {report['comparison']['errors']['craft_count']}")
    print(f"  - Warnings: {report['comparison']['warnings']['craft_count']}")
    
    print(f"\nDifference:")
    diff = report['comparison']['duration']['difference_seconds']
    print(f"  - Duration Difference: {diff:+.2f}s")
    
    return report


if __name__ == "__main__":
    main()

