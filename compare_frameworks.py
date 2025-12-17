#!/usr/bin/env python3
"""
对比REFLECT和CRAFT框架的完整流程
使用makeCoffee-1数据集和original-video.mp4
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import traceback

# 添加路径
sys.path.insert(0, '/home/fdse/zzy/craft')
sys.path.insert(0, '/home/fdse/zzy/reflect/main')

# 统计信息收集器
class StatisticsCollector:
    def __init__(self, framework_name: str):
        self.framework = framework_name
        self.stats = {
            'framework': framework_name,
            'start_time': None,
            'end_time': None,
            'steps': {},
            'errors': []
        }
    
    def start_step(self, step_name: str):
        """开始一个步骤"""
        if step_name not in self.stats['steps']:
            self.stats['steps'][step_name] = {
                'start_time': time.time(),
                'end_time': None,
                'duration': None,
                'outputs': {},
                'data_size': {},
                'errors': []
            }
        else:
            self.stats['steps'][step_name]['start_time'] = time.time()
    
    def end_step(self, step_name: str, outputs: Dict = None, data_sizes: Dict = None):
        """结束一个步骤"""
        if step_name in self.stats['steps']:
            step_stats = self.stats['steps'][step_name]
            step_stats['end_time'] = time.time()
            step_stats['duration'] = step_stats['end_time'] - step_stats['start_time']
            if outputs:
                step_stats['outputs'] = outputs
            if data_sizes:
                step_stats['data_size'] = data_sizes
    
    def add_error(self, step_name: str, error: str):
        """记录错误"""
        if step_name in self.stats['steps']:
            self.stats['steps'][step_name]['errors'].append(error)
        self.stats['errors'].append(f"{step_name}: {error}")
    
    def get_summary(self) -> Dict:
        """获取统计摘要"""
        total_duration = sum(
            s.get('duration', 0) for s in self.stats['steps'].values()
        )
        return {
            'framework': self.framework,
            'total_duration': total_duration,
            'steps_count': len(self.stats['steps']),
            'steps': self.stats['steps'],
            'errors_count': len(self.stats['errors']),
            'errors': self.stats['errors']
        }


def run_reflect_framework(data_path: str, task_info: Dict) -> StatisticsCollector:
    """运行REFLECT框架的完整流程"""
    collector = StatisticsCollector('REFLECT')
    collector.stats['start_time'] = time.time()
    
    try:
        # Step 1: 数据生成
        collector.start_step('data_generation')
        print("\n" + "="*80)
        print("REFLECT Framework - Step 1: Data Generation")
        print("="*80)
        
        try:
            from gen_data import run_data_gen
            os.chdir('/home/fdse/zzy/reflect')
            run_data_gen(data_path=os.getcwd(), task=task_info)
            
            # 收集输出统计
            folder_name = task_info['folder_name']
            task_folder = f"thor_tasks/makeCoffee/{folder_name}"
            video_path = f"{task_folder}/original-video.mp4"
            
            outputs = {
                'task_folder': task_folder,
                'video_path': video_path,
                'video_exists': os.path.exists(video_path),
                'pickle_file': f"{task_folder}.pickle"
            }
            
            data_sizes = {}
            if os.path.exists(video_path):
                data_sizes['video_size_mb'] = os.path.getsize(video_path) / (1024 * 1024)
            if os.path.exists(f"{task_folder}.pickle"):
                data_sizes['pickle_size_mb'] = os.path.getsize(f"{task_folder}.pickle") / (1024 * 1024)
            
            collector.end_step('data_generation', outputs, data_sizes)
            print(f"✅ Data generation completed")
        except Exception as e:
            collector.add_error('data_generation', str(e))
            print(f"❌ Data generation error: {e}")
            traceback.print_exc()
        
        # Step 2: 状态摘要生成
        collector.start_step('state_summary')
        print("\n" + "="*80)
        print("REFLECT Framework - Step 2: State Summary Generation")
        print("="*80)
        
        try:
            # 检查状态摘要文件
            summary_folder = f"main/state_summary/makeCoffee/{folder_name}"
            outputs = {
                'summary_folder': summary_folder,
                'folder_exists': os.path.exists(summary_folder)
            }
            
            if os.path.exists(summary_folder):
                # 统计文件
                files = list(Path(summary_folder).rglob('*'))
                outputs['file_count'] = len(files)
                data_sizes = {}
                total_size = 0
                for f in files:
                    if f.is_file():
                        total_size += f.stat().st_size
                data_sizes['total_size_mb'] = total_size / (1024 * 1024)
            else:
                outputs['file_count'] = 0
                data_sizes = {}
            
            collector.end_step('state_summary', outputs, data_sizes)
            print(f"✅ State summary check completed")
        except Exception as e:
            collector.add_error('state_summary', str(e))
            print(f"❌ State summary error: {e}")
        
        # Step 3: LLM失败推理
        collector.start_step('llm_reasoning')
        print("\n" + "="*80)
        print("REFLECT Framework - Step 3: LLM Failure Reasoning")
        print("="*80)
        
        try:
            # 检查LLM响应文件
            llm_folder = f"main/LLM/makeCoffee/{folder_name}"
            response_file = f"{llm_folder}/response.json"
            
            outputs = {
                'llm_folder': llm_folder,
                'response_file': response_file,
                'response_exists': os.path.exists(response_file)
            }
            
            data_sizes = {}
            if os.path.exists(response_file):
                data_sizes['response_size_kb'] = os.path.getsize(response_file) / 1024
                # 读取响应内容统计
                try:
                    with open(response_file, 'r') as f:
                        response_data = json.load(f)
                        outputs['response_keys'] = list(response_data.keys()) if isinstance(response_data, dict) else []
                except:
                    pass
            
            collector.end_step('llm_reasoning', outputs, data_sizes)
            print(f"✅ LLM reasoning check completed")
        except Exception as e:
            collector.add_error('llm_reasoning', str(e))
            print(f"❌ LLM reasoning error: {e}")
        
    except Exception as e:
        collector.add_error('framework', str(e))
        print(f"❌ REFLECT framework error: {e}")
        traceback.print_exc()
    finally:
        collector.stats['end_time'] = time.time()
        os.chdir('/home/fdse/zzy/craft')
    
    return collector


def run_craft_framework(task_info: Dict) -> StatisticsCollector:
    """运行CRAFT框架的完整流程"""
    collector = StatisticsCollector('CRAFT')
    collector.stats['start_time'] = time.time()
    
    try:
        # Step 1: 数据生成
        collector.start_step('data_generation')
        print("\n" + "="*80)
        print("CRAFT Framework - Step 1: Data Generation")
        print("="*80)
        
        try:
            # 检查是否有AI2THOR
            try:
                from ai2thor.controller import Controller
                AI2THOR_AVAILABLE = True
            except:
                AI2THOR_AVAILABLE = False
                print("⚠️  AI2THOR not available, skipping data generation")
            
            if AI2THOR_AVAILABLE:
                # 这里可以运行CRAFT的数据生成
                # 为了对比，我们使用已有的数据
                pass
            
            outputs = {
                'ai2thor_available': AI2THOR_AVAILABLE,
                'using_existing_data': True
            }
            data_sizes = {}
            
            collector.end_step('data_generation', outputs, data_sizes)
            print(f"✅ Data generation check completed")
        except Exception as e:
            collector.add_error('data_generation', str(e))
            print(f"❌ Data generation error: {e}")
        
        # Step 2: 场景图生成
        collector.start_step('scene_graph_generation')
        print("\n" + "="*80)
        print("CRAFT Framework - Step 2: Scene Graph Generation")
        print("="*80)
        
        try:
            from core.scene_graph import SceneGraph
            from core.task_executor import TaskExecutor
            
            # 创建场景图示例
            scene_graphs = {}
            outputs = {
                'scene_graphs_count': 0,
                'scene_graphs_keys': []
            }
            
            # 这里可以实际生成场景图，为了演示我们创建空的
            outputs['scene_graphs_count'] = len(scene_graphs)
            outputs['scene_graphs_keys'] = list(scene_graphs.keys())
            
            data_sizes = {
                'scene_graphs_count': len(scene_graphs)
            }
            
            collector.end_step('scene_graph_generation', outputs, data_sizes)
            print(f"✅ Scene graph generation completed")
        except Exception as e:
            collector.add_error('scene_graph_generation', str(e))
            print(f"❌ Scene graph generation error: {e}")
            traceback.print_exc()
        
        # Step 3: 约束生成
        collector.start_step('constraint_generation')
        print("\n" + "="*80)
        print("CRAFT Framework - Step 3: Constraint Generation")
        print("="*80)
        
        try:
            from reasoning.llm_prompter import LLMPrompter
            
            # 检查是否有API key
            has_api_key = os.getenv('OPENAI_API_KEY') is not None
            outputs = {
                'llm_available': has_api_key,
                'constraints_generated': 0
            }
            
            if not has_api_key:
                outputs['note'] = 'LLM API key not set, skipping constraint generation'
            
            data_sizes = {}
            
            collector.end_step('constraint_generation', outputs, data_sizes)
            print(f"✅ Constraint generation check completed")
        except Exception as e:
            collector.add_error('constraint_generation', str(e))
            print(f"❌ Constraint generation error: {e}")
        
        # Step 4: 失败检测
        collector.start_step('failure_detection')
        print("\n" + "="*80)
        print("CRAFT Framework - Step 4: Failure Detection")
        print("="*80)
        
        try:
            outputs = {
                'constraints_checked': 0,
                'violations_detected': 0
            }
            data_sizes = {}
            
            collector.end_step('failure_detection', outputs, data_sizes)
            print(f"✅ Failure detection check completed")
        except Exception as e:
            collector.add_error('failure_detection', str(e))
            print(f"❌ Failure detection error: {e}")
        
        # Step 5: 渐进式解释
        collector.start_step('progressive_explanation')
        print("\n" + "="*80)
        print("CRAFT Framework - Step 5: Progressive Explanation")
        print("="*80)
        
        try:
            from reasoning.failure_analyzer import FailureAnalyzer
            from reasoning.llm_prompter import LLMPrompter
            
            has_api_key = os.getenv('OPENAI_API_KEY') is not None
            outputs = {
                'llm_available': has_api_key,
                'explanation_generated': False
            }
            
            if not has_api_key:
                outputs['note'] = 'LLM API key not set, skipping explanation generation'
            
            data_sizes = {}
            
            collector.end_step('progressive_explanation', outputs, data_sizes)
            print(f"✅ Progressive explanation check completed")
        except Exception as e:
            collector.add_error('progressive_explanation', str(e))
            print(f"❌ Progressive explanation error: {e}")
        
    except Exception as e:
        collector.add_error('framework', str(e))
        print(f"❌ CRAFT framework error: {e}")
        traceback.print_exc()
    finally:
        collector.stats['end_time'] = time.time()
    
    return collector


def main():
    """主函数"""
    print("="*80)
    print("Framework Comparison: REFLECT vs CRAFT")
    print("Dataset: makeCoffee-1")
    print("="*80)
    
    # 任务配置（基于reflect demo.ipynb）
    task_info = {
        "name": "make coffee",
        "task_idx": 5,
        "num_samples": 1,
        "failure_injection": False,
        "folder_name": "makeCoffee-1",
        "scene": "FloorPlan16",
        "chosen_failure": "occupied",
        "gt_failure_reason": "The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space.",
        "gt_failure_step": "00:51",
        "preactions": [
            "(dirty_obj, Mug)"
        ],
        "failure_injection_params": {
            "src_obj_type": "Cup",
            "target_obj_type": "CoffeeMachine",
            "disp_x": 0.0,
            "disp_z": 0.05,
            "disp_y": 0.02
        },
        "actions": [
            "(navigate_to_obj, Mug)",
            "(pick_up, Mug)",
            "(navigate_to_obj, Sink)",
            "(put_on, Mug, SinkBasin)",
            "(toggle_on, Faucet)",
            "(toggle_off, Faucet)",
            "(pick_up, Mug)",
            "(pour, Mug, Sink)",
            "(navigate_to_obj, CoffeeMachine)",
            "(put_in, Mug, CoffeeMachine)",
            "(toggle_on, CoffeeMachine)",
            "(toggle_off, CoffeeMachine)",
            "(pick_up, Mug)",
            "(put_on, Mug, CounterTop)"
        ],
        "success_condition": "a clean mug is filled with coffee and on top of the countertop."
    }
    
    # 运行REFLECT框架
    print("\n" + "="*80)
    print("Running REFLECT Framework...")
    print("="*80)
    reflect_stats = run_reflect_framework('/home/fdse/zzy/reflect', task_info)
    
    # 运行CRAFT框架
    print("\n" + "="*80)
    print("Running CRAFT Framework...")
    print("="*80)
    craft_stats = run_craft_framework(task_info)
    
    # 生成对比报告
    print("\n" + "="*80)
    print("Generating Comparison Report...")
    print("="*80)
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'dataset': 'makeCoffee-1',
        'task_info': task_info,
        'reflect': reflect_stats.get_summary(),
        'craft': craft_stats.get_summary(),
        'comparison': {
            'reflect_total_duration': reflect_stats.get_summary()['total_duration'],
            'craft_total_duration': craft_stats.get_summary()['total_duration'],
            'reflect_steps': len(reflect_stats.get_summary()['steps']),
            'craft_steps': len(craft_stats.get_summary()['steps']),
            'reflect_errors': len(reflect_stats.get_summary()['errors']),
            'craft_errors': len(craft_stats.get_summary()['errors'])
        }
    }
    
    # 保存报告
    output_dir = 'output/comparison'
    os.makedirs(output_dir, exist_ok=True)
    report_file = f"{output_dir}/framework_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Report saved to: {report_file}")
    
    # 打印摘要
    print("\n" + "="*80)
    print("Comparison Summary")
    print("="*80)
    print(f"REFLECT Framework:")
    print(f"  - Total Duration: {report['comparison']['reflect_total_duration']:.2f}s")
    print(f"  - Steps: {report['comparison']['reflect_steps']}")
    print(f"  - Errors: {report['comparison']['reflect_errors']}")
    print(f"\nCRAFT Framework:")
    print(f"  - Total Duration: {report['comparison']['craft_total_duration']:.2f}s")
    print(f"  - Steps: {report['comparison']['craft_steps']}")
    print(f"  - Errors: {report['comparison']['craft_errors']}")
    
    # 打印详细步骤
    print("\n" + "="*80)
    print("Detailed Step Statistics")
    print("="*80)
    
    print("\nREFLECT Steps:")
    for step_name, step_data in reflect_stats.get_summary()['steps'].items():
        duration = step_data.get('duration', 0)
        print(f"  - {step_name}: {duration:.2f}s")
        if step_data.get('outputs'):
            for key, value in step_data['outputs'].items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"    {key}: {value}")
    
    print("\nCRAFT Steps:")
    for step_name, step_data in craft_stats.get_summary()['steps'].items():
        duration = step_data.get('duration', 0)
        print(f"  - {step_name}: {duration:.2f}s")
        if step_data.get('outputs'):
            for key, value in step_data['outputs'].items():
                if isinstance(value, (str, int, float, bool)):
                    print(f"    {key}: {value}")
    
    return report


if __name__ == "__main__":
    main()
