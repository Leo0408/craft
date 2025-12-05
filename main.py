"""
Main entry point for CRAFT framework
"""

import argparse
import json
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from core.task_executor import TaskExecutor, Action, ActionStatus
from core.scene_graph import SceneGraph, Node, Edge
from perception.object_detector import ObjectDetector
from perception.scene_analyzer import SceneAnalyzer
from reasoning.llm_prompter import LLMPrompter
from reasoning.failure_analyzer import FailureAnalyzer
from correction.correction_planner import CorrectionPlanner
from utils.config_loader import load_config
from utils.data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='CRAFT: Robot Failure Analysis Framework')
    parser.add_argument('--config', type=str, default='config/config.json',
                       help='Path to configuration file')
    parser.add_argument('--task', type=str, required=True,
                       help='Path to task JSON file')
    parser.add_argument('--data-root', type=str, default='data',
                       help='Root directory for data files')
    parser.add_argument('--output', type=str, default='output',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load task information
    data_loader = DataLoader(args.data_root)
    task_info = data_loader.load_task_info(args.task)
    
    print(f"Task: {task_info.get('name', 'Unknown')}")
    print(f"Actions: {len(task_info.get('actions', []))}")
    
    # Initialize components
    print("\nInitializing components...")
    
    # Object detector
    detector = ObjectDetector(
        model_name=config['model_settings']['object_detector'],
        device=config['model_settings']['device'],
        threshold=config['detection_thresholds']['object_detection_threshold']
    )
    
    # Scene analyzer
    scene_analyzer = SceneAnalyzer()
    
    # LLM prompter
    llm_prompter = LLMPrompter(
        gpt_version=config['model_settings']['llm_model'],
        api_key=config.get('openai_api_key')
    )
    
    # Failure analyzer
    failure_analyzer = FailureAnalyzer(llm_prompter)
    
    # Correction planner
    correction_planner = CorrectionPlanner(llm_prompter)
    
    # Task executor
    actions = task_info.get('actions', [])
    task_executor = TaskExecutor(task_info['name'], actions)
    
    print("Components initialized successfully!")
    
    # Process key frames (simplified example)
    print("\nProcessing scene...")
    object_list = task_info.get('object_list', [])
    scene_graphs = {}
    
    # Example: process a few key frames
    key_frames = [0, 100, 200]  # Simplified
    
    for frame_idx in key_frames:
        try:
            frame_data = data_loader.load_frame_data(frame_idx, task_info.get('folder_name', ''))
            
            if frame_data['rgb'] is None:
                continue
            
            # Detect objects
            detections = detector.detect_with_depth(
                frame_data['rgb'],
                frame_data['depth'] if frame_data['depth'] is not None else None,
                object_list,
                config['camera_intrinsics']
            )
            
            # Compute spatial relations
            relations = scene_analyzer.compute_spatial_relations(detections)
            
            # Build scene graph
            scene_graph = scene_analyzer.build_scene_graph(detections, relations, task_info)
            scene_graphs[frame_idx] = scene_graph
            
            print(f"Frame {frame_idx}: {scene_graph.to_text()[:100]}...")
            
        except Exception as e:
            print(f"Error processing frame {frame_idx}: {e}")
            continue
    
    # Analyze failures
    print("\nAnalyzing failures...")
    failure_analysis = failure_analyzer.analyze_failure(
        task_executor,
        scene_graphs,
        task_info
    )
    
    print(f"\nFailure Analysis:")
    print(f"Type: {failure_analysis.get('failure_type', 'unknown')}")
    if 'explanations' in failure_analysis:
        for explanation in failure_analysis['explanations']:
            print(f"  - {explanation}")
    
    # Generate correction plan if failure detected
    if failure_analysis.get('failure_type') != 'none':
        print("\nGenerating correction plan...")
        final_state = list(scene_graphs.values())[-1] if scene_graphs else None
        correction_plan = correction_planner.generate_correction_plan(
            task_info,
            task_executor.actions,
            failure_analysis.get('explanations', [''])[0] if failure_analysis.get('explanations') else '',
            final_state,
            task_info.get('success_condition', '')
        )
        
        print("\nCorrection Plan:")
        for i, action in enumerate(correction_plan, 1):
            print(f"  {i}. {action.get('type', '')} {action.get('target', '')}")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = os.path.join(args.output, 'analysis_result.json')
    with open(output_file, 'w') as f:
        json.dump({
            'failure_analysis': failure_analysis,
            'correction_plan': correction_plan if 'correction_plan' in locals() else []
        }, f, indent=2)
    
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()




