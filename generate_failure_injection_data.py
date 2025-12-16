"""
Generate AI2THOR data for failure injection test cases
This script actually runs AI2THOR and generates data for the 6 failure scenarios
"""

import os
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.gen_data import run_data_gen
    from utils.constants import TASK_DICT
    AI2THOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import AI2THOR utilities: {e}")
    print("   This script requires AI2THOR to be installed and configured.")
    AI2THOR_AVAILABLE = False


# Define failure injection test cases with AI2THOR task configurations
FAILURE_INJECTION_TASKS = {
    "case_1_occlusion": {
        "name": "PickUp Apple with Occlusion",
        "task_idx": None,  # Will need to find appropriate task or create custom
        "folder_name": "failure_injection_case_1_occlusion",
        "scene": "FloorPlan1",  # Adjust scene as needed
        "num_samples": 1,
        "failure_injection": True,
        "chosen_failure": None,  # Occlusion is simulated by camera angle
        "actions": [
            "navigate_to_obj, Apple",
            "pick_up, Apple",  # This will cause occlusion
        ],
        "failure_type": "occlusion",
        "description": "Apple picked up but occluded by arm"
    },
    
    "case_2_container_conflict": {
        "name": "Put Cup in Closed Drawer",
        "task_idx": None,
        "folder_name": "failure_injection_case_2_container",
        "scene": "FloorPlan1",
        "num_samples": 1,
        "failure_injection": True,
        "chosen_failure": None,  # Container closed is a precondition violation
        "actions": [
            "navigate_to_obj, Cup",
            "pick_up, Cup",
            "navigate_to_obj, Drawer",
            "put_in, Drawer, Cup",  # Will fail if drawer is closed
        ],
        "failure_type": "container_conflict",
        "description": "Cup placed near closed drawer"
    },
    
    "case_3_causal_chain": {
        "name": "Heat Kettle without Filling",
        "task_idx": None,
        "folder_name": "failure_injection_case_3_causal",
        "scene": "FloorPlan1",
        "num_samples": 1,
        "failure_injection": True,
        "chosen_failure": "missing_step",  # Skip fill step
        "specified_missing_steps": [1],  # Skip the fill step
        "actions": [
            "navigate_to_obj, Kettle",
            "pick_up, Kettle",
            "navigate_to_obj, Faucet",
            "put_on, SinkBasin, Kettle",
            "toggle_on, Faucet",  # Fill step - this will be skipped
            "toggle_off, Faucet",
            "pick_up, Kettle",
            "navigate_to_obj, Stove",
            "put_on, Stove, Kettle",
            "toggle_on, Stove",  # Heat without water
        ],
        "failure_type": "causal_chain",
        "description": "Kettle heated without filling"
    },
    
    "case_4_teleport": {
        "name": "Move Mug with Teleport Detection",
        "task_idx": None,
        "folder_name": "failure_injection_case_4_teleport",
        "scene": "FloorPlan1",
        "num_samples": 1,
        "failure_injection": True,
        "chosen_failure": None,  # Teleport will be detected in post-processing
        "actions": [
            "navigate_to_obj, Mug",
            "pick_up, Mug",
            "navigate_to_obj, CounterTop",
            "put_on, CounterTop, Mug",
        ],
        "failure_type": "teleport",
        "description": "Mug moved with possible teleport"
    },
    
    "case_5_near_not_inside": {
        "name": "Place Apple Near Microwave",
        "task_idx": None,
        "folder_name": "failure_injection_case_5_near",
        "scene": "FloorPlan1",
        "num_samples": 1,
        "failure_injection": True,
        "chosen_failure": None,
        "actions": [
            "navigate_to_obj, Apple",
            "pick_up, Apple",
            "navigate_to_obj, Microwave",
            "put_in, Microwave, Apple",  # May place near but not inside
        ],
        "failure_type": "near_not_inside",
        "description": "Apple placed near closed microwave"
    },
    
    "case_6_state_oscillation": {
        "name": "Open Fridge with State Jitter",
        "task_idx": None,
        "folder_name": "failure_injection_case_6_oscillation",
        "scene": "FloorPlan1",
        "num_samples": 1,
        "failure_injection": True,
        "chosen_failure": None,  # State oscillation is detected in post-processing
        "actions": [
            "navigate_to_obj, Fridge",
            "toggle_on, Fridge",  # Open fridge - state may oscillate
        ],
        "failure_type": "state_oscillation",
        "description": "Fridge state oscillating"
    }
}


def generate_failure_injection_data(case_ids=None, output_dir="thor_tasks"):
    """
    Generate AI2THOR data for failure injection test cases
    
    Args:
        case_ids: List of case IDs to generate (None = all)
        output_dir: Directory to save data
    """
    if not AI2THOR_AVAILABLE:
        print("❌ AI2THOR utilities not available. Cannot generate data.")
        print("   Please ensure AI2THOR is installed and utils.gen_data is accessible.")
        return
    
    if case_ids is None:
        case_ids = list(FAILURE_INJECTION_TASKS.keys())
    
    print("=" * 80)
    print("Generating AI2THOR Failure Injection Data")
    print("=" * 80)
    
    results = {}
    
    for case_id in case_ids:
        if case_id not in FAILURE_INJECTION_TASKS:
            print(f"⚠️  Unknown case ID: {case_id}")
            continue
        
        task_config = FAILURE_INJECTION_TASKS[case_id]
        print(f"\n{'='*80}")
        print(f"Generating: {task_config['name']} ({case_id})")
        print(f"Description: {task_config['description']}")
        print(f"{'='*80}")
        
        try:
            # Note: You may need to adjust task_idx based on your TASK_DICT
            # For now, we'll use a placeholder or find the right task
            if task_config.get('task_idx') is None:
                # Try to find a suitable task or use a default
                # You may need to customize this based on your TASK_DICT
                print("⚠️  Note: task_idx not set. You may need to configure this.")
                print("   For now, using a placeholder task configuration.")
            
            # Run data generation
            run_data_gen(data_path=output_dir, task=task_config)
            
            results[case_id] = {
                "status": "success",
                "folder": task_config['folder_name'],
                "path": f"{output_dir}/{task_config['folder_name']}"
            }
            print(f"✅ Successfully generated data for {case_id}")
            print(f"   Data saved to: {output_dir}/{task_config['folder_name']}")
            
        except Exception as e:
            print(f"❌ Error generating data for {case_id}: {e}")
            results[case_id] = {
                "status": "error",
                "error": str(e)
            }
    
    # Save generation summary
    summary_path = f"{output_dir}/failure_injection_generation_summary.json"
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump({
            "cases": results,
            "total": len(results),
            "successful": sum(1 for r in results.values() if r.get('status') == 'success'),
            "failed": sum(1 for r in results.values() if r.get('status') == 'error')
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Generation Summary")
    print(f"{'='*80}")
    print(f"Total cases: {len(results)}")
    print(f"Successful: {sum(1 for r in results.values() if r.get('status') == 'success')}")
    print(f"Failed: {sum(1 for r in results.values() if r.get('status') == 'error')}")
    print(f"\nSummary saved to: {summary_path}")
    
    return results


def list_available_cases():
    """List all available failure injection test cases"""
    print("Available Failure Injection Test Cases:")
    print("=" * 80)
    for case_id, config in FAILURE_INJECTION_TASKS.items():
        print(f"\n{case_id}:")
        print(f"  Name: {config['name']}")
        print(f"  Type: {config['failure_type']}")
        print(f"  Description: {config['description']}")
        print(f"  Actions: {len(config['actions'])} steps")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate AI2THOR failure injection data')
    parser.add_argument('--cases', nargs='+', help='Case IDs to generate (default: all)')
    parser.add_argument('--list', action='store_true', help='List available cases')
    parser.add_argument('--output', default='thor_tasks', help='Output directory')
    
    args = parser.parse_args()
    
    if args.list:
        list_available_cases()
    else:
        generate_failure_injection_data(case_ids=args.cases, output_dir=args.output)

