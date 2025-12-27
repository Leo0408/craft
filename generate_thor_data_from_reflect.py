import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.gen_data import run_data_gen
    from utils.constants import TASK_DICT
    from tqdm import tqdm
    AI2THOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import AI2THOR utilities or tqdm: {e}")
    AI2THOR_AVAILABLE = False

def main():
    if not AI2THOR_AVAILABLE:
        print("AI2THOR or tqdm is not available. Please install them first.")
        print("pip3 install ai2thor tqdm")
        return

    # Path to reflect tasks.json
    reflect_tasks_path = os.path.join(project_root.parent, 'reflect', 'main', 'tasks.json')
    
    if not os.path.exists(reflect_tasks_path):
        print(f"Error: Could not find reflect tasks.json at {reflect_tasks_path}")
        return

    with open(reflect_tasks_path, 'r') as f:
        tasks = json.load(f)

    task_items = list(tasks.items())
    print(f"Found {len(task_items)} task templates in reflect/main/tasks.json")

    # Create thor_tasks directory if it doesn't exist
    os.makedirs('thor_tasks', exist_ok=True)

    pbar = tqdm(task_items, desc="Overall Progress")
    for task_id, task_config in pbar:
        pbar.set_description(f"Processing {task_id}: {task_config['name']}")
        try:
            # The run_data_gen expects a data_path, usually "." or project root
            run_data_gen(data_path=".", task=task_config)
        except Exception as e:
            print(f"\n[ERROR] Failed to generate data for {task_id}: {e}")
            import traceback
            traceback.print_exc()

    print("\n✅ All tasks processed. Data saved in thor_tasks/")

if __name__ == "__main__":
    main()

