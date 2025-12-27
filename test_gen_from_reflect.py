import os
import json
import sys
from pathlib import Path

print("Script started...")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from utils.gen_data import run_data_gen
    from utils.constants import TASK_DICT
    AI2THOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Could not import AI2THOR utilities: {e}")
    AI2THOR_AVAILABLE = False

def main():
    if not AI2THOR_AVAILABLE:
        print("AI2THOR is not available.")
        return

    # Use Task 1 from reflect
    task = {
        "name": "make coffee",
        "task_idx": 5,
        "num_samples": 1,
        "failure_injection": False,
        "folder_name": "makeCoffee-test",
        "scene": "FloorPlan16",
        "actions": [
            "(navigate_to_obj, Mug)",
            "(pick_up, Mug)",
            "(navigate_to_obj, CoffeeMachine)",
            "(put_in, Mug, CoffeeMachine)",
        ]
    }

    print(f"--- Generating test data for {task['name']} ---")
    run_data_gen(data_path=".", task=task)
    print("Test generation complete.")

if __name__ == "__main__":
    main()

