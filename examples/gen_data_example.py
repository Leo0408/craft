"""
Example: Using REFLECT-style data generation with failure injection
"""

import sys
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from craft.utils.gen_data import run_data_gen

# Task configuration
task = {
    "name": "make coffee",
    "task_idx": 5,  # Index in TASK_DICT
    "num_samples": 1,  # Number of samples to generate
    "failure_injection": True,  # Enable failure injection
    "folder_name": "makeCoffee-1",
    "scene": "FloorPlan16",  # AI2THOR scene
    "actions": [
        "navigate_to_obj, Mug",
        "pick_up, Mug",
        "navigate_to_obj, Sink",
        "put_on, SinkBasin, Mug",
        "toggle_on, Faucet",
        "toggle_off, Faucet",
        "pick_up, Mug",
        "navigate_to_obj, CoffeeMachine",
        "put_in, CoffeeMachine, Mug",  # This may fail if machine already has a cup
    ],
    # Optional: Specify failure injection parameters
    # "chosen_failure": "drop",  # or "failed_action", "missing_step"
    # "failure_injection_params": {...}
}

# Run data generation
print("Starting data generation with failure injection...")
run_data_gen(data_path=".", task=task)
print("Data generation completed!")

