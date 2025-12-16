"""
Notebook Integration Example for test_failure_injection.py

Use this code in demo1.ipynb to run the failure injection test
"""

# Option 1: Direct import (if craft is in sys.path)
# Make sure craft parent directory is in sys.path first
import sys
from pathlib import Path

# Add craft parent directory to path if not already there
craft_dir = Path.cwd()  # Assuming notebook is in craft directory
if str(craft_dir) not in sys.path:
    sys.path.insert(0, str(craft_dir))
    print(f"✅ Added {craft_dir} to sys.path")

# Now import and run
from test_failure_injection import run_comparison_test

# Run the test
results = run_comparison_test()

# Display results
print("\n" + "="*80)
print("Test Complete!")
print("="*80)
print(f"CRAFT Accuracy: {results['craft']['correct']}/{results['craft']['total']}")
print(f"REFLECT Accuracy: {results['reflect']['correct']}/{results['reflect']['total']}")

