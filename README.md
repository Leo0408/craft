# CRAFT: Core Robot Analysis Framework for Tasks

CRAFT is a framework for robot failure analysis and correction, inspired by the REFLECT framework. It provides a modular architecture for analyzing robot task execution, detecting failures, and generating correction plans using Large Language Models (LLMs).

## Overview

CRAFT enables robots to:
- **Perceive** the environment through multi-modal sensors (RGB-D, audio)
- **Understand** the scene using hierarchical scene graphs
- **Reason** about failures using LLM-based analysis
- **Correct** failures by generating executable recovery plans

## Architecture

The framework is organized into four main modules:

### 1. Core Module (`core/`)
- **SceneGraph**: Hierarchical representation of the robot's environment
- **TaskExecutor**: Manages task execution and tracks action history

### 2. Perception Module (`perception/`)
- **ObjectDetector**: Detects objects in RGB-D images
- **SceneAnalyzer**: Analyzes spatial relationships and object states

### 3. Reasoning Module (`reasoning/`)
- **LLMPrompter**: Interface for querying Large Language Models
- **FailureAnalyzer**: Analyzes failures using progressive reasoning

### 4. Correction Module (`correction/`)
- **CorrectionPlanner**: Generates executable plans to recover from failures

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for object detection)
- OpenAI API key (for LLM queries)

### Setup

1. Clone or navigate to the craft directory:
```bash
cd /home/fdse/zzy/craft
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure the framework:
   - Copy `config/config.json` and update with your settings
   - Set your OpenAI API key in the config file or as environment variable:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

## Quick Start

### Basic Usage

```python
from craft import SceneGraph, Node, Edge, TaskExecutor
from craft.reasoning import LLMPrompter, FailureAnalyzer
from craft.correction import CorrectionPlanner

# Create a scene graph
scene_graph = SceneGraph()
scene_graph.add_node(Node("coffee machine", "CoffeeMachine", state="closed"))
scene_graph.add_node(Node("cup", "Mug", state="empty"))

# Initialize LLM components
llm_prompter = LLMPrompter(gpt_version="gpt-3.5-turbo")
failure_analyzer = FailureAnalyzer(llm_prompter)
correction_planner = CorrectionPlanner(llm_prompter)

# Analyze failures and generate corrections
# ... (see examples for full workflow)
```

### Running Examples

```bash
# Run the simple example
python examples/simple_example.py

# Run the main pipeline
python main.py --task data/tasks/make_coffee.json --data-root data

# Run the interactive Jupyter notebook demo
jupyter notebook demo.ipynb
# or
jupyter lab demo.ipynb
```

### Interactive Demo Notebook

The `demo.ipynb` notebook provides a complete walkthrough of the CRAFT framework:

1. **Scene Graph Creation** - Learn how to build hierarchical scene representations
2. **Perception Module** - See object detection and scene analysis in action
3. **Task Execution** - Track robot actions and identify failures
4. **Failure Analysis** - Use LLM to analyze and explain failures
5. **Correction Planning** - Generate executable recovery plans

To run the demo:
```bash
cd /home/fdse/zzy/craft
jupyter notebook demo.ipynb
```

## Project Structure

```
craft/
├── core/                   # Core modules
│   ├── __init__.py
│   ├── scene_graph.py     # Scene graph representation
│   └── task_executor.py   # Task execution tracking
├── perception/            # Perception modules
│   ├── __init__.py
│   ├── object_detector.py # Object detection
│   └── scene_analyzer.py  # Scene analysis
├── reasoning/             # Reasoning modules
│   ├── __init__.py
│   ├── llm_prompter.py    # LLM interface
│   └── failure_analyzer.py # Failure analysis
├── correction/            # Correction modules
│   ├── __init__.py
│   └── correction_planner.py # Correction planning
├── utils/                 # Utility functions
│   ├── __init__.py
│   ├── config_loader.py   # Configuration loading
│   └── data_loader.py     # Data loading
├── config/                # Configuration files
│   └── config.json        # Main configuration
├── examples/              # Example scripts
│   └── simple_example.py  # Basic usage examples
├── data/                  # Data directory (create as needed)
├── output/                # Output directory (created automatically)
├── main.py                # Main entry point
├── requirements.txt       # Python dependencies
├── __init__.py            # Package initialization
└── README.md              # This file
```

## Configuration

Edit `config/config.json` to customize:

- **API Keys**: Set your OpenAI API key
- **Camera Intrinsics**: Configure camera parameters for 3D reconstruction
- **Detection Thresholds**: Adjust object detection sensitivity
- **Model Settings**: Choose object detector and LLM model
- **Processing Settings**: Configure frame sampling and processing parameters

Example configuration:
```json
{
    "openai_api_key": "your-key-here",
    "model_settings": {
        "object_detector": "mdetr",
        "llm_model": "gpt-3.5-turbo",
        "device": "cuda:0"
    },
    "detection_thresholds": {
        "object_detection_threshold": 0.7
    }
}
```

## Usage Examples

### Example 1: Creating a Scene Graph

```python
from craft.core import SceneGraph, Node, Edge

# Create scene graph
sg = SceneGraph()
sg.add_node(Node("coffee machine", "CoffeeMachine", state="closed"))
sg.add_node(Node("cup", "Mug", state="empty"))
sg.add_node(Node("table", "Table"))

# Add relationships
edge = Edge(
    sg.get_node("cup"),
    sg.get_node("table"),
    "on_top_of"
)
sg.add_edge(edge)

# Convert to text description
print(sg.to_text())
# Output: "coffee machine (closed), cup (empty), table. cup (empty) is on_top_of table."
```

### Example 2: Task Execution Tracking

```python
from craft.core import TaskExecutor

actions = [
    {"type": "pick_up", "target": "Mug"},
    {"type": "put_in", "source": "Mug", "target": "CoffeeMachine"}
]

executor = TaskExecutor("make coffee", actions)

# Mark actions as success/failure
executor.mark_action_success(0)
executor.mark_action_failed(1, "Coffee machine already contains a cup")

# Get failed actions
failed = executor.get_failed_actions()
for action in failed:
    print(f"{action.action_type}: {action.failure_reason}")
```

### Example 3: Failure Analysis

```python
from craft.reasoning import LLMPrompter, FailureAnalyzer

llm_prompter = LLMPrompter(gpt_version="gpt-3.5-turbo")
analyzer = FailureAnalyzer(llm_prompter)

# Verify subgoal
is_success, explanation = llm_prompter.verify_subgoal(
    task="make coffee",
    subgoal="put cup in coffee machine",
    observation="a coffee machine (closed), a purple cup, a table. a blue cup is inside the coffee machine."
)

print(f"Success: {is_success}")
print(f"Explanation: {explanation}")
```

### Example 4: Correction Planning

```python
from craft.correction import CorrectionPlanner
from craft.reasoning import LLMPrompter

llm_prompter = LLMPrompter()
planner = CorrectionPlanner(llm_prompter)

correction_plan = planner.generate_correction_plan(
    task_info={"name": "make coffee"},
    original_plan=original_actions,
    failure_explanation="Coffee machine already contains a cup",
    final_state=scene_graph,
    expected_goal="a clean mug filled with coffee"
)

for action in correction_plan:
    print(f"{action['type']} {action.get('target', '')}")
```

## Integration with REFLECT

CRAFT is designed to be compatible with the REFLECT framework. You can:

1. **Use REFLECT's data format**: CRAFT can load data from REFLECT's zarr format
2. **Generate scene graphs**: Use `ReflectSceneGraphBuilder` to generate scene graphs from REFLECT data
3. **Leverage existing models**: Integrate MDETR and other models from REFLECT
4. **Extend functionality**: Add new modules while maintaining compatibility

### Using REFLECT Dataset

```python
from craft.utils import ReflectDataLoader
from craft.perception import ReflectSceneGraphBuilder

# Load REFLECT data
data_loader = ReflectDataLoader(data_root="/path/to/reflect")
zarr_group = data_loader.load_zarr_file("makeCoffee2")

# Generate scene graph
builder = ReflectSceneGraphBuilder(detector=detector, scene_analyzer=analyzer)
scene_graph = builder.process_frame_from_zarr(
    zarr_group, step_idx=100, object_list=["coffee machine", "cup"]
)
```

See [REFLECT_INTEGRATION.md](REFLECT_INTEGRATION.md) for detailed guide.

## Extending CRAFT

### Adding a New Object Detector

1. Create a new detector class inheriting from `ObjectDetector`
2. Implement the `detect_objects()` method
3. Update `config.json` to include your detector

### Adding a New LLM Provider

1. Extend `LLMPrompter` or create a new class
2. Implement the `query()` method for your provider
3. Update configuration to select your provider

### Adding Custom Analysis

1. Create a new analyzer in the `reasoning/` module
2. Implement analysis logic
3. Integrate with `FailureAnalyzer`

## Troubleshooting

### Common Issues

1. **LLM API Errors**: Check your API key and network connection
2. **CUDA Out of Memory**: Reduce batch size or use CPU mode
3. **Import Errors**: Ensure all dependencies are installed and paths are correct

### Debug Mode

Set environment variable for verbose logging:
```bash
export CRAFT_DEBUG=1
python main.py --task your_task.json
```

## Citation

If you use CRAFT in your research, please cite:

```bibtex
@software{craft2024,
  title={CRAFT: Core Robot Analysis Framework for Tasks},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/craft}
}
```

## License

[Specify your license here]

## Acknowledgments

CRAFT is inspired by the REFLECT framework:
- Liu, Z., Bahety, A., & Song, S. (2023). REFLECT: Summarizing Robot Experiences for Failure Explanation and Correction. arXiv preprint arXiv:2306.15724.

## Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

