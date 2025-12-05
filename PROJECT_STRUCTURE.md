# CRAFT Project Structure

## Directory Layout

```
craft/
├── __init__.py                 # Package initialization
├── main.py                     # Main entry point
├── test_basic.py              # Basic functionality tests
├── requirements.txt           # Python dependencies
├── README.md                  # Main documentation
├── PROJECT_STRUCTURE.md       # This file
├── .gitignore                 # Git ignore rules
│
├── core/                      # Core modules
│   ├── __init__.py
│   ├── scene_graph.py         # Scene graph representation
│   └── task_executor.py       # Task execution tracking
│
├── perception/                # Perception modules
│   ├── __init__.py
│   ├── object_detector.py     # Object detection (RGB-D)
│   └── scene_analyzer.py      # Scene analysis and spatial relations
│
├── reasoning/                 # Reasoning modules
│   ├── __init__.py
│   ├── llm_prompter.py        # LLM interface
│   └── failure_analyzer.py    # Failure analysis
│
├── correction/                # Correction modules
│   ├── __init__.py
│   └── correction_planner.py  # Correction plan generation
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── config_loader.py       # Configuration loading
│   └── data_loader.py         # Data loading utilities
│
├── config/                    # Configuration files
│   └── config.json            # Main configuration
│
├── examples/                  # Example scripts
│   └── simple_example.py      # Basic usage examples
│
├── data/                      # Data directory (user-created)
└── output/                    # Output directory (auto-created)
```

## Module Dependencies

```
main.py
  ├── core (TaskExecutor, SceneGraph)
  ├── perception (ObjectDetector, SceneAnalyzer)
  ├── reasoning (LLMPrompter, FailureAnalyzer)
  ├── correction (CorrectionPlanner)
  └── utils (load_config, DataLoader)

core/
  └── scene_graph.py (standalone)

perception/
  ├── object_detector.py (standalone)
  └── scene_analyzer.py → core.scene_graph

reasoning/
  ├── llm_prompter.py (standalone)
  └── failure_analyzer.py → core.scene_graph, core.task_executor, llm_prompter

correction/
  └── correction_planner.py → core.task_executor, core.scene_graph, reasoning.llm_prompter

utils/
  ├── config_loader.py (standalone)
  └── data_loader.py (standalone)
```

## Key Components

### Core Module
- **SceneGraph**: Hierarchical representation of robot environment
- **TaskExecutor**: Tracks task execution and action history

### Perception Module
- **ObjectDetector**: Detects objects in RGB-D images
- **SceneAnalyzer**: Analyzes spatial relationships and object states

### Reasoning Module
- **LLMPrompter**: Interface for querying Large Language Models
- **FailureAnalyzer**: Analyzes failures using progressive reasoning

### Correction Module
- **CorrectionPlanner**: Generates executable plans to recover from failures

## Usage Flow

1. **Data Loading**: Load task info and frame data
2. **Perception**: Detect objects and build scene graphs
3. **Task Execution**: Track actions and detect failures
4. **Failure Analysis**: Use LLM to analyze failures
5. **Correction Planning**: Generate recovery plans

## Testing

Run basic tests:
```bash
cd /home/fdse/zzy
python3 craft/test_basic.py
```

Run examples:
```bash
cd /home/fdse/zzy
python3 craft/examples/simple_example.py
```

## Integration Points

The framework is designed to integrate with:
- REFLECT framework data formats
- MDETR object detector
- OpenAI GPT models (or compatible APIs)
- Custom robot execution systems




