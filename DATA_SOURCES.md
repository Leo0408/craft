# Data Sources Guide

CRAFT supports two data sources for generating scene graphs:

## 1. Real-World Data (REFLECT Dataset)

### Overview
Uses actual robot execution recordings stored in zarr format.

### Setup

1. **Install dependencies**:
```bash
pip install zarr imagecodecs
```

2. **Prepare data**:
   - Ensure REFLECT dataset is available
   - Data should be in zarr format: `reflect_dataset/real_data/{folder_name}/replay_buffer.zarr`

### Usage

```python
from craft.utils import ReflectDataLoader
from craft.perception import ReflectSceneGraphBuilder, ObjectDetector, SceneAnalyzer

# Initialize
data_loader = ReflectDataLoader(data_root="/path/to/reflect")
detector = ObjectDetector(model_name="mdetr", device="cuda:0")
scene_analyzer = SceneAnalyzer()

builder = ReflectSceneGraphBuilder(
    detector=detector,
    scene_analyzer=scene_analyzer,
    camera_intrinsics={
        "fx": 914.27246,
        "fy": 913.2658,
        "cx": 647.0733,
        "cy": 356.32526
    }
)

# Load data
task_info = data_loader.load_task_info("makeCoffee2")
zarr_group = data_loader.load_zarr_file("makeCoffee2")

# Process frame
rgb = data_loader.load_frame_rgb(zarr_group, step_idx=100)
depth = data_loader.load_frame_depth(zarr_group, step_idx=100)

scene_graph = builder.process_frame(
    rgb=rgb,
    depth=depth,
    step_idx=100,
    object_list=task_info.get('object_list', []),
    task_info=task_info
)
```

### Data Format

Zarr structure:
```
replay_buffer.zarr/
├── data/
│   ├── stage/          # Task stage info
│   ├── gripper_pos/    # Gripper positions
│   └── videos/
│       ├── color/      # RGB: {step_idx}.0.0.0
│       └── depth/      # Depth: {step_idx}.0.0
```

## 2. Simulated Environment (AI2THOR)

### Overview
Generates data from AI2THOR simulation environment, similar to REFLECT's simulation workflow.

### Setup

1. **Install AI2THOR**:
```bash
pip install ai2thor
```

2. **Ensure REFLECT utilities are available**:
   - The simulated builder uses some REFLECT utilities
   - Make sure reflect is in Python path or install reflect dependencies

### Usage

```python
from craft.utils import SimulatedDataGenerator
from craft.perception import SimulatedSceneGraphBuilder, SceneAnalyzer

# Initialize simulation
generator = SimulatedDataGenerator(scene="FloorPlan16")
controller = generator.initialize()

# Execute actions
actions = [
    "MoveAhead",
    "PickupObject",
    # ... more actions
]
events = generator.execute_actions(actions)

# Generate scene graphs
builder = SimulatedSceneGraphBuilder(scene_analyzer=SceneAnalyzer())
object_list = ["Mug", "CoffeeMachine", "Sink"]

scene_graphs = {}
for step_idx, event in enumerate(events):
    scene_graph = builder.process_event(
        step_idx=step_idx,
        event=event,
        object_list=object_list,
        task=task_info
    )
    scene_graphs[step_idx] = scene_graph
    print(f"Frame {step_idx}: {scene_graph.to_text()}")
```

### Workflow

Similar to REFLECT's `generate_scene_graphs()`:
1. Initialize AI2THOR controller
2. Execute task actions
3. For each event:
   - Extract instance masks
   - Convert depth to point clouds
   - Compute 3D bounding boxes
   - Build scene graph

## Comparison

| Feature | Real-World Data | Simulated Data |
|---------|----------------|----------------|
| **Source** | Robot recordings | AI2THOR simulation |
| **Format** | Zarr files | AI2THOR events |
| **Object Detection** | MDETR/DETIC | Instance masks (GT) |
| **Point Clouds** | From depth images | From depth + masks |
| **Failure Scenarios** | Real failures | Controllable injection |
| **Setup Complexity** | Medium | Low (if reflect available) |
| **Data Size** | Large | Generated on-the-fly |

## Integration with CRAFT

Both data sources produce `SceneGraph` objects that can be used with:

- **Failure Analysis**: `FailureAnalyzer.analyze_failure()`
- **Correction Planning**: `CorrectionPlanner.generate_correction_plan()`
- **Task Execution**: `TaskExecutor` for tracking actions

## Example: Complete Workflow

```python
# Option 1: Real-world data
from craft.utils import ReflectDataLoader
from craft.perception import ReflectSceneGraphBuilder

data_loader = ReflectDataLoader("/path/to/reflect")
zarr_group = data_loader.load_zarr_file("makeCoffee2")
task_info = data_loader.load_task_info("makeCoffee2")

builder = ReflectSceneGraphBuilder(detector=detector, ...)
scene_graph = builder.process_frame_from_zarr(
    zarr_group, step_idx=100, 
    object_list=task_info['object_list'],
    task_info=task_info
)

# Option 2: Simulated data
from craft.utils import SimulatedDataGenerator
from craft.perception import SimulatedSceneGraphBuilder

generator = SimulatedDataGenerator(scene="FloorPlan16")
events = generator.execute_actions(actions)

builder = SimulatedSceneGraphBuilder(...)
for step_idx, event in enumerate(events):
    scene_graph = builder.process_event(
        step_idx, event, object_list, task_info
    )

# Both produce SceneGraph objects for failure analysis
from craft.reasoning import FailureAnalyzer
analyzer = FailureAnalyzer(llm_prompter)
failure_analysis = analyzer.analyze_failure(
    task_executor, scene_graphs, task_info
)
```

## Troubleshooting

### Real-World Data Issues

- **Zarr file not found**: Check path and folder structure
- **RGB/Depth loading fails**: Verify zarr structure matches expected format
- **No objects detected**: Check object_list and detector configuration

### Simulated Data Issues

- **AI2THOR import error**: Install with `pip install ai2thor`
- **REFLECT utilities not found**: Ensure reflect is in Python path
- **Point cloud issues**: Check event structure and depth data

## Next Steps

- Process multiple frames for temporal analysis
- Integrate with failure detection
- Generate correction plans from scene graphs



