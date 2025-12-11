# REFLECT Dataset Integration Guide

This guide explains how to use CRAFT with REFLECT dataset format to generate scene graphs.

## Overview

CRAFT provides integration with REFLECT's data format through:
- `ReflectDataLoader`: Loads data from REFLECT zarr files
- `ReflectSceneGraphBuilder`: Generates scene graphs from RGB-D images

## Scene Graph Generation Process

The scene graph generation follows REFLECT's approach:

1. **Object Detection**: Use MDETR or DETIC to detect objects in RGB images
2. **Point Cloud Generation**: Convert depth images to 3D point clouds
3. **3D Bounding Boxes**: Compute 3D bounding boxes from point clouds
4. **Spatial Relations**: Analyze spatial relationships (inside, on_top_of, near, etc.)
5. **Object States**: Detect object states using CLIP (optional)
6. **Scene Graph Construction**: Build hierarchical scene graph with nodes and edges

## Usage Example

```python
from craft.utils import ReflectDataLoader
from craft.perception import ReflectSceneGraphBuilder, ObjectDetector, SceneAnalyzer

# 1. Initialize data loader
data_loader = ReflectDataLoader(data_root="/path/to/reflect")

# 2. Load task information
folder_name = "makeCoffee2"
task_info = data_loader.load_task_info(folder_name)
object_list = task_info.get('object_list', [])

# 3. Load zarr file
zarr_group = data_loader.load_zarr_file(folder_name)

# 4. Initialize scene graph builder
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

# 5. Process frames and generate scene graphs
step_idx = 100
rgb = data_loader.load_frame_rgb(zarr_group, step_idx)
depth = data_loader.load_frame_depth(zarr_group, step_idx)

if rgb is not None and depth is not None:
    scene_graph = builder.process_frame(
        rgb=rgb,
        depth=depth,
        step_idx=step_idx,
        object_list=object_list,
        distractor_list=task_info.get('distractor_list', []),
        task_info=task_info
    )
    
    print("Scene graph:")
    print(scene_graph.to_text())
```

## Data Format

REFLECT dataset uses zarr format with the following structure:

```
replay_buffer.zarr/
├── data/
│   ├── stage/          # Task stage information
│   ├── gripper_pos/    # Gripper positions
│   └── videos/
│       ├── color/      # RGB images: {step_idx}.0.0.0
│       └── depth/      # Depth images: {step_idx}.0.0
```

## Key Differences from REFLECT

While CRAFT maintains compatibility with REFLECT's data format, there are some architectural differences:

1. **Modular Design**: CRAFT separates concerns into distinct modules
2. **Simplified Interface**: Easier to use and extend
3. **Flexible Detectors**: Can use different object detectors
4. **Unified Scene Graph**: Uses CRAFT's scene graph structure

## Integration Points

### Object Detection
- Supports MDETR (as in REFLECT)
- Can be extended to support DETIC or other detectors
- CLIP-based confirmation (optional)

### Point Cloud Processing
- Uses Open3D for point cloud operations
- Voxel downsampling for efficiency
- Statistical outlier removal for denoising

### Spatial Relations
- Based on point cloud distances
- Supports: inside, on_top_of, near, above, below, etc.
- Configurable thresholds

### Object States
- CLIP-based state detection (optional)
- Supports states like: open/closed, filled/empty, etc.

## Configuration

Update `config/config.json` with your camera intrinsics:

```json
{
    "camera_intrinsics": {
        "fx": 914.27246,
        "fy": 913.2658,
        "cx": 647.0733,
        "cy": 356.32526
    },
    "processing_settings": {
        "voxel_size": 0.01
    }
}
```

## Troubleshooting

### Zarr File Not Found
- Check that the zarr file path is correct
- Ensure the folder structure matches REFLECT format

### No Objects Detected
- Verify object_list matches the objects in the scene
- Check detector threshold settings
- Ensure RGB images are properly loaded

### Point Cloud Issues
- Verify depth images are valid (non-zero values)
- Check camera intrinsics are correct
- Adjust voxel_size if point cloud is too sparse/dense

## Next Steps

1. Process multiple frames to build temporal scene graphs
2. Integrate with failure analysis pipeline
3. Use scene graphs for correction planning
4. Extend to support audio data (as in REFLECT)




