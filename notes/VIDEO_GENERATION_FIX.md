# 视频生成问题修复

## 问题描述

在 Step 2 结束时出现警告：
```
⚠️  Video generation skipped: VideoGenerator.generate_video() missing 2 required positional arguments: 'scene_graphs' and 'step_indices'
```

## 问题原因

1. **`VideoGenerator.generate_video()` 需要参数**：
   - `frames`: 帧列表 ✅ (已有)
   - `scene_graphs`: 场景图列表 ❌ (Step 2 时还没有)
   - `step_indices`: 步骤索引列表 ❌ (Step 2 时还没有)

2. **时序问题**：
   - Step 2: 提取 frames（✅ 完成）
   - Step 3: 生成 scene_graphs（❌ 还没做）
   - 所以 Step 2 无法使用完整的 `generate_video()` 方法

## 解决方案

### 方案 1: 在 Step 2 生成简单视频（已实现）

在 Step 2 使用 `cv2.VideoWriter` 直接生成简单视频（只包含 frames，不需要 scene_graphs）：

```python
# 使用 cv2 直接生成视频（不需要 scene_graphs）
if len(frames) > 0 and cv2 is not None:
    output_path = "output/videos/craft_ai2thor_workflow_simple.mp4"
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 2.0, (w, h))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    print(f"✅ Simple video generated: {output_path}")
```

**优点**：
- ✅ 可以在 Step 2 立即看到视频
- ✅ 不需要等待 Step 3 的 scene_graphs
- ✅ 视频包含动作标注（已在 frames 中添加）

**缺点**：
- ⚠️ 不包含场景图可视化

### 方案 2: 在 Step 3 之后生成完整视频（推荐）

在 Step 3 生成 scene_graphs 后，使用完整的 `generate_video()` 方法：

```python
# 在 Step 3 之后
if len(scene_graphs_craft) > 0 and len(frames) > 0:
    step_indices = list(range(len(frames)))
    action_infos = [
        {
            "type": r.get('action_name', 'N/A'),
            "target": r.get('params', {}).get('objectId', 'N/A'),
            "status": r.get('status', 'UNKNOWN')
        }
        for r in action_results
    ]
    
    video_path = video_generator.generate_video(
        frames=frames,
        scene_graphs=scene_graphs_craft,
        step_indices=step_indices,
        action_infos=action_infos,
        output_filename="craft_ai2thor_workflow_full.mp4",
        fps=2
    )
    print(f"✅ Full video with scene graphs generated: {video_path}")
```

**优点**：
- ✅ 包含场景图可视化
- ✅ 包含完整的动作信息
- ✅ 更丰富的视频内容

## 当前实现

已修复 Step 2，现在会：

1. ✅ **生成简单视频**：使用 cv2 直接生成，包含动作标注
2. ✅ **输出路径**：`output/videos/craft_ai2thor_workflow_simple.mp4`
3. ✅ **提示信息**：告知用户完整视频将在 Step 3 后生成

## 视频输出位置

修复后，Step 2 结束时会生成：
- **简单视频**: `output/videos/craft_ai2thor_workflow_simple.mp4`
  - 包含所有 frames
  - 包含动作标注（Step X: action, Status: SUCCESS/FAILED）
  - 不包含场景图可视化

## 完整视频（Step 3 之后）

如果需要包含场景图的完整视频，可以在 Step 3 之后添加：

```python
# 在 Step 3 之后添加完整视频生成
if len(scene_graphs_craft) > 0 and len(frames) > 0:
    step_indices = [i+1 for i in range(len(frames))]
    action_infos = [
        {
            "type": r.get('action_name', 'N/A'),
            "target": r.get('params', {}).get('objectId', 'N/A')[:30] if r.get('params', {}).get('objectId') else 'N/A',
            "status": r.get('status', 'UNKNOWN')
        }
        for r in action_results
    ]
    
    try:
        video_path_full = video_generator.generate_video(
            frames=frames,
            scene_graphs=scene_graphs_craft,
            step_indices=step_indices,
            action_infos=action_infos,
            output_filename="craft_ai2thor_workflow_full.mp4",
            fps=2
        )
        print(f"✅ Full video with scene graphs generated: {video_path_full}")
    except Exception as e:
        print(f"⚠️  Full video generation failed: {e}")
```

## 总结

| 视频类型 | 生成时机 | 包含内容 | 文件位置 |
|---------|---------|---------|---------|
| **简单视频** | Step 2 结束 | Frames + 动作标注 | `output/videos/craft_ai2thor_workflow_simple.mp4` |
| **完整视频** | Step 3 之后（可选） | Frames + 场景图 + 动作信息 | `output/videos/craft_ai2thor_workflow_full.mp4` |

## 影响评估

**警告的影响**：
- ⚠️ **之前**: 视频生成完全失败，无法看到视频
- ✅ **现在**: 可以生成简单视频，可以看到执行过程

**建议**：
- Step 2 结束后可以立即查看简单视频
- 如果需要场景图可视化，等待 Step 3 完成后生成完整视频

