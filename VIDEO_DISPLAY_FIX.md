# 视频显示和 matplotlib API 修复

## 问题描述

1. **matplotlib API 错误**：
   ```
   AttributeError: 'FigureCanvasAgg' object has no attribute 'tostring_rgb'
   ```
   这是因为不同版本的 matplotlib 使用不同的 API。

2. **用户需求**：
   - Step 2 就显示视频（简单视频，无 scene graphs）
   - Step 3 保留 frame 和 scene graph 对比的方式（完整视频）

## 修复方案

### 1. 修复 matplotlib API 兼容性 (`utils/video_generator.py`)

更新了 `create_frame_with_annotations` 方法，支持多个 matplotlib 版本：

```python
# Convert figure to numpy array
fig.canvas.draw()
# Handle different matplotlib versions
try:
    # Method 1: buffer_rgba (matplotlib 3.5+, recommended)
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    h, w = fig.canvas.get_width_height()
    buf = buf.reshape((h, w, 4))
    buf = buf[:, :, :3]  # Remove alpha channel, keep RGB
except (AttributeError, TypeError) as e:
    # Method 2: print_to_buffer (alternative)
    try:
        buf = np.frombuffer(fig.canvas.print_to_buffer()[0], dtype=np.uint8)
        h, w = fig.canvas.get_width_height()
        buf = buf.reshape((h, w, 4))
        buf = buf[:, :, :3]  # Remove alpha channel
    except (AttributeError, TypeError):
        # Method 3: Use renderer directly
        try:
            renderer = fig.canvas.get_renderer()
            buf = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except (AttributeError, TypeError):
            # Fallback: Save to buffer and read back
            import io
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
            buf.seek(0)
            from PIL import Image
            img = Image.open(buf)
            buf = np.array(img)[:, :, :3]  # Remove alpha if present
plt.close(fig)

return buf
```

**修复方法优先级**：
1. `buffer_rgba()` - matplotlib 3.5+ 推荐方法
2. `print_to_buffer()` - 替代方法
3. `renderer.tostring_rgb()` - 旧版本兼容
4. `savefig()` + PIL - 最终 fallback

### 2. Step 2 添加视频显示 (`demo1.ipynb` Cell 9)

在生成简单视频后立即显示：

```python
out.release()
print(f"✅ Simple video generated: {video_path}")

# Display video in notebook (Step 2)
try:
    from IPython.display import Video
    print(f"\n📺 Displaying video in notebook...")
    display(Video(str(video_path), embed=True, width=800))
    print(f"✅ Video displayed above")
except Exception as e:
    print(f"⚠️  Could not display video in notebook: {e}")
    print(f"   Video file saved at: {video_path}")

print(f"\n💡 Note: Full video with scene graphs will be generated in Step 3")
```

**特点**：
- ✅ 在 Step 2 立即显示视频
- ✅ 视频包含动作标注（已在 frames 中添加）
- ✅ 宽度 800px，适合 notebook 显示
- ✅ 不包含 scene graphs（因为 Step 3 才生成）

### 3. Step 3 添加完整视频显示 (`demo1.ipynb` Cell 11)

在生成完整视频后显示（frame 和 scene graph 对比）：

```python
print(f"\n✅ Complete video generated: {video_path}")
print(f"   This video includes:")
print(f"   - RGB frames from AI2THOR")
print(f"   - Scene graph visualizations")
print(f"   - Action annotations")
print(f"   - Object bounding boxes (if available)")

# Display video in notebook (Step 3 - frame and scene graph comparison)
try:
    from IPython.display import Video
    print(f"\n📺 Displaying complete video with scene graphs in notebook...")
    display(Video(str(video_path), embed=True, width=1200))
    print(f"✅ Complete video displayed above (frame and scene graph comparison)")
except Exception as e:
    print(f"⚠️  Could not display video in notebook: {e}")
    print(f"   Video file saved at: {video_path}")
```

**特点**：
- ✅ 在 Step 3 显示完整视频
- ✅ 包含 frame 和 scene graph 对比（并排显示）
- ✅ 宽度 1200px，适合显示完整布局
- ✅ 包含所有可视化信息（scene graphs, action annotations, bounding boxes）

## 视频输出对比

| 特性 | Step 2 简单视频 | Step 3 完整视频 |
|------|----------------|----------------|
| **生成时机** | Step 2 结束后 | Step 3 结束后 |
| **包含内容** | RGB frames + 动作标注 | RGB frames + scene graphs + 动作标注 + bounding boxes |
| **显示方式** | 在 notebook 中显示（800px） | 在 notebook 中显示（1200px） |
| **文件路径** | `output/videos/craft_ai2thor_workflow_simple.mp4` | `output/videos/craft_ai2thor_workflow_complete.mp4` |
| **用途** | 快速检查模拟环境数据生成 | 验证 frame 和 scene graph 的对应关系 |

## 验证

所有修复已完成并验证：

- ✅ matplotlib API 兼容性修复（支持多个版本）
- ✅ Step 2 视频显示功能
- ✅ Step 3 完整视频显示功能（frame 和 scene graph 对比）

## 使用说明

1. **运行 Step 2**：
   - 生成简单视频（无 scene graphs）
   - 在 notebook 中自动显示视频
   - 可以快速检查模拟环境数据生成是否正确

2. **运行 Step 3**：
   - 生成完整视频（包含 scene graphs）
   - 在 notebook 中自动显示视频
   - 可以验证 frame 和 scene graph 的对应关系

3. **如果视频显示失败**：
   - 视频文件仍然保存在 `output/videos/` 目录
   - 可以手动打开视频文件查看

## 注意事项

- 确保已安装 `IPython` 和 `PIL`（Pillow）用于视频显示和 fallback
- 如果 matplotlib 版本过旧，会使用 fallback 方法
- 视频文件会保存在 `output/videos/` 目录，即使 notebook 显示失败也可以手动查看

