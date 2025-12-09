# 视频输出指南：CRAFT vs REFLECT

## 当前状态

### demo1 的视频输出

demo1 现在会生成**两个视频**：

1. **简单视频**（Step 2 生成）
   - 文件名：`output/videos/craft_ai2thor_workflow_simple.mp4`
   - 内容：RGB 帧 + 动作标注（文本）
   - 特点：快速生成，用于初步检查

2. **完整视频**（Step 3 之后生成）✅ **新增**
   - 文件名：`output/videos/craft_ai2thor_workflow_complete.mp4`
   - 内容：
     - RGB 帧（左侧）
     - 场景图可视化（右上）
     - 动作信息（右下）
     - 对象边界框（如果可用）
   - 特点：类似 REFLECT demo 的输出格式

---

## 视频内容对比

### REFLECT Demo 视频

REFLECT 的视频通常包含：
- RGB 帧
- 子目标验证结果
- 场景描述
- 动作序列

### CRAFT Demo 视频（完整版）

CRAFT 的完整视频包含：
- **RGB 帧**（左侧大图）
  - 来自 AI2THOR 的原始帧
  - 包含动作标注（文本叠加）
  
- **场景图可视化**（右上）
  - 节点（对象）
  - 边（关系）
  - 网络图布局
  
- **动作信息**（右下）
  - Step 编号
  - Action 类型
  - Target 对象
  - Status（SUCCESS/FAILED）
  - 节点和边的数量

- **对象边界框**（如果可用）
  - 在 RGB 帧上标注对象位置
  - 颜色编码

---

## 视频生成流程

### Step 2: 简单视频生成

```python
# 提取帧并添加文本标注
for event, action_result in zip(events_craft, action_results):
    frame = extract_frame_from_event(event)
    annotated_frame = add_text_annotation(frame, action_result)
    frames.append(annotated_frame)

# 使用 cv2 直接生成视频
video_path = "output/videos/craft_ai2thor_workflow_simple.mp4"
out = cv2.VideoWriter(video_path, fourcc, 2.0, (w, h))
for frame in frames:
    out.write(frame)
out.release()
```

**输出**：
- ✅ 快速生成
- ✅ 包含动作标注
- ⚠️ 不包含场景图

### Step 3 之后: 完整视频生成

```python
# 使用 VideoGenerator 生成完整视频
video_generator = VideoGenerator(output_dir="output/videos")
video_path = video_generator.generate_video(
    frames=frames,                    # 从 Step 2 提取的帧
    scene_graphs=scene_graphs_craft,   # 从 Step 3 生成的场景图
    step_indices=step_indices,         # 步骤索引
    action_infos=action_infos,         # 动作信息
    output_filename="craft_ai2thor_workflow_complete.mp4",
    fps=2.0
)
```

**输出**：
- ✅ 包含场景图可视化
- ✅ 包含动作信息
- ✅ 类似 REFLECT 的输出格式

---

## 视频文件位置

所有视频保存在：`output/videos/`

```
output/videos/
├── craft_ai2thor_workflow_simple.mp4    # Step 2 生成的简单视频
└── craft_ai2thor_workflow_complete.mp4  # Step 3 之后生成的完整视频
```

---

## 如何检查模拟环境数据生成

### 1. 查看简单视频（快速检查）

```bash
# 在 Jupyter 中
from IPython.display import Video
Video("output/videos/craft_ai2thor_workflow_simple.mp4")
```

**检查内容**：
- ✅ 动作是否正确执行
- ✅ 帧是否正常提取
- ✅ 动作标注是否正确

### 2. 查看完整视频（详细检查）

```bash
# 在 Jupyter 中
from IPython.display import Video
Video("output/videos/craft_ai2thor_workflow_complete.mp4")
```

**检查内容**：
- ✅ 场景图是否正确生成
- ✅ 对象关系是否正确
- ✅ 动作与场景图的对应关系
- ✅ 状态变化是否准确

---

## 与 REFLECT 的对比

| 方面 | REFLECT Demo | CRAFT Demo (完整版) |
|------|-------------|---------------------|
| **视频内容** | RGB + 子目标验证 | RGB + 场景图 + 动作信息 |
| **可视化** | 子目标状态 | 场景图网络图 |
| **标注** | 子目标描述 | 动作 + 状态 |
| **检查方式** | LLM 验证子目标 | 可执行约束验证 |
| **优势** | 细粒度检测 | 结构化场景表示 |

---

## 故障排除

### 问题 1: 视频未生成

**检查**：
1. `frames` 变量是否在 Step 2 中正确创建
2. `scene_graphs_craft` 是否在 Step 3 中正确生成
3. `cv2` 是否可用

**解决**：
```python
# 检查变量
print(f"Frames: {len(frames)}")
print(f"Scene graphs: {len(scene_graphs_craft)}")
print(f"cv2 available: {cv2 is not None}")
```

### 问题 2: 视频文件不存在

**检查**：
```bash
ls -lh output/videos/
```

**可能原因**：
- 路径问题
- 权限问题
- 生成失败但未报错

### 问题 3: 视频无法播放

**检查**：
- 视频编码格式（mp4v）
- 帧尺寸是否一致
- 文件是否完整

**解决**：
```python
# 检查视频文件
import os
video_path = "output/videos/craft_ai2thor_workflow_complete.mp4"
if os.path.exists(video_path):
    size = os.path.getsize(video_path)
    print(f"Video size: {size / 1024 / 1024:.2f} MB")
else:
    print("Video file not found")
```

---

## 总结

### CRAFT 的视频输出

1. **简单视频**（Step 2）
   - 快速生成
   - 包含动作标注
   - 用于初步检查

2. **完整视频**（Step 3 之后）✅
   - 包含场景图可视化
   - 类似 REFLECT 的输出格式
   - 用于详细检查数据生成

### 使用建议

- **快速检查**：使用简单视频
- **详细检查**：使用完整视频
- **对比 REFLECT**：使用完整视频（格式更相似）

---

**更新日期**：2024年
**相关文件**：
- `utils/video_generator.py` - 视频生成器实现
- `demo1.ipynb` - Step 2 和 Step 3 的视频生成代码

