# STEP 1 (ALTERNATIVE) 和 STEP 2 使用说明

## 问题描述

在使用 `STEP 1 (ALTERNATIVE): LOAD PRE-GENERATED REFLECT DATA` 加载 REFLECT 预生成数据后，执行 `STEP 2` 时，可视化显示的不是从 REFLECT 框架加载的预生成数据，而是尝试从其他路径重新加载数据。

## 问题原因

`STEP 2` 的代码没有检查 `STEP 1 (ALTERNATIVE)` 中已经加载的 `frames` 变量，而是总是尝试从 `ego_img` 目录或从 events 中重新提取 frames。

## 修复方案

已修复 `STEP 2` 的代码，现在会：

1. **首先检查** `STEP 1 (ALTERNATIVE)` 中是否已经加载了 `frames`
2. **如果已加载**，直接使用这些 frames（来自 REFLECT 的预生成数据）
3. **如果未加载**，才执行原来的逻辑（从 `ego_img` 目录加载或从 events 提取）

## 使用方法

### STEP 1 (ALTERNATIVE): 加载预生成数据

```python
# STEP 1 (ALTERNATIVE): LOAD PRE-GENERATED REFLECT DATA
# 设置 REFLECT 数据路径
REFLECT_DATA_ROOT = "/home/fdse/zzy/reflect"
TASK_NAME = "makeCoffee-1"  # 或 "boilWater-1"
TASK_FOLDER = f"thor_tasks/makeCoffee/{TASK_NAME}"

reflect_task_path = os.path.join(REFLECT_DATA_ROOT, TASK_FOLDER)

# 加载数据
# - task.json
# - events/step_*.pickle
# - ego_img/img_step_*.png
# - original-video.mp4

# 这些数据会被存储在以下变量中：
# - task_info_craft: 任务信息
# - events_craft: 事件列表
# - frames: 帧列表（从 ego_img 目录加载）
# - frame_annotations: 帧标注列表
```

### STEP 2: 视频显示

执行 `STEP 2` 时，代码会自动：

1. 检查 `frames` 变量是否存在且不为空
2. 如果存在，使用这些 frames（来自 STEP 1 (ALTERNATIVE)）
3. 如果不存在，尝试从 `ego_img` 目录加载或从 events 提取

**重要提示**：
- 如果执行了 `STEP 1 (ALTERNATIVE)`，`STEP 2` 会自动使用已加载的数据
- 不需要修改 `STEP 2` 的代码
- 可视化会正确显示从 REFLECT 预生成数据加载的 frames

## 数据路径说明

### REFLECT 数据目录结构

```
/home/fdse/zzy/reflect/
└── thor_tasks/
    └── makeCoffee/
        └── makeCoffee-1/
            ├── task.json              # 任务配置
            ├── events/                # 事件文件
            │   ├── step_0.pickle
            │   ├── step_1.pickle
            │   └── ...
            ├── ego_img/               # 帧图像
            │   ├── img_step_0.png
            │   ├── img_step_1.png
            │   └── ...
            └── original-video.mp4     # 原始视频
```

### 在 STEP 1 (ALTERNATIVE) 中设置路径

```python
REFLECT_DATA_ROOT = "/home/fdse/zzy/reflect"
TASK_NAME = "makeCoffee-1"  # 根据实际任务修改
TASK_FOLDER = f"thor_tasks/makeCoffee/{TASK_NAME}"
```

## 验证修复

执行以下步骤验证修复是否成功：

1. **执行 STEP 1 (ALTERNATIVE)**：
   - 应该看到 "✅ Loaded X frames" 的消息
   - 确认 `frames` 变量已创建

2. **执行 STEP 2**：
   - 应该看到 "✅ Using X frames from STEP 1 (ALTERNATIVE)" 的消息
   - 可视化应该显示从 REFLECT 预生成数据加载的 frames

3. **检查输出**：
   - 视频应该包含从 `ego_img` 目录加载的帧
   - 帧标注应该正确显示

## 常见问题

### Q: STEP 2 仍然显示 "No frames from STEP 1 (ALTERNATIVE)"

**A**: 检查是否在同一个 notebook session 中执行了 STEP 1 (ALTERNATIVE)。确保：
- STEP 1 (ALTERNATIVE) 的 cell 已执行
- `frames` 变量已创建且不为空
- 没有重启 kernel

### Q: 数据路径不正确

**A**: 检查 `REFLECT_DATA_ROOT` 和 `TASK_NAME` 是否正确：
```python
# 验证路径是否存在
import os
reflect_task_path = os.path.join(REFLECT_DATA_ROOT, TASK_FOLDER)
print(f"Path exists: {os.path.exists(reflect_task_path)}")
print(f"Path: {reflect_task_path}")
```

### Q: 帧数量不匹配

**A**: 检查 `ego_img` 目录中的图像文件：
```python
import glob
img_files = glob.glob(os.path.join(reflect_task_path, "ego_img", "img_step_*.png"))
print(f"Found {len(img_files)} image files")
```

## 技术细节

### 修复的代码位置

- **文件**: `demo1.ipynb`
- **Cell**: STEP 2 (Cell index 11)
- **修改**: 在帧提取逻辑开始处添加了检查

### 修复后的逻辑流程

```
STEP 2 开始
  ↓
检查 frames 是否已存在 (来自 STEP 1 (ALTERNATIVE))
  ↓
[是] → 使用已存在的 frames
  ↓
[否] → 尝试从 ego_img 目录加载
  ↓
[仍无 frames] → 从 events 提取
  ↓
生成视频
```

## 总结

修复后，`STEP 2` 现在可以正确使用 `STEP 1 (ALTERNATIVE)` 中加载的 REFLECT 预生成数据。只需按顺序执行两个步骤，无需额外配置。

