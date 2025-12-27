# DETIC + CLIP 集成说明

## 概述

已成功将 **DETIC + CLIP + ByteTrack + Memory** 方案集成到 CRAFT++ 框架中，作为 MDETR 的替代方案。

## 主要更改

### 1. 新增文件

#### `perception/detic_clip_detector.py`
- **DeticClipDetector** 类：实现 DETIC + CLIP 检测器
- **功能**：
  - DETIC 开放词表检测（21k 类别）
  - CLIP 语义过滤和 prompt 扩展
  - ByteTrack 多目标跟踪（可选）
  - 与 Environment Memory 集成

#### `DETIC_CLIP_SETUP.md`
- 详细的安装和配置指南
- 故障排除说明

### 2. 修改的文件

#### `demo2.ipynb`
- **新增 Step 4 (Alternative)**：DETIC + CLIP 检测器初始化
- **保留原 Step 4**：MDETR 检测器初始化（用于对比）
- **添加选择机制**：通过 `DETECTION_METHOD` 变量选择检测方法
- **更新 Step 6 说明**：添加两种方法的对比说明

#### `Method.md`
- **更新 Section 1.5.1**：添加 DETIC + CLIP 方案说明
- **更新 Section 1.5.4**：添加两种方案的完整流程对比

## 使用方法

### 在 demo2.ipynb 中

1. **选择检测方法**（在 Step 4 Alternative 中）：
   ```python
   DETECTION_METHOD = 'detic_clip'  # 或 'mdetr'
   ```

2. **如果选择 DETIC + CLIP**：
   - 确保已安装依赖（见 `DETIC_CLIP_SETUP.md`）
   - 运行 Step 4 Alternative cell
   - 如果 DETIC 不可用，会自动回退到 MDETR

3. **如果选择 MDETR**：
   - 运行原 Step 4 cell
   - 使用原有的 MDETR 检测流程

## 方案对比

| 特性 | DETIC + CLIP | MDETR |
|------|-------------|-------|
| **检测能力** | 21k 类别，更强 | 开放词表，但可能检测不到 |
| **语义过滤** | CLIP 自动过滤 | 需要手动配置 |
| **Prompt 扩展** | 自动扩展（"cup" → "a cup", "the cup"） | 不支持 |
| **跟踪** | ByteTrack 内置 | 需要单独配置 |
| **依赖** | detectron2, CLIP, ByteTrack | transformers, timm |
| **设置复杂度** | 中等 | 简单 |

## 核心优势

### DETIC + CLIP 方案

1. **更强的检测能力**
   - DETIC 支持 21k 类别的开放词表
   - 比 MDETR 更鲁棒，检测成功率更高

2. **CLIP 语义过滤**
   - 自动扩展对象名称（prompt expansion）
   - 通过语义相似度过滤误检
   - 提高检测准确性

3. **内置跟踪支持**
   - ByteTrack 提供稳定的多目标跟踪
   - 处理遮挡和 ID 切换
   - 与 Environment Memory 无缝集成

4. **更好的 Memory 集成**
   - 跟踪 ID 提供稳定的对象标识
   - 时序平滑更准确
   - 遮挡预测更可靠

## 工作流程

```
RGB-D Stream
    ↓
DETIC Detection (21k classes)
    ↓
CLIP Semantic Filtering
    - Prompt expansion
    - Similarity filtering (threshold: 0.25)
    ↓
ByteTrack Tracking
    - Multi-object tracking
    - Handle occlusion
    ↓
Scene Graph Construction
    - Nodes with DETIC + CLIP confidence
    - Edges with spatial relations
    ↓
Environment Memory
    - Temporal smoothing
    - Occlusion handling
    ↓
Smoothed Scene Graph
```

## 安装要求

### 必需依赖

```bash
# DETIC
git clone https://github.com/facebookresearch/Detic.git
cd Detic && pip install -r requirements.txt

# CLIP
pip install git+https://github.com/openai/CLIP.git

# ByteTrack (可选)
pip install byte-track
```

详细安装步骤见 `DETIC_CLIP_SETUP.md`。

## 配置参数

### DeticClipDetector 参数

```python
detector = DeticClipDetector(
    device="cuda:0",           # 设备
    detic_threshold=0.3,       # DETIC 检测阈值（0.0-1.0）
    clip_threshold=0.25,       # CLIP 语义相似度阈值（0.0-1.0）
    use_tracking=True          # 是否启用 ByteTrack 跟踪
)
```

### 阈值调整建议

- **detic_threshold**: 
  - 0.3（默认）：平衡准确率和召回率
  - 0.2：提高召回率（更多检测）
  - 0.4：提高准确率（更少误检）

- **clip_threshold**:
  - 0.25（默认）：平衡语义匹配
  - 0.2：更宽松的匹配
  - 0.3：更严格的匹配

## 故障排除

### DETIC 不可用
- 检查 DETIC 是否正确安装
- 确认模型权重已下载
- 查看错误日志

### CLIP 不可用
- 安装 CLIP：`pip install git+https://github.com/openai/CLIP.git`
- 检查网络连接（首次运行需下载模型）

### 自动回退
- 如果 DETIC + CLIP 初始化失败，会自动回退到 MDETR
- 检查控制台输出了解回退原因

## 下一步

1. **测试 DETIC + CLIP 方案**：
   - 运行 demo2.ipynb
   - 选择 `DETECTION_METHOD = 'detic_clip'`
   - 观察检测结果

2. **对比两种方案**：
   - 使用相同数据测试两种方法
   - 比较检测准确率和性能

3. **调整参数**：
   - 根据实际场景调整阈值
   - 优化 prompt 扩展策略

## 参考

- DETIC 论文：https://arxiv.org/abs/2201.02605
- CLIP 论文：https://arxiv.org/abs/2103.00020
- ByteTrack 论文：https://arxiv.org/abs/2110.06864
- Method.md：更新的方法说明

