# 框架对比脚本运行指南

## 快速开始

### 1. 基本运行

```bash
cd /home/fdse/zzy/craft
python3 run_framework_comparison.py
```

### 2. 运行前检查

确保以下路径和文件存在：

```bash
# 检查数据集是否存在
ls -lh /home/fdse/zzy/reflect/thor_tasks/makeCoffee/makeCoffee-1/original-video.mp4

# 检查任务JSON文件
ls -lh /home/fdse/zzy/reflect/thor_tasks/makeCoffee/makeCoffee-1/task.json

# 检查状态摘要（REFLECT）
ls -lh /home/fdse/zzy/reflect/main/state_summary/makeCoffee/makeCoffee-1/
```

## 运行输出

脚本会输出：

1. **控制台输出**：
   - REFLECT框架的详细统计
   - CRAFT框架的详细统计
   - 对比摘要

2. **JSON报告文件**：
   - 位置：`output/comparison/framework_comparison_YYYYMMDD_HHMMSS.json`
   - 包含所有详细的统计数据

3. **Markdown报告**：
   - 位置：`FRAMEWORK_COMPARISON_REPORT.md`
   - 包含格式化的对比分析

## 查看结果

### 查看JSON报告

```bash
# 查看最新的报告
ls -lt output/comparison/ | head -5

# 查看报告内容
cat output/comparison/framework_comparison_*.json | jq .
```

### 查看Markdown报告

```bash
# 查看报告
cat FRAMEWORK_COMPARISON_REPORT.md

# 或者在编辑器中打开
code FRAMEWORK_COMPARISON_REPORT.md
```

## LLM配置

脚本已内置poloapi配置（与demo1.ipynb相同）：

- **API_KEY**: `sk-wJJVkr6BUx8LruNeHNUCdmE1ARiB4qpLcdHHr3p4zVZTt8Fr`
- **BASE_URL**: `https://poloai.top/v1`
- **Model**: `gpt-3.5-turbo`

脚本会自动使用这些配置初始化LLM，无需额外设置环境变量。

如果你需要修改配置，可以编辑 `run_framework_comparison.py` 文件中的以下变量：

```python
API_KEY = "your-api-key-here"
POLOAPI_BASE_URL = "https://poloai.top/v1"
```

### 如果需要查看详细错误信息

```bash
# 运行并保存完整输出
python3 run_framework_comparison.py 2>&1 | tee comparison_output.log
```

## 脚本功能说明

### REFLECT框架检查的步骤

1. **数据加载**：
   - 加载视频文件
   - 分析视频统计信息（帧数、时长、大小等）
   - 加载任务JSON配置

2. **状态摘要**：
   - 检查状态摘要文件夹
   - 统计文件数量和大小

3. **LLM推理**：
   - 检查LLM响应文件是否存在
   - 如果有，分析响应内容

### CRAFT框架检查的步骤

1. **数据加载**：
   - 加载视频文件
   - 加载任务JSON配置

2. **场景图生成**：
   - 尝试从事件文件生成场景图
   - 统计场景图数量

3. **约束生成**：
   - 检查LLM是否可用
   - 如果可用，可以生成约束

4. **失败检测**：
   - 基于约束进行失败检测
   - 统计违规数量

5. **渐进式解释**：
   - 生成失败解释
   - 需要LLM支持

## 常见问题

### Q: 脚本提示事件文件未找到？

A: REFLECT使用pickle格式存储事件，脚本会查找JSON格式。这是正常的，不影响基本统计。

### Q: LLM功能无法使用？

A: 需要设置OPENAI_API_KEY环境变量。如果不设置，相关步骤会显示警告但不会报错。

### Q: 如何运行其他数据集？

A: 修改脚本中的 `folder_name` 变量，例如：
```python
folder_name = 'makeCoffee-2'  # 或其他数据集名称
```

### Q: 输出目录不存在？

A: 脚本会自动创建 `output/comparison/` 目录，无需手动创建。

## 完整运行示例

```bash
# 1. 进入工作目录
cd /home/fdse/zzy/craft

# 2. 运行对比脚本
python3 run_framework_comparison.py

# 3. 查看输出（等待运行完成）
# 脚本会自动输出统计信息

# 4. 查看生成的报告
cat FRAMEWORK_COMPARISON_REPORT.md
```

## 输出示例

运行成功后，你会看到类似以下的输出：

```
================================================================================
Framework Comparison: REFLECT vs CRAFT
Dataset: makeCoffee-1
Video: original-video.mp4
================================================================================

================================================================================
Running REFLECT Framework...
================================================================================

================================================================================
REFLECT Framework - Detailed Statistics
================================================================================

Total Duration: 0.02 seconds
Steps: 3
Errors: 0
Warnings: 1

...

================================================================================
Comparison Summary
================================================================================
REFLECT Framework:
  - Total Duration: 0.02s
  - Steps: 3
  - Errors: 0
  - Warnings: 1

CRAFT Framework:
  - Total Duration: 0.02s
  - Steps: 5
  - Errors: 0
  - Warnings: 3

✅ Report saved to: output/comparison/framework_comparison_20251217_170833.json
```

