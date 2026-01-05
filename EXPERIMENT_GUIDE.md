# CRAFT vs REFLECT 实验详细指南

本文档详细说明如何开始和运行 CRAFT vs REFLECT 失败检测实验，逐步解释每个 checklist 项。

---

## 📋 快速开始

```bash
# 在项目根目录运行
python3 day_one_checklist.py
```

这将自动完成所有 6 个步骤。下面详细解释每个步骤。

---

## 步骤详解

### Step 1: 搭目录结构

**目的**：创建实验所需的完整目录结构

**代码位置**：`day_one_checklist.py` 的 `create_directory_structure()` 函数

**创建的结构**：
```
craft-experiments/
├── environments/          # AI2-THOR 环境封装（待实现）
├── tasks/                 # 任务定义和 Ground Truth
├── failure_injection/     # 失败注入模块
├── perception/            # 感知模块（scene graph 构建）
├── detectors/             # 失败检测器
│   └── constraints/       # 约束定义
├── evaluation/            # 评估模块
├── runs/                  # 实验结果存储
│   ├── raw_logs/         # 原始日志
│   ├── reflect/          # REFLECT 检测结果
│   └── craft/            # CRAFT 检测结果
└── scripts/               # 运行脚本
```

**为什么需要这个结构**：
- **模块化设计**：每个模块职责明确，便于维护
- **结果管理**：所有实验结果统一存储在 `runs/` 目录
- **可扩展性**：后续可以轻松添加新模块

**手动创建**（如果需要）：
```bash
mkdir -p craft-experiments/{environments,tasks,failure_injection,perception,detectors/constraints,evaluation,runs/{raw_logs,reflect,craft},scripts}
```

---

### Step 2: 写 3 个任务 + GT

**目的**：定义实验使用的任务和对应的成功条件（Ground Truth）

**代码位置**：`day_one_checklist.py` 的 `create_task_definitions()` 函数

#### 2.1 任务定义（task_defs.json）

**位置**：`craft-experiments/tasks/task_defs.json`

**格式**：
```json
{
  "make_coffee": {
    "scene": "FloorPlan16",           # AI2THOR 场景
    "object_list": ["Mug", ...],      # 任务涉及的对象
    "actions": [                       # 动作序列
      "navigate_to_obj, Mug",
      "pick_up, Mug",
      ...
    ],
    "success_condition": "..."        # 成功条件的文字描述
  }
}
```

**当前包含的 3 个任务**：
1. **make_coffee**：制作咖啡
   - 场景：FloorPlan16（厨房场景）
   - 对象：Mug, CoffeeMachine, Sink, CounterTop
   - 动作：导航到杯子 → 拿起杯子 → 放到咖啡机 → 开启咖啡机

2. **make_tea**：泡茶
   - 场景：FloorPlan16
   - 对象：Mug, Kettle, Sink, CounterTop
   - 动作：拿起水壶 → 放到水槽 → 开水龙头 → 关水龙头 → 倒水到杯子

3. **clean_mug**：清洗杯子
   - 场景：FloorPlan16
   - 对象：Mug, Sink, CounterTop
   - 动作：拿起杯子 → 放到水槽 → 开水龙头 → 关水龙头 → 放到台面

**如何添加新任务**：
1. 编辑 `task_defs.json`
2. 在 `ground_truth.py` 中添加对应的 GT 函数

#### 2.2 Ground Truth 函数（ground_truth.py）

**位置**：`craft-experiments/tasks/ground_truth.py`

**核心思想**：**GT 是代码，不是文字** - 这是 CRAFT 的核心优势

**格式**：
```python
def make_coffee_success(state: Dict[str, Any]) -> bool:
    """Check if make_coffee task succeeded"""
    try:
        # 检查咖啡机是否开启
        assert state.get("coffee_machine", {}).get("is_on", False)
        # 检查杯子是否包含咖啡
        assert "coffee" in state.get("mug", {}).get("contains", [])
        return True
    except AssertionError:
        return False
```

**为什么用代码而不是文字**：
- ✅ **可执行**：可以直接在代码中调用和验证
- ✅ **确定性**：相同输入总是得到相同输出
- ✅ **精确性**：避免了自然语言描述的歧义
- ✅ **可测试**：可以编写单元测试验证 GT 函数

**如何使用 GT 函数**：
```python
from tasks.ground_truth import check_task_success

state = {
    "coffee_machine": {"is_on": True},
    "mug": {"contains": ["coffee"]}
}

success = check_task_success("make_coffee", state)
print(f"Task succeeded: {success}")  # True
```

---

### Step 3: 写 2 种 failure injection

**目的**：实现可控的失败注入机制，这是实验的核心资产

**代码位置**：
- `craft-experiments/failure_injection/failure_types.py`：失败类型枚举
- `craft-experiments/failure_injection/injector.py`：注入逻辑
- `craft-experiments/failure_injection/injection_config.json`：注入配置

#### 3.1 失败类型枚举

**位置**：`failure_injection/failure_types.py`

```python
class FailureType(Enum):
    MISSING_PRECONDITION = "MISSING_PRECONDITION"  # 前置条件缺失
    PHYSICAL_IMPOSSIBLE = "PHYSICAL_IMPOSSIBLE"    # 物理不可能
    CAUSAL_BREAK = "CAUSAL_BREAK"                  # 因果链断裂
    PERCEPTION_NOISE = "PERCEPTION_NOISE"          # 感知噪声
```

**为什么需要枚举**：
- 标准化失败类型，便于后续分析和统计
- 类型安全，避免拼写错误

#### 3.2 注入逻辑

**位置**：`failure_injection/injector.py`

**实现的 2 种注入方式**：

##### 方式 1: MISSING_PRECONDITION（前置条件缺失）

```python
def inject_missing_precondition(task, step_index):
    """移除前置条件步骤"""
    modified_task = task.copy()
    actions = modified_task.get("actions", []).copy()
    
    if 0 <= step_index < len(actions):
        removed_action = actions.pop(step_index)
        modified_task["actions"] = actions
        # 标记注入的失败信息
        modified_task["injected_failure"] = {
            "type": "MISSING_PRECONDITION",
            "step": step_index,
            "removed_action": removed_action
        }
    
    return modified_task
```

**示例**：
- 原始任务：`["pick_up, Mug", "put_in, CoffeeMachine, Mug", "toggle_on, CoffeeMachine"]`
- 注入失败（移除步骤 1）：`["pick_up, Mug", "toggle_on, CoffeeMachine"]`
- 结果：杯子没有被放入咖啡机，但后续步骤仍会执行 → **失败场景**

##### 方式 2: PHYSICAL_IMPOSSIBLE（物理不可能）

```python
def inject_capacity_violation(state, object_name):
    """注入容量违反（物理不可能）"""
    modified_state = state.copy()
    if object_name in modified_state:
        obj_state = modified_state[object_name].copy()
        obj_state["volume"] = 1000      # 超大的容量
        obj_state["max_capacity"] = 100 # 但最大容量很小
        modified_state[object_name] = obj_state
    return modified_state
```

**示例**：
- 设置咖啡机容量为 1000，但最大容量为 100
- 这违反了物理约束 → **失败场景**

#### 3.3 注入配置

**位置**：`failure_injection/injection_config.json`

```json
{
  "make_coffee": [
    {
      "type": "MISSING_PRECONDITION",
      "step": 3
    },
    {
      "type": "PHYSICAL_IMPOSSIBLE",
      "object": "coffee_machine"
    }
  ]
}
```

**如何使用**：
```python
import json
from failure_injection.injector import inject_failure

# 加载配置
with open("failure_injection/injection_config.json") as f:
    config = json.load(f)

# 加载任务
with open("tasks/task_defs.json") as f:
    tasks = json.load(f)

# 注入失败
task = tasks["make_coffee"]
failure_config = config["make_coffee"][0]  # 第一个失败配置
failed_task = inject_failure(task, failure_config)
```

**为什么需要配置文件**：
- ✅ **可复现**：相同的配置总是产生相同的失败
- ✅ **可控性**：精确控制失败类型和位置
- ✅ **可扩展**：轻松添加新的失败场景

---

### Step 4: 跑一条「无失败」baseline

**目的**：建立基线，验证系统在正常情况下的行为

**代码位置**：`day_one_checklist.py` 的 `run_baseline_no_failure()` 函数

**当前实现（简化版）**：
```python
def run_baseline_no_failure(base_dir, task_name, task_def):
    """运行无失败的 baseline"""
    result = {
        "task": task_name,
        "failure_injected": False,
        "detected": False,          # 无失败时，不应该检测到失败
        "detector": "BASELINE",
        "status": "SUCCESS"
    }
    return result
```

**完整实现应该包括**：
1. **执行任务**：在 AI2THOR 中运行任务
2. **场景图生成**：从执行结果生成 scene graph
3. **检测器运行**：使用 REFLECT 或简单的检测器
4. **结果记录**：保存检测结果

**预期结果**：
- `failure_injected = False`
- `detected = False`（没有失败，所以不应该检测到失败）
- `status = "SUCCESS"`

**如何扩展**：
```python
# 伪代码
def run_baseline_no_failure(task_def):
    # 1. 执行任务（AI2THOR）
    events = execute_task_in_ai2thor(task_def)
    
    # 2. 生成场景图
    scene_graphs = build_scene_graphs(events)
    
    # 3. 运行检测器（REFLECT 或简单检测器）
    detection_result = reflect_detector.detect(scene_graphs)
    
    # 4. 验证 GT
    final_state = extract_state_from_scene_graph(scene_graphs[-1])
    gt_result = check_task_success(task_def["name"], final_state)
    
    # 5. 返回结果
    return {
        "task": task_def["name"],
        "failure_injected": False,
        "detected": detection_result["failed"],
        "detector": "BASELINE",
        "gt_success": gt_result,
        "status": "SUCCESS" if gt_result else "FAILED"
    }
```

---

### Step 5: 跑一条「有失败」CRAFT

**目的**：验证 CRAFT 能够检测到注入的失败

**代码位置**：`day_one_checklist.py` 的 `run_craft_with_failure()` 函数

**当前实现（简化版）**：
```python
def run_craft_with_failure(base_dir, task_name, task_def, failure_config):
    """运行有失败的 CRAFT 检测"""
    result = {
        "task": task_name,
        "failure_injected": True,
        "failure_type": failure_config.get("type"),
        "detected": True,              # CRAFT 应该检测到失败
        "detector": "CRAFT",
        "attribution_correct": True,   # 归因是否正确
        "status": "FAILURE_DETECTED"
    }
    return result
```

**完整实现应该包括**：
1. **注入失败**：使用 injector 修改任务或状态
2. **执行任务**：在 AI2THOR 中运行失败的任务
3. **场景图生成**：生成带失败的 scene graph
4. **CRAFT 检测**：
   - 生成约束
   - 验证约束（检查 pre/post conditions）
   - 检测失败
5. **归因分析**：确定失败的原因和位置

**预期结果**：
- `failure_injected = True`
- `detected = True`（CRAFT 检测到失败）
- `failure_type = "MISSING_PRECONDITION"` 或 `"PHYSICAL_IMPOSSIBLE"`
- `attribution_correct = True`（正确归因到失败的步骤）

**如何扩展**（参考 demo3.ipynb 的流程）：
```python
# 伪代码
def run_craft_with_failure(task_def, failure_config):
    # 1. 注入失败
    failed_task = inject_failure(task_def, failure_config)
    
    # 2. 执行任务（AI2THOR）
    events = execute_task_in_ai2thor(failed_task)
    
    # 3. 生成场景图（参考 demo3）
    scene_graphs = build_scene_graphs(events)
    
    # 4. CRAFT 检测（参考 demo3 Step 9-11）
    constraints = generate_constraints(task_def, scene_graphs[0])
    compiled_constraints = compile_constraints(constraints)
    
    violations = []
    for action_idx, action in enumerate(task_def["actions"]):
        # 检查 preconditions
        pre_sg = scene_graphs[action_idx - 1] if action_idx > 0 else scene_graphs[0]
        for constraint in get_preconditions(compiled_constraints, action_idx):
            if not validate_constraint(constraint, pre_sg):
                violations.append({
                    "type": "PRECONDITION_VIOLATION",
                    "step": action_idx,
                    "constraint": constraint
                })
                break  # CRAFT：precondition 失败后立即停止
        
        # 检查 postconditions
        post_sg = scene_graphs[action_idx + 1] if action_idx < len(scene_graphs) - 1 else scene_graphs[-1]
        for constraint in get_postconditions(compiled_constraints, action_idx):
            if not validate_constraint(constraint, post_sg):
                violations.append({
                    "type": "POSTCONDITION_VIOLATION",
                    "step": action_idx,
                    "constraint": constraint
                })
    
    # 5. 验证归因
    expected_failure_step = failure_config.get("step")
    detected_failure_step = violations[0]["step"] if violations else None
    attribution_correct = (expected_failure_step == detected_failure_step)
    
    # 6. 返回结果
    return {
        "task": task_def["name"],
        "failure_injected": True,
        "failure_type": failure_config.get("type"),
        "detected": len(violations) > 0,
        "detector": "CRAFT",
        "violations": violations,
        "attribution_correct": attribution_correct,
        "status": "FAILURE_DETECTED" if violations else "SUCCESS"
    }
```

**与 demo3 的关系**：
- demo3.ipynb 展示了完整的 CRAFT 工作流
- 可以参考 demo3 中的约束生成、编译和验证逻辑
- 关键步骤：
  - Step 9: 约束生成（Action-aware）
  - Step 10: 约束编译
  - Step 11: 失败检测（Precondition/Postcondition 验证）

---

### Step 6: 输出一行 results.csv

**目的**：保存实验结果，便于后续分析和论文写作

**代码位置**：`day_one_checklist.py` 的 `save_results_csv()` 函数

**位置**：`craft-experiments/runs/results.csv`

**格式**：
```csv
attribution_correct,detected,detector,failure_injected,failure_step,failure_type,status,task,timestamp
,False,BASELINE,False,,,SUCCESS,make_coffee,2026-01-05T23:33:10.370108
True,True,CRAFT,True,3,MISSING_PRECONDITION,FAILURE_DETECTED,make_coffee,2026-01-05T23:33:10.370137
```

**字段说明**：
- `task`: 任务名称
- `failure_injected`: 是否注入了失败（True/False）
- `failure_type`: 失败类型（如果注入了失败）
- `failure_step`: 失败发生的步骤（如果注入了失败）
- `detected`: 检测器是否检测到失败（True/False）
- `detector`: 检测器名称（BASELINE / CRAFT / REFLECT）
- `attribution_correct`: 归因是否正确（如果检测到失败）
- `status`: 状态（SUCCESS / FAILURE_DETECTED）
- `timestamp`: 时间戳

**如何查看结果**：
```bash
# 查看 CSV 文件
cat craft-experiments/runs/results.csv

# 使用 pandas 分析（Python）
import pandas as pd
df = pd.read_csv("craft-experiments/runs/results.csv")
print(df)

# 使用命令行工具
csvlook craft-experiments/runs/results.csv  # 如果安装了 csvkit
```

**如何扩展**：
1. 运行更多实验，追加到 CSV
2. 添加评估指标（Acc, FPR, Precision, Recall）
3. 按任务、失败类型分组统计
4. 生成可视化图表

---

## 🔄 完整工作流程

### 当前状态（简化版）

```bash
python3 day_one_checklist.py
```

这会在 `craft-experiments/` 目录生成所有基础文件，并运行一个简化的测试。

### 下一步：集成真实实现

#### 1. 集成 AI2THOR 执行（如果可用）

参考 `demo1.ipynb` 或 `utils/gen_data.py` 中的实现：

```python
from ai2thor.controller import Controller
from utils.gen_data import run_data_gen

# 在 run_baseline_no_failure 和 run_craft_with_failure 中
# 替换模拟执行为真实 AI2THOR 执行
events = run_data_gen(data_path=".", task=task_def)
```

#### 2. 集成 CRAFT 检测器

参考 `demo3.ipynb` 中的完整流程：

```python
from craft.core import SceneGraph
from craft.reasoning import ConstraintGenerator
from craft.perception import SceneAnalyzer

# 在 run_craft_with_failure 中
# 使用真实的 CRAFT 检测逻辑
scene_graphs = build_scene_graphs_from_events(events)
constraints = generate_action_aware_constraints(task_def, scene_graphs)
violations = validate_constraints(constraints, scene_graphs)
```

#### 3. 集成 REFLECT 检测器（baseline）

```python
from craft.reasoning import LLMPrompter, FailureAnalyzer

# 在 run_baseline_no_failure 中
# 使用 REFLECT 的 LLM-only 检测
prompter = LLMPrompter(api_key=os.getenv("OPENAI_API_KEY"))
result = prompter.detect_failure(scene_description, task_def)
```

#### 4. 运行完整实验

创建 `scripts/run_all.py`：

```python
import json
from pathlib import Path

# 加载所有任务
with open("tasks/task_defs.json") as f:
    tasks = json.load(f)

# 加载失败配置
with open("failure_injection/injection_config.json") as f:
    failure_configs = json.load(f)

results = []

# 对每个任务运行实验
for task_name, task_def in tasks.items():
    # Baseline（无失败）
    baseline_result = run_baseline_no_failure(task_def)
    results.append(baseline_result)
    
    # CRAFT（有失败）
    for failure_config in failure_configs.get(task_name, []):
        craft_result = run_craft_with_failure(task_def, failure_config)
        results.append(craft_result)

# 保存结果
save_results_csv(results)
```

---

## 📊 评估指标

### 基本指标

1. **准确率（Accuracy）**
   ```python
   accuracy = (TP + TN) / (TP + TN + FP + FN)
   ```

2. **假阳性率（False Positive Rate, FPR）**
   ```python
   FPR = FP / (FP + TN)
   ```

3. **归因准确率（Attribution Accuracy）**
   ```python
   attribution_acc = correct_attributions / total_detections
   ```

### 实现位置

应该在 `evaluation/metrics.py` 中实现：

```python
def calculate_metrics(results):
    """计算评估指标"""
    # 统计 TP, TN, FP, FN
    # 计算 Acc, FPR, Attribution Accuracy
    # 返回指标字典
    pass
```

---

## 🎯 下一步行动

### 立即可以做的：

1. **查看生成的文件**
   ```bash
   cd craft-experiments
   ls -R
   cat tasks/task_defs.json
   cat tasks/ground_truth.py
   cat failure_injection/injection_config.json
   cat runs/results.csv
   ```

2. **理解每个文件的作用**
   - 参考本文档的详细说明
   - 查看代码注释

3. **尝试修改配置**
   - 在 `task_defs.json` 中添加新任务
   - 在 `injection_config.json` 中添加新的失败配置
   - 重新运行 `day_one_checklist.py`

### 需要集成真实实现的：

1. **集成 AI2THOR**（如果可用）
   - 参考 `demo1.ipynb` 或 `utils/gen_data.py`
   - 修改 `run_baseline_no_failure` 和 `run_craft_with_failure`

2. **集成 CRAFT 检测器**
   - 参考 `demo3.ipynb` 的完整流程
   - 使用真实的约束生成和验证逻辑

3. **集成 REFLECT 检测器**
   - 实现 LLM-only 的失败检测
   - 作为 baseline 对比

4. **运行完整实验**
   - 创建 `scripts/run_all.py`
   - 运行所有任务和失败配置
   - 生成完整的 `results.csv`

---

## 📚 参考文档

- [Experiment.md](../Experiment.md): 实验设计文档
- [Method.md](../Method.md): CRAFT++ 框架方法论
- [demo3.ipynb](../demo3.ipynb): CRAFT 完整工作流示例
- [demo4.ipynb](../demo4.ipynb): DETIC + CLIP 真实环境检测示例

---

## ❓ 常见问题

### Q1: 如何添加新任务？

A: 
1. 编辑 `tasks/task_defs.json`，添加新任务定义
2. 在 `tasks/ground_truth.py` 中添加对应的 GT 函数
3. 在 `failure_injection/injection_config.json` 中添加该任务的失败配置

### Q2: 如何添加新的失败类型？

A:
1. 在 `failure_injection/failure_types.py` 中添加新的枚举值
2. 在 `failure_injection/injector.py` 中实现对应的注入函数
3. 在 `failure_injection/injection_config.json` 中使用新类型

### Q3: 当前实现是模拟的，如何集成真实实现？

A:
- 参考本文档的"下一步：集成真实实现"部分
- 参考 `demo3.ipynb` 中的完整 CRAFT 流程
- 参考 `demo1.ipynb` 或 `utils/gen_data.py` 中的 AI2THOR 集成

### Q4: results.csv 的格式可以修改吗？

A:
- 可以，但建议保持核心字段不变（task, detector, detected, failure_injected）
- 可以添加新字段（如 confidence, execution_time 等）
- 修改 `save_results_csv()` 函数

---

如有其他问题，请参考相关文档或查看代码注释。

