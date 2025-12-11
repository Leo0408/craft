# REFLECT 风格的数据生成和失败注入使用指南

## 概述

CRAFT 框架现在支持与 REFLECT 相同的模块化数据生成和失败注入方法。主要改进包括：

1. **模块化的动作原语** (`utils/action_primitives.py`)
2. **任务工具类** (`utils/task_utils.py`)
3. **数据生成主函数** (`utils/gen_data.py`)
4. **失败注入支持** (drop, failed_action, missing_step 等)

## 文件结构

```
craft/utils/
├── constants.py          # 常量和映射定义
├── task_utils.py         # TaskUtil 类（状态管理）
├── action_primitives.py  # 动作原语（navigate_to_obj, pick_up, put_in 等）
└── gen_data.py          # 数据生成主函数（支持失败注入）
```

## 使用方法

### 方法 1: 使用 gen_data.py（推荐）

```python
from craft.utils.gen_data import run_data_gen

# 定义任务配置
task = {
    "name": "make coffee",
    "task_idx": 5,
    "num_samples": 1,
    "failure_injection": True,  # 启用失败注入
    "folder_name": "makeCoffee-1",
    "scene": "FloorPlan16",
    "actions": [
        "navigate_to_obj, Mug",
        "pick_up, Mug",
        "navigate_to_obj, Sink",
        "put_on, SinkBasin, Mug",
        "toggle_on, Faucet",
        "toggle_off, Faucet",
        "pick_up, Mug",
        "navigate_to_obj, CoffeeMachine",
        "put_in, CoffeeMachine, Mug",
    ],
    # 可选：指定失败类型
    # "chosen_failure": "drop",  # 或 "failed_action", "missing_step"
}

# 运行数据生成
run_data_gen(data_path=".", task=task)
```

### 方法 2: 直接使用动作原语

```python
from craft.utils.task_utils import TaskUtil
from craft.utils import action_primitives as ap
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

# 初始化 Controller
controller = Controller(
    agentMode="default",
    scene="FloorPlan16",
    platform=CloudRendering,
    # ... 其他参数
)

# 获取可达位置
reachable_positions = controller.step(action="GetReachablePositions").metadata["actionReturn"]

# 创建 TaskUtil
taskUtil = TaskUtil(
    folder_name="makeCoffee/test",
    controller=controller,
    reachable_positions=reachable_positions,
    failure_injection=False,
    index=0,
    repo_path="."
)

# 执行动作
ap.navigate_to_obj(taskUtil, "Mug")
ap.pick_up(taskUtil, "Mug")
ap.navigate_to_obj(taskUtil, "Sink")
ap.put_on(taskUtil, "Mug", "SinkBasin")
ap.toggle_on(taskUtil, "Faucet")
```

## 失败注入类型

### 1. Drop (掉落失败)
在导航过程中随机掉落手中的物体。

```python
task = {
    ...
    "failure_injection": True,
    "chosen_failure": "drop",
}
```

### 2. Failed Action (动作失败)
模拟动作执行失败。

```python
task = {
    ...
    "failure_injection": True,
    "chosen_failure": "failed_action",
}
```

### 3. Missing Step (缺失步骤)
跳过某个关键动作步骤。

```python
task = {
    ...
    "failure_injection": True,
    "chosen_failure": "missing_step",
    # 可选：指定要跳过的步骤索引
    # "specified_missing_steps": [3, 5],
}
```

## 主要改进

### 与 REFLECT 的一致性

1. **相同的 Controller 初始化参数**
2. **模块化的动作原语系统**
3. **TaskUtil 状态管理**
4. **BFS 路径规划**
5. **失败注入机制**

### 与之前 CRAFT 实现的区别

| 特性 | 旧实现 | 新实现（REFLECT 风格） |
|------|--------|----------------------|
| 动作执行 | 内联 if-elif | 模块化函数 |
| 状态管理 | 局部变量 | TaskUtil 类 |
| 路径规划 | 简单最近距离 | BFS 路径规划 |
| 失败注入 | ❌ 不支持 | ✅ 支持多种类型 |
| 代码复用 | ❌ 不易复用 | ✅ 高度模块化 |

## 在 demo1.ipynb 中使用

更新 Step 1 的代码：

```python
# STEP 1: DATA GENERATION WITH REFLECT-STYLE MODULAR APPROACH
from craft.utils.gen_data import run_data_gen

# Task configuration
task_info_craft = {
    "name": "make coffee",
    "task_idx": 5,
    "num_samples": 1,
    "failure_injection": True,  # Enable failure injection
    "folder_name": "makeCoffee-1",
    "scene": "FloorPlan16",
    "actions": [
        "navigate_to_obj, Mug",
        "pick_up, Mug",
        "navigate_to_obj, Sink",
        "put_on, SinkBasin, Mug",
        "toggle_on, Faucet",
        "toggle_off, Faucet",
        "pick_up, Mug",
        "navigate_to_obj, CoffeeMachine",
        "put_in, CoffeeMachine, Mug",
    ],
}

# Run data generation
print("Starting data generation...")
run_data_gen(data_path=".", task=task_info_craft)

# Get events from TaskUtil (if needed for further processing)
# events_craft = taskUtil.events
```

## 注意事项

1. **路径设置**: 确保 craft 模块在 Python 路径中
2. **AI2THOR 安装**: 需要安装 `pip install ai2thor`
3. **数据保存**: 生成的数据会保存在 `thor_tasks/` 目录下
4. **失败注入**: 失败注入是随机的，每次运行可能不同

## 示例输出

数据生成后会创建以下文件结构：

```
thor_tasks/
└── makeCoffee/
    ├── makeCoffee-1.pickle          # 失败注入记录
    └── makeCoffee-1/                # 具体样本数据
        ├── task.json                # 任务配置
        ├── interact_actions.pickle  # 交互动作记录
        └── nav_actions.pickle       # 导航动作记录
```

## 下一步

- 查看 `examples/gen_data_example.py` 获取完整示例
- 参考 `utils/action_primitives.py` 了解所有可用的动作原语
- 查看 `utils/task_utils.py` 了解 TaskUtil 的完整功能

