# CRAFT vs REFLECT 实验设计文档

本文档描述了 CRAFT vs REFLECT 失败检测实验的完整工程流程，从 0 → 可写论文结果的完整实现指南。

---

## 一、整体工程蓝图

实验本质是一个 **Failure Detection Benchmark**，可抽象为 5 个模块：

```
AI2-THOR
   ↓
Task Executor（任务执行 + failure injection）
   ↓
Perception Output（scene graph / state）
   ↓
Failure Detector
   ├─ REFLECT (LLM-only)
   └─ CRAFT   (constraints + code)
   ↓
Evaluator（Acc / FPR / Attribution）
```

---

## 二、建议的工程目录结构

建议的目录结构（可直接照抄）：

```
craft-experiments/
├── environments/
│   └── ai2thor_env.py          # AI2-THOR wrapper
│
├── tasks/
│   ├── task_defs.json          # 30个任务定义
│   ├── task_executor.py        # 执行任务
│   └── ground_truth.py         # 成功条件 / GT
│
├── failure_injection/
│   ├── injector.py             # 注入逻辑
│   ├── failure_types.py        # 枚举失败类型
│   └── injection_config.json
│
├── perception/
│   ├── scene_graph.py          # scene graph 构建
│   └── memory.py               # environment memory
│
├── detectors/
│   ├── reflect_detector.py
│   ├── craft_detector.py
│   └── constraints/
│       ├── coffee.py
│       └── container.py
│
├── evaluation/
│   ├── metrics.py              # Acc / FPR / Attribution
│   └── evaluate.py
│
├── runs/
│   ├── raw_logs/
│   ├── reflect/
│   ├── craft/
│   └── results.csv
│
├── scripts/
│   ├── run_reflect.py
│   ├── run_craft.py
│   └── run_all.py
│
└── README.md
```

👉 **重要**：之后所有实验 = 往 `runs/` 里不断丢结果

---

## 三、Step 1：任务与 Ground Truth

### 3.1 任务定义（task_defs.json）

**原则**：先不要 LLM 生成任务，先用静态任务：

```json
{
  "make_coffee": {
    "scene": "Kitchen",
    "steps": [
      "Pickup(cup)",
      "PutIn(cup, coffee_machine)",
      "ToggleOn(coffee_machine)"
    ]
  }
}
```

👉 **原则**：
- 动作必须是你能执行的 API
- 失败注入时能精确定位 step index

### 3.2 Ground Truth（成功条件）

Ground Truth 使用代码定义，而不是文字描述。这是 CRAFT 的核心优势之一。

```python
# tasks/ground_truth.py
def make_coffee_success(state):
    assert state["coffee_machine"].is_on
    assert state["cup"].contains("coffee")
```

👉 **GT 是代码，不是文字**

---

## 四、Step 2：Failure Injection（最重要的实验资产）

### 4.1 定义失败类型

```python
# failure_types.py
from enum import Enum

class FailureType(Enum):
    MISSING_PRECONDITION = 1
    PHYSICAL_IMPOSSIBLE = 2
    CAUSAL_BREAK = 3
    PERCEPTION_NOISE = 4
```

### 4.2 注入方式（示例）

#### 🔴 前置条件缺失

```python
def inject_missing_precondition(task):
    task.steps.remove("PutIn(cup, coffee_machine)")
```

#### 🔴 物理不可能

```python
def inject_capacity_violation(state):
    state["cup"].volume = 1000
```

#### 🔴 感知噪声（超重要）

```python
def inject_occlusion(perception):
    perception.hide("cup", frames=3)
```

👉 这一步是 RQ3 的关键

### 4.3 注入配置（实验可控）

```json
{
  "make_coffee": [
    {
      "type": "MISSING_PRECONDITION",
      "step": 1
    },
    {
      "type": "PERCEPTION_NOISE",
      "frames": 5
    }
  ]
}
```

---

## 五、Step 3：REFLECT vs CRAFT（如何并行跑）

### 5.1 REFLECT（baseline）

```python
# reflect_detector.py
def detect_failure(video, caption):
    prompt = f"""
    Here is a robot execution.
    {caption}
    Did the task fail? Why?
    """
    return call_llm(prompt)
```

**输出**：

```json
{
  "failed": true,
  "reason": "The robot may not have placed the cup correctly."
}
```

👉 **不做验证，只存**

### 5.2 CRAFT（论文主角）

```python
# craft_detector.py
def detect_failure(scene_graph, memory):
    run_constraints(scene_graph)
    run_physics_checks(scene_graph)
    run_temporal_checks(memory)
```

**失败 = assert 报错**

```json
{
  "failed": true,
  "type": "CapacityViolation",
  "at_step": 2
}
```

---

## 六、Step 4：评估指标怎么"工程化"

### 6.1 保存统一日志格式

```json
{
  "task": "make_coffee",
  "failure_injected": true,
  "failure_type": "MISSING_PRECONDITION",
  "detected": true,
  "detector": "CRAFT",
  "attribution_correct": true
}
```

### 6.2 自动算指标

```python
# evaluation/metrics.py
def accuracy(results):
    return sum(r.correct for r in results) / len(results)
```

---

## 七、Step 5：Cursor 怎么用（重点）

### 7.1 正确姿势

1. 你写目录结构 + 空文件
2. 在 Cursor 里：
   - 一次只让它补一个函数
   - 给它"接口 + 示例"

### 7.2 ❌ 错误用法

```
"帮我实现整个实验系统"
```

### 7.3 ✅ 正确提示词示例

```
Implement inject_missing_precondition(task)
Input: task with steps list
Output: modified task
Do NOT change other fields
```

---

## 八、Day 1 Checklist（今天就该做的 6 件事）

### Day 1 Checklist：

- ✅ 搭目录结构
- ✅ 写 3 个任务 + GT
- ✅ 写 2 种 failure injection
- ✅ 跑一条「无失败」baseline
- ✅ 跑一条「有失败」CRAFT
- ✅ 输出一行 results.csv

---

## 九、实验流程总结

### 9.1 完整实验流程

1. **数据生成**：使用 AI2-THOR 生成任务执行数据
2. **Failure Injection**：注入失败场景
3. **感知输出**：生成 scene graph / state
4. **失败检测**：
   - REFLECT：LLM-only 检测
   - CRAFT：constraints + code 检测
5. **评估**：计算 Acc / FPR / Attribution

### 9.2 关键设计原则

1. **GT 是代码**：成功条件用代码定义，不是文字
2. **Failure Injection 可控**：通过配置文件精确控制失败类型和位置
3. **统一日志格式**：所有结果保存为统一格式，便于后续分析
4. **并行运行**：REFLECT 和 CRAFT 可以并行运行，结果独立保存

---

## 十、下一步建议

如果你愿意，下一步可以帮你：

- 写 `run_all.py`
- 给你 `results.csv` 的标准论文格式
- 把现有的 detic+clip 接到 perception 模块里

你只要说一句：
👉 **"下一步我先实现哪一块？"**

---

## 附录：与 Method.md 的关系

本实验设计文档（Experiment.md）与 Method.md 的关系：

- **Method.md**：描述 CRAFT++ 框架的方法论和技术细节
- **Experiment.md**：描述如何设计和运行 CRAFT vs REFLECT 的对比实验

两者结合使用：
- 使用 Method.md 理解 CRAFT++ 的技术原理
- 使用 Experiment.md 设计和运行实验

---

## 参考

- [Method.md](./Method.md)：CRAFT++ 框架完整方法论
- [demo3.ipynb](./demo3.ipynb)：CRAFT 完整工作流示例
- [demo4.ipynb](./demo4.ipynb)：DETIC + CLIP 真实环境检测示例

