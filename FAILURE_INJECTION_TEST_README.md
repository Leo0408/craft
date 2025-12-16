# 失败注入测试说明

## 概述

本测试脚本实现了6个从REFLECT论文中提取的真实场景失败检测问题，并对比CRAFT和REFLECT两种方法的检测准确率。

## 测试案例

1. **视觉遮挡导致的误失败** (Case 1)
   - 场景：物体被抓取后被机械臂遮挡，从场景图中消失
   - REFLECT误判：认为物体掉落，判定失败
   - CRAFT改进：使用环境记忆识别遮挡，判定成功

2. **容器冲突导致的假成功** (Case 2)
   - 场景：物体被放在关闭的容器前，视觉检测认为"靠近"即"内部"
   - REFLECT误判：判定成功放入
   - CRAFT改进：使用几何约束检查容器状态和体积，判定失败

3. **因果链错误** (Case 3)
   - 场景：跳过关键步骤（如未加水就加热水壶）
   - REFLECT误判：只检查空间关系，无法检测因果链
   - CRAFT改进：使用前置条件验证，判定失败

4. **瞬移检测** (Case 4)
   - 场景：物体位置突然跳变（>1.5m）
   - REFLECT误判：只检查最终状态，判定成功
   - CRAFT改进：使用不变式检测物理不可能的运动，判定失败

5. **靠近≠放入** (Case 5)
   - 场景：物体放在容器附近但未真正放入
   - REFLECT误判：将"靠近"误判为"内部"
   - CRAFT改进：使用几何体积检测，判定失败

6. **状态振荡** (Case 6)
   - 场景：容器状态在连续帧中振荡（开/关/开）
   - REFLECT误判：可能误判或不确定
   - CRAFT改进：使用状态平滑，返回不确定状态

## 使用方法

### 重要说明

**`test_failure_injection.py` 使用的是模拟数据**，它创建了模拟的SceneGraph来测试CRAFT和REFLECT的检测逻辑，**并没有实际运行AI2THOR**。

如果需要使用真实的AI2THOR数据，需要先运行数据生成脚本。

### 方法1：使用模拟数据测试（快速测试）

直接运行测试脚本（使用模拟的SceneGraph）：

```bash
python test_failure_injection.py
```

### 方法2：生成真实AI2THOR数据（推荐用于完整测试）

首先生成真实的AI2THOR失败注入数据：

```bash
# 生成所有失败场景的数据
python generate_failure_injection_data.py

# 或者只生成特定场景
python generate_failure_injection_data.py --cases case_1_occlusion case_2_container_conflict

# 查看所有可用场景
python generate_failure_injection_data.py --list
```

生成的数据会保存在 `thor_tasks/failure_injection_case_*/` 目录下，包含：
- `ego_img/`: 每帧的RGB图像
- `events/`: 每帧的AI2THOR事件数据（pickle格式）

然后可以修改 `test_failure_injection.py` 来加载这些真实数据。

### 输出说明

测试脚本会输出：
1. 每个测试案例的详细结果
2. CRAFT和REFLECT的检测结果对比
3. 准确率统计
4. 结果保存到 `output/failure_injection_results.json`

### 输出示例

```
================================================================================
Testing: Visual Occlusion False Failure (case_1)
Description: Apple picked up but occluded by arm → REFLECT thinks it failed
Ground Truth: success
================================================================================

[CRAFT Detection]
Result: success
Reason: Object occluded but likely held (memory-based reasoning)
Evaluation: ✓ Correct

[REFLECT Detection]
Result: failure
Reason: Object not visible in scene graph (REFLECT thinks it was dropped)
Evaluation: ✗ Expected success, got failure

...

================================================================================
SUMMARY
================================================================================

CRAFT Accuracy: 83.3% (5/6)
REFLECT Accuracy: 33.3% (2/6)
```

## 结果文件

测试结果保存在 `output/failure_injection_results.json`，包含：
- `craft_accuracy`: CRAFT准确率
- `reflect_accuracy`: REFLECT准确率
- `craft_results`: CRAFT详细结果
- `reflect_results`: REFLECT详细结果

## 集成到demo1

### 方法1：直接导入（推荐）

在demo1.ipynb中，确保已经设置了sys.path（通常在第一个cell中），然后：

```python
# 确保craft父目录在sys.path中（demo1的第一个cell通常已经做了）
# 如果还没有，添加：
import sys
from pathlib import Path
craft_dir = Path.cwd()  # 假设notebook在craft目录中
if str(craft_dir) not in sys.path:
    sys.path.insert(0, str(craft_dir))

# 然后导入并运行
from test_failure_injection import run_comparison_test
results = run_comparison_test()
```

### 方法2：使用exec执行

如果导入有问题，可以使用exec：

```python
# 在demo1.ipynb末尾添加新cell
import sys
from pathlib import Path

# 确保路径正确
craft_dir = Path.cwd()
if str(craft_dir) not in sys.path:
    sys.path.insert(0, str(craft_dir))

# 执行脚本
with open('test_failure_injection.py', 'r') as f:
    code = f.read()
    exec(code)

# 运行测试
results = run_comparison_test()
```

### 方法3：复制代码到notebook

如果上述方法都不行，可以直接将测试代码复制到notebook的cell中。

## 数据生成说明

### 当前状态

- **`test_failure_injection.py`**: 使用模拟的SceneGraph，**不生成AI2THOR数据**
- **`generate_failure_injection_data.py`**: 实际运行AI2THOR生成数据，**会生成真实数据到thor_tasks目录**

### 为什么thor_tasks里没有失败注入数据？

因为 `test_failure_injection.py` 只做模拟测试，不会生成数据。要生成真实数据，需要运行：

```bash
python generate_failure_injection_data.py
```

这会为6个失败场景生成真实的AI2THOR执行数据，保存在：
- `thor_tasks/failure_injection_case_1_occlusion/`
- `thor_tasks/failure_injection_case_2_container/`
- `thor_tasks/failure_injection_case_3_causal/`
- `thor_tasks/failure_injection_case_4_teleport/`
- `thor_tasks/failure_injection_case_5_near/`
- `thor_tasks/failure_injection_case_6_oscillation/`

## 注意事项

1. **模拟测试**：`test_failure_injection.py` 使用模拟的场景图，快速测试检测逻辑
2. **真实数据**：`generate_failure_injection_data.py` 生成真实的AI2THOR数据，需要AI2THOR环境
3. REFLECT检测器是简化实现，主要展示其基于LLM的检测方式
4. CRAFT检测器展示了约束验证、环境记忆等核心功能
5. 实际准确率可能因LLM响应而有所变化
6. 生成AI2THOR数据需要：
   - AI2THOR已安装
   - 网络连接（使用CloudRendering）
   - 可能需要较长时间（每个场景约1-5分钟）

## 扩展

要添加新的测试案例：

1. 在 `FailureInjector` 类中添加新的 `create_case_X()` 方法
2. 在 `create_all_cases()` 中添加新案例
3. 在 `CRAFTDetector` 和 `REFLECTDetector` 中添加相应的检测逻辑

