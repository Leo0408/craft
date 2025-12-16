# REFLECT gen_data.py 包装函数使用说明

## 概述

已创建了一个包装函数，使用 REFLECT 的稳定 `gen_data.py` 实现来生成 AI2THOR 失败注入测试数据。这可以避免 Controller 初始化卡住的问题。

## 实现方式

### 1. 主要函数

- **`generate_ai2thor_failure_case_data_with_reflect()`**: 
  - 直接调用 REFLECT 的 `run_data_gen()` 函数
  - 将我们的测试用例配置转换为 REFLECT 需要的格式
  - 处理 REFLECT 的 TASK_DICT 映射

- **`generate_ai2thor_failure_case_data()`**: 
  - 主函数，默认使用 REFLECT 方法（`use_reflect=True`）
  - 如果 REFLECT 不可用，自动回退到自定义实现
  - 统一返回格式

### 2. 配置转换

REFLECT 需要的 task 配置格式：
```python
reflect_task = {
    "task_idx": 11,  # 使用自定义索引（不冲突）
    "folder_name": "failure_case_xxx",
    "scene": "FloorPlan1",
    "actions": ["navigate_to_obj, Apple", ...],
    "failure_injection": True,
    "chosen_failure": "drop" | "failed_action" | "missing_step",
    "num_samples": 1,
    "failure_injection_params": {...}
}
```

### 3. 特殊失败类型处理

- **因果链错误 (causal_chain)**: 
  - 自动转换为 `missing_step` 失败类型
  - 自动找到需要跳过的步骤索引

## 使用方法

### 在 demo1.ipynb 中使用

```python
# 默认使用 REFLECT（推荐）
data = generate_ai2thor_failure_case_data(
    case_id="case_1_occlusion",
    task_config=case_config,
    save_data=True,
    use_reflect=True  # 默认值
)

# 如果 REFLECT 不可用，自动回退到自定义实现
data = generate_ai2thor_failure_case_data(
    case_id="case_1_occlusion",
    task_config=case_config,
    save_data=True,
    use_reflect=False  # 强制使用自定义实现
)
```

## REFLECT 路径配置

默认 REFLECT 路径：
- `gen_data.py`: `/home/fdse/zzy/reflect/main/gen_data.py`
- `constants.py`: `/home/fdse/zzy/reflect/main/constants.py`

如果路径不同，需要修改 `generate_ai2thor_failure_case_data_with_reflect()` 函数中的路径。

## 数据格式

两种方法返回相同的数据格式：
```python
{
    "events": [...],           # AI2THOR events
    "action_results": [...],   # 动作执行结果
    "initial_sg": SceneGraph, # 初始场景图
    "final_sg": SceneGraph,    # 最终场景图
    "data_path": "thor_tasks/failure_injection/...",
    "case_id": "case_1_occlusion",
    "method": "reflect" | "custom"  # 标记使用的方法
}
```

## 优势

1. **稳定性**: 使用 REFLECT 的经过验证的实现
2. **避免卡住**: REFLECT 的实现已经处理了各种边界情况
3. **自动回退**: 如果 REFLECT 不可用，自动使用自定义实现
4. **统一接口**: 两种方法返回相同格式的数据

## 故障排除

### REFLECT 模块导入失败

如果遇到 `ImportError`:
1. 检查 REFLECT 路径是否正确
2. 确保 REFLECT 已正确安装
3. 检查 Python 路径设置

### TASK_DICT 冲突

如果遇到 TASK_DICT 相关错误:
- 函数会自动处理，临时添加 `task_idx=11` 到 TASK_DICT
- 执行完成后会自动清理

## 注意事项

1. **首次运行**: REFLECT 方法仍然需要下载 AI2THOR 资源（如果未下载）
2. **数据位置**: REFLECT 生成的数据保存在 `thor_tasks/failure_injection/{folder_name}/`
3. **兼容性**: 确保 REFLECT 和 CRAFT 使用相同版本的 AI2THOR

