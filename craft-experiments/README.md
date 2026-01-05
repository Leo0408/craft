# CRAFT vs REFLECT 实验框架

本目录包含了 CRAFT vs REFLECT 失败检测实验的完整工程结构。

## 目录结构

```
craft-experiments/
├── environments/          # AI2-THOR 环境封装
├── tasks/                 # 任务定义和 Ground Truth
│   ├── task_defs.json    # 任务定义
│   └── ground_truth.py   # GT 函数（代码形式）
├── failure_injection/     # 失败注入模块
│   ├── failure_types.py  # 失败类型枚举
│   ├── injector.py       # 注入逻辑
│   └── injection_config.json  # 注入配置
├── perception/            # 感知模块（scene graph 等）
├── detectors/             # 失败检测器
│   ├── reflect_detector.py
│   ├── craft_detector.py
│   └── constraints/       # 约束定义
├── evaluation/            # 评估模块
├── runs/                  # 实验结果
│   ├── raw_logs/
│   ├── reflect/
│   ├── craft/
│   └── results.csv        # 结果汇总
└── scripts/               # 运行脚本
```

## Day One Checklist

使用 `day_one_checklist.py` 脚本可以快速搭建实验框架并运行基础测试：

```bash
# 在项目根目录运行
python3 day_one_checklist.py
```

该脚本会：
1. ✅ 创建完整的目录结构
2. ✅ 生成 3 个任务定义和 Ground Truth 函数
3. ✅ 实现 2 种 failure injection（MISSING_PRECONDITION, PHYSICAL_IMPOSSIBLE）
4. ✅ 运行一条「无失败」baseline
5. ✅ 运行一条「有失败」CRAFT 检测
6. ✅ 输出结果到 `runs/results.csv`

## 任务定义

当前包含 3 个任务：
- `make_coffee`: 制作咖啡
- `make_tea`: 泡茶
- `clean_mug`: 清洗杯子

任务定义在 `tasks/task_defs.json` 中，Ground Truth 函数在 `tasks/ground_truth.py` 中。

## Failure Injection

支持 2 种失败类型：
1. **MISSING_PRECONDITION**: 移除前置条件步骤
2. **PHYSICAL_IMPOSSIBLE**: 注入物理不可能的状态

配置在 `failure_injection/injection_config.json` 中。

## 结果格式

结果保存在 `runs/results.csv`，包含以下字段：
- `task`: 任务名称
- `failure_injected`: 是否注入了失败
- `failure_type`: 失败类型
- `detected`: 是否检测到失败
- `detector`: 检测器名称（BASELINE / CRAFT / REFLECT）
- `attribution_correct`: 归因是否正确
- `status`: 状态（SUCCESS / FAILURE_DETECTED）
- `timestamp`: 时间戳

## 下一步

1. 集成实际的 AI2THOR 执行逻辑
2. 实现完整的 REFLECT 和 CRAFT 检测器
3. 运行更多实验，扩展 results.csv
4. 添加评估指标计算（Acc / FPR / Attribution）

## 参考文档

- [Experiment.md](../Experiment.md): 完整实验设计文档
- [Method.md](../Method.md): CRAFT++ 框架方法论

