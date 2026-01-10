# Demo4 - DETIC官方Demo测试

## 简介

`demo4.ipynb` 是基于DETIC官方demo代码的简化版本，用于测试DETIC检测器是否能正常工作。它使用demo2的第一帧图像作为测试图像。

## 使用方法

1. **打开notebook**
   ```bash
   jupyter notebook demo4.ipynb
   ```

2. **按顺序执行所有cells**

3. **检查输出**
   - Cell 1: 路径设置
   - Cell 2: 从demo2加载第一帧并保存为图片
   - Cell 3: 初始化DETIC检测器
   - Cell 4: 运行检测并可视化结果

## 关键特性

- 使用官方DETIC代码 (`Detic/detic/predictor.py` 的 `VisualizationDemo`)
- 在DETIC目录下工作，确保相对路径正确
- 使用LVIS词汇表进行检测
- 可视化检测结果并保存

## 输出文件

- `test_frame_from_demo2.jpg`: 从demo2保存的第一帧图像
- `demo4_output.jpg`: DETIC检测结果可视化

## 与demo2的区别

- demo4只测试DETIC检测功能，不涉及场景图构建
- 使用官方代码，避免自定义wrapper的问题
- 更简单直接，便于调试

## 如果遇到问题

1. **确保DETIC目录存在**：`craft/Detic/`
2. **确保metadata文件存在**：`Detic/datasets/metadata/`
3. **确保模型权重存在**：`Detic/models/`
4. **重启kernel**：如果导入模块有问题，重启Jupyter kernel




