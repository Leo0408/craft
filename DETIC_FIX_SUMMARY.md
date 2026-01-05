# DETIC检测器修复总结

## 问题描述

在使用`demo2.ipynb`中的DETIC+CLIP方法进行测试时，DETIC模型加载失败，错误信息：
```
FileNotFoundError: [Errno 2] No such file or directory: 'datasets/metadata/lvis_v1_train_cat_info.json'
```

## 问题原因

DETIC的配置文件中使用相对路径（如`'datasets/metadata/lvis_v1_train_cat_info.json'`），这些路径是相对于DETIC根目录的。当当前工作目录不是DETIC根目录时，DETIC在初始化模型时会找不到这些metadata文件。

## 修复方案

参考官方DETIC demo（`Detic/demo.py`和`Detic/detic/predictor.py`）的实现方式，对`perception/detic_clip_detector.py`进行了以下修复：

### 1. 临时改变工作目录

在创建`DefaultPredictor`之前，临时将工作目录切换到DETIC根目录，确保相对路径能正确解析：

```python
original_cwd = os.getcwd()
try:
    if detic_root and os.path.exists(detic_root):
        os.chdir(detic_root)
        print(f"📁 Temporarily changed working directory to DETIC root: {detic_root}")
    
    self.detic_model = DefaultPredictor(cfg)
finally:
    # Always restore original working directory
    os.chdir(original_cwd)
    if detic_root and os.path.exists(detic_root):
        print(f"📁 Restored working directory to: {original_cwd}")
```

### 2. 修复配置路径

在创建DefaultPredictor之前，将配置中的相对路径转换为绝对路径（如果可能）：

```python
# Fix CAT_FREQ_PATH (used by federated loss, if enabled)
if detic_root:
    if hasattr(cfg.MODEL.ROI_BOX_HEAD, 'CAT_FREQ_PATH'):
        cat_freq_path = cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
        if not os.path.isabs(cat_freq_path):
            cat_freq_path_abs = os.path.join(detic_root, cat_freq_path)
            if os.path.exists(cat_freq_path_abs):
                cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = cat_freq_path_abs
```

### 3. 增强错误处理

在`reset_cls_test`调用时添加了更详细的错误信息和路径检查。

## 使用方法

### 1. 确保metadata文件存在

metadata文件应该位于`Detic/datasets/metadata/`目录下：
- `lvis_v1_clip_a+cname.npy`
- `lvis_v1_train_cat_info.json`
- 其他metadata文件

可以使用以下命令检查：
```bash
cd /home/fdse/zzy/craft/Detic
ls -la datasets/metadata/
```

### 2. 重启Jupyter Kernel

由于修改了Python模块，建议：
1. 重启Jupyter kernel（Kernel → Restart Kernel）
2. 重新运行所有cells

### 3. 测试检测器

在`demo2.ipynb`中，确保：
- Cell 8中设置`DETECTION_METHOD = 'detic_clip'`
- Cell 9中初始化`scene_graph_builder`（这个很重要，否则Cell 16会报错）
- 按顺序执行所有cells

### 4. 验证DETIC加载

查看Cell 8的输出，应该看到：
```
✅ DETIC model loaded
✅ Set up default LVIS classifier (1203 classes)
```

而不是：
```
⚠️  Failed to load DETIC model: FileNotFoundError...
```

## 参考

修复参考了以下官方代码：
- `Detic/demo.py`: 官方demo脚本
- `Detic/detic/predictor.py`: 官方VisualizationDemo类
- `Detic/detic/modeling/utils.py`: reset_cls_test函数

官方demo的关键步骤：
1. 设置`cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'`
2. 创建`DefaultPredictor(cfg)`
3. 调用`reset_cls_test(model, classifier_path, num_classes)`设置classifier

我们的实现遵循了相同的模式，并添加了路径修复以确保在不同工作目录下都能正常工作。

## 其他注意事项

1. **如果metadata文件缺失**：可以尝试重新下载DETIC仓库，metadata文件应该已经包含在仓库中。

2. **如果仍然有问题**：检查DETIC根目录是否正确识别。代码会尝试以下路径：
   - `craft/Detic`
   - 当前目录下的`Detic`
   - `/home/fdse/zzy/craft/Detic`

3. **CLIP-only模式**：如果DETIC加载失败，检测器会自动切换到CLIP-only模式，仍然可以工作，但检测精度可能会降低。


