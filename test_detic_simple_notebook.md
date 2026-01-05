# DETIC官方示例测试（Notebook版本）

## 问题诊断

您遇到的"CenterNet未注册"问题说明DETIC的安装可能不完整。让我们用官方最简洁的方式测试。

## 方法1: 直接使用官方demo.py（推荐）

在notebook中创建一个新的cell，运行：

```python
# 切换到Detic目录
import os
os.chdir('Detic')

# 运行官方demo（需要先下载测试图像）
# 从官方README: https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg
import urllib.request
urllib.request.urlretrieve(
    'https://web.eecs.umich.edu/~fouhey/fun/desk/desk.jpg', 
    'desk.jpg'
)

# 运行官方demo
!python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input desk.jpg \
    --output out.jpg \
    --vocabulary lvis \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth

# 显示结果
from PIL import Image
result = Image.open('out.jpg')
result
```

## 方法2: 使用自定义词汇表测试

```python
!python demo.py \
    --config-file configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml \
    --input desk.jpg \
    --output out2.jpg \
    --vocabulary custom \
    --custom_vocabulary coffee,machine,cup,table \
    --confidence-threshold 0.3 \
    --opts MODEL.WEIGHTS models/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth
```

## 如果仍然遇到"CenterNet未注册"错误

这说明DETIC的安装有问题。需要：

1. **重新安装DETIC**:
```bash
cd Detic
pip install -e .
```

2. **确保CenterNet2已安装**:
```bash
cd third_party/CenterNet2
pip install -e .
cd ../..
```

3. **检查导入**:
```python
import sys
sys.path.insert(0, 'Detic/third_party/CenterNet2/')
sys.path.insert(0, 'Detic')

# 应该能导入
from centernet.config import add_centernet_config
from detic.config import add_detic_config

# 检查注册
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
print("已注册的proposal generators:", list(PROPOSAL_GENERATOR_REGISTRY._obj_map.keys()))
```

## 对比我们的实现

如果官方demo能工作，但我们的代码不能，说明问题在：
1. 我们的注册逻辑（我们已经手动注册了FCOS为CenterNet）
2. 我们的配置方式
3. 我们的reset_cls_test调用时机

## 建议

1. **先测试官方demo**，确认DETIC本身能工作
2. **如果官方demo能工作**，对比我们的实现和官方demo的差异
3. **如果官方demo也不能工作**，说明DETIC安装有问题，需要重新安装
