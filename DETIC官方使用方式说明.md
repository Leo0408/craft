# DETIC 官方使用方式说明

## 📋 官方使用方式（来自 demo.py）

DETIC 的官方使用方式非常简单：

```python
import sys

# 1. 添加 CenterNet2 路径（相对于 DETIC 根目录）
sys.path.insert(0, 'third_party/CenterNet2/')

# 2. 导入配置
from centernet.config import add_centernet_config
from detic.config import add_detic_config

# 3. 设置配置
from detectron2.config import get_cfg
cfg = get_cfg()
add_centernet_config(cfg)  # 先添加 CenterNet2 配置
add_detic_config(cfg)      # 再添加 DETIC 配置

# 4. 加载模型
cfg.merge_from_file("configs/xxx.yaml")
cfg.MODEL.WEIGHTS = "models/xxx.pth"
```

## 🔑 关键点

1. **路径设置**：`sys.path.insert(0, 'third_party/CenterNet2/')`
   - 这是相对于 DETIC 根目录的路径
   - 需要先切换到 DETIC 目录，或者使用绝对路径

2. **导入顺序**：
   - 先导入 `centernet.config`
   - 再导入 `detic.config`
   - 配置时先调用 `add_centernet_config(cfg)`
   - 再调用 `add_detic_config(cfg)`

3. **不需要手动创建 centernet 包装**：
   - CenterNet2 应该已经包含 `centernet` 目录
   - 如果使用 `git clone --recurse-submodules`，会自动包含

## ✅ 已修复的问题

1. **符号链接修复**：`centernet/modeling/backbone/bifpn.py` 现在正确指向 `adet/modeling/backbone/bifpn.py`
2. **导入顺序**：`detic_clip_detector.py` 现在按照官方方式导入
3. **配置顺序**：先添加 `add_centernet_config`，再添加 `add_detic_config`

## 🚀 下一步

在 Jupyter Notebook 中：

1. **重启 kernel**（清除已注册的模块）
2. **运行验证脚本**：
   ```python
   exec(open('简化验证脚本.py').read())
   ```
3. **重新运行 Step 4**（初始化 DETIC + CLIP 检测器）

应该看到：
```
📁 Added CenterNet2 path: /home/fdse/zzy/craft/Detic/third_party/CenterNet2
📁 Using local weights: /home/fdse/zzy/craft/Detic/models/...
📁 Using default config: /home/fdse/zzy/craft/Detic/configs/...
✅ DETIC model loaded
✅ CLIP model loaded (ViT-B/32)
✅ DETIC + CLIP detector initialized successfully!
```

