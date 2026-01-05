# 正确的Detector初始化方式

## 问题

`DeticClipDetector.__init__()`的参数名可能与预期不同。

## 解决方案

检查`__init__`方法的实际签名，然后使用正确的参数名。

在notebook中运行：

```python
# 检查__init__方法的签名
import inspect
from craft.perception.detic_clip_detector import DeticClipDetector

sig = inspect.signature(DeticClipDetector.__init__)
print("DeticClipDetector.__init__ 参数:")
for param_name, param in sig.parameters.items():
    if param_name != 'self':
        print(f"  {param_name}: {param}")
```

然后根据实际的参数名初始化detector。
