"""
重新加载DeticClipDetector模块以使用最新的代码（包括新的参数）
在Jupyter notebook中运行这个代码块
"""
import importlib
import sys

# 清除相关的模块缓存
modules_to_reload = [
    'perception.detic_clip_detector',
    'craft.perception.detic_clip_detector',
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        print(f"🔄 清除模块缓存: {module_name}")
        del sys.modules[module_name]

# 如果detector已经初始化，需要重新初始化
# 因为旧的detector实例使用的是旧版本的类

print("✅ 模块缓存已清除")
print("💡 提示: 现在需要重新初始化detector才能使用新的参数")
print("   例如: detector = DeticClipDetector(detic_threshold=0.3, clip_threshold=0.25)")
