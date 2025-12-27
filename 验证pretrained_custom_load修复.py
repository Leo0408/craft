# ============================================================================
# 验证 pretrained_custom_load 修复
# ============================================================================

import sys
import os

print("=" * 60)
print("验证 pretrained_custom_load 修复")
print("=" * 60)

timm_file = "/home/fdse/zzy/craft/Detic/detic/modeling/backbone/timm.py"

print("\n1. 检查 CustomResNet.__init__ 修复:")
with open(timm_file, 'r') as f:
    lines = f.readlines()
    in_class = False
    in_init = False
    found_default_cfg = False
    found_pretrained_custom_load = False
    
    for i, line in enumerate(lines):
        if 'class CustomResNet' in line:
            in_class = True
            print(f"   ✅ 找到 CustomResNet 类 (行 {i+1})")
        if in_class and 'def __init__' in line:
            in_init = True
            print(f"   ✅ 找到 __init__ 方法 (行 {i+1})")
        if in_init:
            if 'kwargs.pop' in line and 'default_cfg' in line:
                found_default_cfg = True
                print(f"   ✅ 找到 default_cfg 移除代码 (行 {i+1}):")
                print(f"      {line.strip()}")
            if 'kwargs.pop' in line and 'pretrained_custom_load' in line:
                found_pretrained_custom_load = True
                print(f"   ✅ 找到 pretrained_custom_load 移除代码 (行 {i+1}):")
                print(f"      {line.strip()}")
            if 'super().__init__' in line:
                if not found_pretrained_custom_load:
                    print(f"   ❌ 在 super().__init__ 之前未找到 pretrained_custom_load 移除")
                break

print("\n2. 检查 create_timm_resnet 函数:")
with open(timm_file, 'r') as f:
    content = f.read()
    if 'pretrained_custom_load=True' in content:
        print("   ✅ 传递 pretrained_custom_load 给 build_model_with_cfg")
    if 'default_cfg=cfg_to_use' in content:
        print("   ✅ 传递 default_cfg 给 build_model_with_cfg")

print("\n" + "=" * 60)
if found_default_cfg and found_pretrained_custom_load:
    print("✅ 修复验证完成 - 两个参数都已移除")
else:
    print("⚠️  修复可能不完整")
print("=" * 60)
print("\n💡 下一步:")
print("   1. 重启 Jupyter kernel (清除模块缓存)")
print("   2. 重新运行 Step 4 (初始化 DETIC + CLIP 检测器)")
print("   3. 应该不再出现 'unexpected keyword argument' 错误")

