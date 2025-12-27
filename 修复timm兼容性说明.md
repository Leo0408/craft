# 修复 timm 兼容性问题

## 🔍 问题

新版本的 `timm` 库（>= 0.9）中，`DefaultCfg` 对象不支持直接赋值：
```python
default_cfgs_resnet['resnet50_in21k']['url'] = '...'  # ❌ 错误
```

## ✅ 解决方案

修改了 `/home/fdse/zzy/craft/Detic/detic/modeling/backbone/timm.py` 中的 `create_timm_resnet` 函数：

1. **不修改 `default_cfgs_resnet` 字典**：避免直接赋值到只读字典
2. **创建本地配置字典**：将 `DefaultCfg` 对象转换为普通字典
3. **直接使用本地配置**：将创建的配置直接传递给 `build_model_with_cfg`

## 🔧 关键修改

```python
# 旧代码（会出错）:
default_cfgs_resnet['resnet50_in21k'] = copy.deepcopy(default_cfgs_resnet['resnet50'])
default_cfgs_resnet['resnet50_in21k']['url'] = '...'  # ❌ 不支持赋值

# 新代码（修复后）:
base_cfg = default_cfgs_resnet.get('resnet50')
base_cfg_dict = dict(base_cfg.__dict__)  # 转换为字典
resnet50_in21k_cfg = copy.deepcopy(base_cfg_dict)
resnet50_in21k_cfg['url'] = '...'  # ✅ 可以修改字典
# 直接使用 resnet50_in21k_cfg，不存储回 default_cfgs_resnet
```

## 📋 下一步

1. **重启 kernel**（清除模块缓存）
2. **重新运行 Step 4**

## ⚠️ 如果仍然失败

如果 Jupyter 仍然使用缓存的旧代码，可以：

1. 重启 kernel
2. 或者手动清除缓存：
   ```python
   import sys
   if 'detic.modeling.backbone.timm' in sys.modules:
       del sys.modules['detic.modeling.backbone.timm']
   ```

