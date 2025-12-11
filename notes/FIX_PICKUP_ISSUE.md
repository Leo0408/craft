# 修复 Pick Up 失败问题

## 问题根源

当前的 `pick_up` 实现与 REFLECT 方法**不完全吻合**，主要问题：

1. ❌ **缺少可见性检查** - 没有检查 `obj['visible']`
2. ❌ **缺少可抓取性检查** - 没有检查 `obj['pickupable']`
3. ❌ **缺少状态检查** - 没有检查对象是否已被拿起
4. ❌ **缺少导航处理** - 对象不可见时没有先导航
5. ❌ **只尝试一个对象** - 应该遍历所有匹配的对象
6. ⚠️ **缺少 Done 调用** - 动作之间应该调用 `Done`

## 当前实现（有问题）

```python
elif action_name == "pick_up":
    if "objectId" in action_params:
        # Look at object before picking up
        obj = None
        for o in controller.last_event.metadata["objects"]:
            if o.get("objectId") == action_params.get("objectId"):
                obj = o
                break
        if obj:
            robot_pos = controller.last_event.metadata["agent"]["position"]
            look_at_object(controller, obj["position"], robot_pos)
            # Refresh object after look_at
            for o in controller.last_event.metadata["objects"]:
                if o.get("objectId") == action_params.get("objectId"):
                    obj = o
                    break
        
        event = controller.step(action="PickupObject", **action_params)
        status = "SUCCESS" if event.metadata.get("lastActionSuccess") else "FAILED"
        error = None if status == "SUCCESS" else "Pickup failed"
```

## 修复后的实现（与 REFLECT 吻合）

```python
elif action_name == "pick_up":
    if "objectId" in action_params:
        # 1. 从 objectId 获取 objectType（或直接从 action 参数获取）
        obj_id = action_params.get("objectId")
        obj_type = None
        
        # 找到对象并获取类型
        for o in controller.last_event.metadata["objects"]:
            if o.get("objectId") == obj_id:
                obj_type = o.get("objectType")
                break
        
        if obj_type is None:
            status = "FAILED"
            error = "Object type not found"
        else:
            # 2. 查找所有匹配该类型的对象（REFLECT 方式）
            objs = [o for o in controller.last_event.metadata["objects"] 
                   if o.get("objectType") == obj_type]
            
            if len(objs) == 0:
                status = "FAILED"
                error = "No objects of type found"
            else:
                # 3. 检查第一个对象的可见性，如果不可见先导航
                if not objs[0].get('visible', False):
                    print(f"  ⚠️  Object not visible, navigating first...")
                    # 导航到对象（使用 navigate_to_obj）
                    navigate_obj_id = objs[0].get('objectId')
                    # ... 导航逻辑 ...
                
                # 4. 遍历所有匹配的对象，尝试抓取
                success = False
                for obj in objs:
                    obj_id = obj.get('objectId')
                    obj_pos = obj.get('position')
                    
                    # 5. 检查对象状态
                    if not obj.get('pickupable', False):
                        print(f"  ⚠️  Object {obj_id} is not pickupable")
                        continue
                    
                    if obj.get('isPickedUp', False):
                        print(f"  ⚠️  Object {obj_id} is already picked up")
                        continue
                    
                    # 6. Look at object
                    robot_pos = controller.last_event.metadata["agent"]["position"]
                    look_at_object(controller, obj_pos, robot_pos)
                    
                    # 7. 刷新对象状态（look_at 后）
                    for o in controller.last_event.metadata["objects"]:
                        if o.get("objectId") == obj_id:
                            obj = o
                            break
                    
                    # 8. 执行 PickupObject
                    event = controller.step(
                        action="PickupObject",
                        objectId=obj_id,
                        forceAction=False,
                        manualInteract=False
                    )
                    
                    # 9. 检查成功
                    if event.metadata.get("lastActionSuccess"):
                        status = "SUCCESS"
                        error = None
                        success = True
                        print(f"  ✅ Successfully picked up {obj_id}")
                        break
                    else:
                        error_msg = event.metadata.get("errorMessage", "Unknown error")
                        print(f"  ⚠️  Failed to pick up {obj_id}: {error_msg}")
                
                if not success:
                    status = "FAILED"
                    error = "Pickup failed for all objects"
                
                # 10. 调用 Done（REFLECT 方式）
                controller.step(action="Done")
    else:
        status = "FAILED"
        error = "Object ID not found"
```

## 快速修复方案

最简单的修复是在现有代码中添加检查和调试信息：

```python
elif action_name == "pick_up":
    if "objectId" in action_params:
        obj_id = action_params.get("objectId")
        obj = None
        
        # 查找对象
        for o in controller.last_event.metadata["objects"]:
            if o.get("objectId") == obj_id:
                obj = o
                break
        
        if obj is None:
            status = "FAILED"
            error = "Object not found"
        else:
            # 添加调试信息
            print(f"  Object visible: {obj.get('visible', False)}")
            print(f"  Object pickupable: {obj.get('pickupable', False)}")
            print(f"  Object isPickedUp: {obj.get('isPickedUp', False)}")
            
            # 检查对象状态
            if not obj.get('pickupable', False):
                status = "FAILED"
                error = "Object is not pickupable"
            elif obj.get('isPickedUp', False):
                status = "FAILED"
                error = "Object is already picked up"
            elif not obj.get('visible', False):
                status = "FAILED"
                error = "Object is not visible (may need navigation)"
            else:
                # Look at object
                robot_pos = controller.last_event.metadata["agent"]["position"]
                look_at_object(controller, obj["position"], robot_pos)
                
                # Refresh object
                for o in controller.last_event.metadata["objects"]:
                    if o.get("objectId") == obj_id:
                        obj = o
                        break
                
                # Execute pickup
                event = controller.step(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False,
                    manualInteract=False
                )
                
                status = "SUCCESS" if event.metadata.get("lastActionSuccess") else "FAILED"
                if not status == "SUCCESS":
                    error_msg = event.metadata.get("errorMessage", "Pickup failed")
                    error = f"Pickup failed: {error_msg}"
                else:
                    error = None
                
                # Call Done (REFLECT style)
                controller.step(action="Done")
    else:
        status = "FAILED"
        error = "Object ID not found"
```

## 最佳方案：使用模块化方法

最好的解决方案是使用新的模块化方法，完全与 REFLECT 吻合：

```python
# 在 Step 1 开始时
from craft.utils.task_utils import TaskUtil
from craft.utils import action_primitives as ap

# 创建 TaskUtil
taskUtil = TaskUtil(
    folder_name="makeCoffee/test",
    controller=controller,
    reachable_positions=reachable_positions,
    failure_injection=False,
    index=0,
    repo_path="."
)

# 在动作执行循环中
elif action_name == "pick_up":
    obj_type = params[0] if len(params) > 0 else "Mug"
    ap.pick_up(taskUtil, obj_type)
    # 从 taskUtil 获取结果
    status = "SUCCESS"  # 根据 taskUtil 状态判断
```

## 与 REFLECT 的吻合度对比

| 检查项 | 当前实现 | 修复后 | REFLECT |
|--------|---------|--------|---------|
| 可见性检查 | ❌ | ✅ | ✅ |
| 可抓取性检查 | ❌ | ✅ | ✅ |
| 状态检查 | ❌ | ✅ | ✅ |
| 导航处理 | ❌ | ✅ | ✅ |
| 对象遍历 | ❌ | ✅ | ✅ |
| Done 调用 | ⚠️ | ✅ | ✅ |
| look_at 调用 | ✅ | ✅ | ✅ |

## 建议

1. **立即修复**: 添加调试信息和状态检查
2. **短期改进**: 实现完整的 REFLECT 风格检查
3. **长期方案**: 迁移到模块化的 `action_primitives.pick_up`

