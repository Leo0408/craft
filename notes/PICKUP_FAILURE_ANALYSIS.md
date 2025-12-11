# Pick Up 失败原因分析

## 问题描述

在 Step 1 执行 `pick_up, Mug` 时出现失败：
```
Action 7/9: pick_up, Mug
Status: ❌ FAILED
Error: Pickup failed
```

## 原因分析

### 1. 当前实现方式（demo1.ipynb）

demo1.ipynb 中仍在使用**旧的内联实现方式**，而不是新的模块化方法。当前的 pick_up 实现存在以下问题：

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
```

### 2. 与 REFLECT 方法的差异

#### REFLECT 的 pick_up 实现（正确方式）

```python
def pick_up(taskUtil, obj_type, fail_execution=False, replan=False):
    # 1. 查找所有匹配的对象
    objs = [obj for obj in taskUtil.controller.last_event.metadata["objects"] 
            if obj["objectType"] == obj_type]
    
    # 2. 如果对象不可见，先导航
    if not objs[0]['visible'] and objs[0]['objectType'] not in taskUtil.objs_w_unk_loc:
        navigate_to_obj(taskUtil, objs[0]['objectType'], replan=replan)
    
    # 3. 遍历所有匹配的对象，尝试抓取
    for obj in objs:
        obj_id = obj['objectId']
        obj_pos = obj['position']
        
        # 4. 先 look_at 对象
        robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
        look_at(taskUtil, target_pos=obj_pos, robot_pos=robot_pos, replan=replan)
        
        # 5. 执行 PickupObject
        e = taskUtil.controller.step(
            action="PickupObject",
            objectId=obj_id,
            forceAction=False,
            manualInteract=False
        )
        
        # 6. 如果成功就退出循环
        if e.metadata['lastActionSuccess']:
            break
```

### 3. 失败的可能原因

#### 原因 1: 对象不可见
- **问题**: 对象可能在视野外或被遮挡
- **REFLECT 处理**: 先检查 `obj['visible']`，如果不可见则先导航
- **当前实现**: 没有检查可见性，直接尝试抓取

#### 原因 2: 机器人位置不合适
- **问题**: 机器人可能离对象太远或角度不对
- **REFLECT 处理**: 使用 `navigate_to_obj` 确保机器人靠近对象
- **当前实现**: 只使用 `look_at_object`，可能位置不够近

#### 原因 3: 对象状态问题
- **问题**: 对象可能已经被拿起、被其他物体遮挡、或状态不对
- **REFLECT 处理**: 遍历所有匹配的对象，尝试抓取每个
- **当前实现**: 只尝试一个对象 ID

#### 原因 4: look_at 后对象状态未刷新
- **问题**: look_at 后需要刷新对象状态
- **REFLECT 处理**: 在 look_at 后重新获取对象信息
- **当前实现**: 有刷新，但可能不够

#### 原因 5: 缺少 Done 动作
- **问题**: AI2THOR 需要在动作之间调用 `Done`
- **REFLECT 处理**: 每次动作后调用 `controller.step(action="Done")`
- **当前实现**: 可能缺少这个步骤

## 解决方案

### 方案 1: 使用新的模块化方法（推荐）

更新 demo1.ipynb 的 Step 1，使用新的模块化方法：

```python
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

# 使用模块化的动作原语
ap.pick_up(taskUtil, "Mug")
```

### 方案 2: 修复当前内联实现

如果继续使用内联实现，需要添加以下改进：

```python
elif action_name == "pick_up":
    if "objectId" in action_params:
        # 1. 查找所有匹配的对象（按 objectType，不只是 objectId）
        obj_type = "Mug"  # 从 action 中获取
        objs = [o for o in controller.last_event.metadata["objects"] 
                if o.get("objectType") == obj_type]
        
        if len(objs) == 0:
            status = "FAILED"
            error = "Object not found"
        else:
            # 2. 检查可见性，如果不可见先导航
            if not objs[0]['visible']:
                # 先导航到对象
                navigate_to_obj(...)
            
            # 3. 遍历所有匹配的对象
            success = False
            for obj in objs:
                obj_id = obj['objectId']
                obj_pos = obj['position']
                
                # 4. Look at object
                robot_pos = controller.last_event.metadata["agent"]["position"]
                look_at_object(controller, obj_pos, robot_pos)
                
                # 5. 刷新对象状态
                for o in controller.last_event.metadata["objects"]:
                    if o.get("objectId") == obj_id:
                        obj = o
                        break
                
                # 6. 执行 PickupObject
                event = controller.step(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False,
                    manualInteract=False
                )
                
                # 7. 检查成功
                if event.metadata.get("lastActionSuccess"):
                    status = "SUCCESS"
                    success = True
                    break
            
            if not success:
                status = "FAILED"
                error = "Pickup failed for all objects"
            
            # 8. 调用 Done
            controller.step(action="Done")
```

## 与 REFLECT 方法的吻合度

| 特性 | 当前实现 | REFLECT 方法 | 吻合度 |
|------|---------|-------------|--------|
| 对象查找 | 只按 objectId | 按 objectType，遍历所有匹配 | ❌ 不吻合 |
| 可见性检查 | ❌ 无 | ✅ 有 | ❌ 不吻合 |
| 导航处理 | ❌ 无 | ✅ 有 | ❌ 不吻合 |
| 对象遍历 | ❌ 只尝试一个 | ✅ 遍历所有匹配 | ❌ 不吻合 |
| look_at 调用 | ✅ 有 | ✅ 有 | ✅ 吻合 |
| Done 调用 | ⚠️ 可能缺少 | ✅ 有 | ⚠️ 部分吻合 |

## 建议

1. **立即修复**: 使用新的模块化方法（`action_primitives.pick_up`）
2. **长期改进**: 完全迁移到 REFLECT 风格的实现
3. **调试建议**: 
   - 检查对象的 `visible` 属性
   - 检查对象的 `pickupable` 属性
   - 检查对象是否已经被拿起（`isPickedUp`）
   - 检查机器人到对象的距离

## 调试代码

```python
# 在 pick_up 之前添加调试信息
obj = None
for o in controller.last_event.metadata["objects"]:
    if o.get("objectId") == action_params.get("objectId"):
        obj = o
        break

if obj:
    print(f"Object visible: {obj.get('visible')}")
    print(f"Object pickupable: {obj.get('pickupable')}")
    print(f"Object isPickedUp: {obj.get('isPickedUp')}")
    print(f"Object position: {obj.get('position')}")
    robot_pos = controller.last_event.metadata["agent"]["position"]
    distance = np.sqrt(
        (obj['position']['x'] - robot_pos['x'])**2 + 
        (obj['position']['z'] - robot_pos['z'])**2
    )
    print(f"Distance to object: {distance:.2f}")
```

