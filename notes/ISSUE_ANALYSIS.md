# 问题分析：pick_up 和 navigate_to_obj 失败

## 问题 1: pick_up 失败 - 对象已被拿起

### 现象
```
Action 7/9: pick_up, Mug
⚠️  Object Mug|-00.82|+00.84|+01.60... is already picked up
Status: ❌ FAILED
Error: Pickup failed: No valid objects to pick up
```

### 原因分析

**REFLECT 的处理方式**：
- REFLECT 的 `pick_up` **不检查 `isPickedUp`**
- 它直接尝试 `PickupObject`，让 AI2THOR 自己处理
- 如果对象已经被机器人拿起，应该检查**机器人是否已经拿着这个对象**

**当前实现的问题**：
- 我们检查了 `isPickedUp` 并跳过，但**没有检查机器人是否已经拿着这个对象**
- 如果机器人已经拿着对象，应该视为**成功**，而不是失败

### REFLECT 代码（action_primitives.py:151-178）
```python
for obj in objs:
    obj_id = obj['objectId']
    obj_pos = obj['position']
    # look at object
    robot_pos = taskUtil.controller.last_event.metadata['agent']['position']
    look_at(taskUtil, target_pos=obj_pos, robot_pos=robot_pos, replan=replan)
    e = taskUtil.controller.step(
        action="PickupObject",
        objectId=obj_id,
        forceAction=False,
        manualInteract=False
    )
    if e.metadata['lastActionSuccess']:
        break
```

**关键发现**：REFLECT **不检查 `isPickedUp`**，直接尝试抓取。如果对象已经被机器人拿起，`PickupObject` 会失败，但 REFLECT 会继续尝试其他对象。

### 修复方案

应该检查**机器人是否已经拿着这个对象**：

```python
# 检查机器人是否已经拿着这个对象
robot_holding_obj = False
for o in controller.last_event.metadata["objects"]:
    if o.get("objectId") == obj_id and o.get("isPickedUp", False):
        robot_holding_obj = True
        break

if robot_holding_obj:
    # 机器人已经拿着这个对象，视为成功
    status = "SUCCESS"
    error = None
    event = controller.last_event
    print(f"  ✅ Robot is already holding {obj_id[:30]}...")
    success = True
    break
elif obj.get('isPickedUp', False):
    # 对象被拿起但不是机器人拿的（可能是其他原因），跳过
    print(f"  ⚠️  Object {obj_id[:30]}... is already picked up (not by robot)")
    continue
```

---

## 问题 2: navigate_to_obj 失败 - Teleport 失败

### 现象
```
Action 8/9: navigate_to_obj, CoffeeMachine
⚠️  Teleport failed, using last event
Status: ❌ FAILED
Error: Teleport failed
```

### 原因分析

**REFLECT 的处理方式**：
- REFLECT 使用 **BFS 路径规划**（`findPath`）
- 使用 `forceAction=True` 和 `standing=True`
- 如果路径找不到，会打印错误但不会抛出异常

**当前实现的问题**：
- 使用简单的**最近距离**方法
- 使用 `forceAction=False`（默认）
- 可能位置不可达

### REFLECT 代码（action_primitives.py:66-72）
```python
e = taskUtil.controller.step(
    action="Teleport",
    position=dict(x=x, y=y, z=z),
    forceAction=True,  # 关键：使用 forceAction=True
    standing=True      # 关键：使用 standing=True
)
```

### 修复方案

1. **使用 forceAction=True**（REFLECT 方式）
2. **使用 standing=True**（REFLECT 方式）
3. **改进路径规划**（可选，使用 BFS）

```python
event = controller.step(
    action='Teleport', 
    x=nearest_pos['x'], 
    y=nearest_pos['y'], 
    z=nearest_pos['z'],
    forceAction=True,  # REFLECT 使用 True
    standing=True      # REFLECT 使用 True
)
```

---

## 总结

| 问题 | REFLECT 处理 | 当前实现 | 修复方案 |
|------|-------------|---------|---------|
| pick_up 已拿起 | 不检查，直接尝试 | 检查并跳过 | 检查机器人是否拿着，如果是则成功 |
| navigate Teleport | forceAction=True, standing=True | forceAction 默认 False | 使用 forceAction=True, standing=True |

