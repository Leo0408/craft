"""
修复后的 pick_up 实现（与 REFLECT 方法吻合）
可以直接替换 demo1.ipynb 中的 pick_up 部分
"""

# 替换 demo1.ipynb 中 elif action_name == "pick_up": 部分的代码

elif action_name == "pick_up":
    if "objectId" in action_params:
        obj_id = action_params.get("objectId")
        obj = None
        
        # 1. 查找对象
        for o in controller.last_event.metadata["objects"]:
            if o.get("objectId") == obj_id:
                obj = o
                break
        
        if obj is None:
            status = "FAILED"
            error = "Object not found"
            event = controller.last_event
        else:
            # 2. 获取对象类型，查找所有匹配的对象（REFLECT 方式）
            obj_type = obj.get("objectType")
            objs = [o for o in controller.last_event.metadata["objects"] 
                   if o.get("objectType") == obj_type]
            
            if len(objs) == 0:
                status = "FAILED"
                error = "No objects of type found"
                event = controller.last_event
            else:
                # 3. 检查第一个对象的可见性
                if not objs[0].get('visible', False):
                    print(f"  ⚠️  Object not visible, may need closer navigation")
                    # 可以在这里添加导航逻辑，或者继续尝试
                
                # 4. 遍历所有匹配的对象，尝试抓取（REFLECT 方式）
                success = False
                event = None
                
                for obj in objs:
                    obj_id = obj.get('objectId')
                    obj_pos = obj.get('position')
                    
                    # 5. 检查对象状态
                    if not obj.get('pickupable', False):
                        print(f"  ⚠️  Object {obj_id[:30]}... is not pickupable")
                        continue
                    
                    if obj.get('isPickedUp', False):
                        print(f"  ⚠️  Object {obj_id[:30]}... is already picked up")
                        continue
                    
                    # 6. Look at object（REFLECT 方式）
                    robot_pos = controller.last_event.metadata["agent"]["position"]
                    look_at_object(controller, obj_pos, robot_pos)
                    
                    # 7. 刷新对象状态（look_at 后对象状态可能改变）
                    for o in controller.last_event.metadata["objects"]:
                        if o.get("objectId") == obj_id:
                            obj = o
                            break
                    
                    # 8. 执行 PickupObject（REFLECT 参数）
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
                        print(f"  ✅ Successfully picked up {obj_id[:30]}...")
                        break
                    else:
                        error_msg = event.metadata.get("errorMessage", "Unknown error")
                        print(f"  ⚠️  Failed to pick up {obj_id[:30]}...: {error_msg}")
                
                if not success:
                    status = "FAILED"
                    if event:
                        error_msg = event.metadata.get("errorMessage", "Pickup failed")
                        error = f"Pickup failed: {error_msg}"
                    else:
                        error = "Pickup failed: No valid objects to pick up"
                    event = controller.last_event if event is None else event
                
                # 10. 调用 Done（REFLECT 方式）
                controller.step(action="Done")
    else:
        print(f"  Status: ❌ FAILED - Object ID not found")
        event = controller.last_event
        status = "FAILED"
        error = "Object ID not found"

