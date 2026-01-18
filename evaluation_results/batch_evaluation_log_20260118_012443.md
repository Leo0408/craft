# CRAFT 批量评估详细日志
生成时间: 2026-01-18 01:24:43
配置: LLM分析=False, GPT模型=gpt-3.5-turbo, 实例过滤=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
数据集数量: 10

================================================================================

## [1/10] boilWater/boilWater-1

### 数据加载信息

✅ 加载了 49 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-13
   Action 3: (put_in, Pot, Sink) -> Frames 14-18
   Action 4: (toggle_on, Faucet) -> Frames 19-23
   Action 5: (toggle_off, Faucet) -> Frames 24-28
   Action 6: (pick_up, Pot) -> Frames 29-33
   Action 7: (navigate_to_obj, StoveBurner-4) -> Frames 34-38
   Action 8: (put_on, Pot, StoveBurner-4) -> Frames 39-43
   Action 9: (toggle_on, StoveBurner-4) -> Frames 44-48

### CRAFT 流程详细信息

   ✅ 加载了 49 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 49 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Pot)
         检查动作 4/10: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/10: (pick_up, Pot)
         检查动作 9/10: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pot must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:36
- **Failure Reason**: Dropped Pot

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [2/10] boilWater/boilWater-10

### 数据加载信息

✅ 加载了 53 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(toggle_on, Faucet)', '(toggle_off, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-4
   Action 1: (pick_up, Pot) -> Frames 5-9
   Action 2: (navigate_to_obj, Sink) -> Frames 10-14
   Action 3: (toggle_on, Faucet) -> Frames 15-20
   Action 4: (toggle_off, Faucet) -> Frames 21-25
   Action 5: (put_in, Pot, Sink) -> Frames 26-30
   Action 6: (pick_up, Pot) -> Frames 31-36
   Action 7: (navigate_to_obj, StoveBurner-4) -> Frames 37-41
   Action 8: (put_on, Pot, StoveBurner-4) -> Frames 42-46
   Action 9: (toggle_on, StoveBurner-4) -> Frames 47-52

### CRAFT 流程详细信息

   ✅ 加载了 53 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 53 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(toggle_on, Faucet)', '(toggle_off, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Pot)
         检查动作 4/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 4-8 内未满足): Faucet must be toggled on
         检查动作 5/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 5 (窗口 5-9) 满足): Faucet must be toggled off
         检查动作 6/10: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Pot must be inside Sink
         检查动作 7/10: (pick_up, Pot)
         检查动作 9/10: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pot must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:25', '00:26', '00:27', '00:28', '00:29', '00:30', '00:31']
- **Failure Reason**: The robot put the pot in sink after the faucet was turned off, as a result, the pot was not filled with water.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [3/10] boilWater/boilWater-2

### 数据加载信息

✅ 加载了 56 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene', 'object_list', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-4
   Action 1: (pick_up, Pot) -> Frames 5-10
   Action 2: (navigate_to_obj, Sink) -> Frames 11-15
   Action 3: (put_in, Pot, Sink) -> Frames 16-21
   Action 4: (toggle_on, Faucet) -> Frames 22-27
   Action 5: (toggle_off, Faucet) -> Frames 28-32
   Action 6: (pick_up, Pot) -> Frames 33-38
   Action 7: (navigate_to_obj, StoveBurner-4) -> Frames 39-43
   Action 8: (put_on, Pot, StoveBurner-4) -> Frames 44-49
   Action 9: (toggle_on, StoveBurner-4) -> Frames 50-55

### CRAFT 流程详细信息

   ✅ 加载了 56 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 56 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Pot)
         检查动作 4/10: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/10: (pick_up, Pot)
         检查动作 9/10: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pot must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: blocking
- **Failure Step**: 00:08
- **Failure Reason**: The robot cannot pick up the pot due to a bread occluding the pot.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [4/10] boilWater/boilWater-3

### 数据加载信息

✅ 加载了 53 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(toggle_on, Faucet)', '(put_in, Pot, Sink)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-4
   Action 1: (pick_up, Pot) -> Frames 5-9
   Action 2: (navigate_to_obj, Sink) -> Frames 10-14
   Action 3: (toggle_on, Faucet) -> Frames 15-20
   Action 4: (put_in, Pot, Sink) -> Frames 21-25
   Action 5: (toggle_off, Faucet) -> Frames 26-30
   Action 6: (pick_up, Pot) -> Frames 31-36
   Action 7: (navigate_to_obj, StoveBurner-4) -> Frames 37-41
   Action 8: (put_on, Pot, StoveBurner-4) -> Frames 42-46
   Action 9: (toggle_on, StoveBurner-4) -> Frames 47-52

### CRAFT 流程详细信息

   ✅ 加载了 53 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 53 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(toggle_on, Faucet)', '(put_in, Pot, Sink)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Pot)
         检查动作 4/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 4-8 内未满足): Faucet must be toggled on
         检查动作 5/10: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Pot must be inside Sink
         检查动作 6/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/10: (pick_up, Pot)
         检查动作 9/10: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pot must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: occupied_put
- **Failure Step**: 00:18
- **Failure Reason**: An apple is inside the pot at the beginning of the task execution, and the robot never removed it from the pot.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [5/10] boilWater/boilWater-4

### 数据加载信息

✅ 加载了 51 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-4
   Action 1: (pick_up, Pot) -> Frames 5-9
   Action 2: (navigate_to_obj, Sink) -> Frames 10-14
   Action 3: (put_in, Pot, Sink) -> Frames 15-19
   Action 4: (toggle_on, Faucet) -> Frames 20-24
   Action 5: (toggle_off, Faucet) -> Frames 25-29
   Action 6: (pick_up, Pot) -> Frames 30-34
   Action 7: (navigate_to_obj, StoveBurner-4) -> Frames 35-39
   Action 8: (put_on, Pot, StoveBurner-4) -> Frames 40-44
   Action 9: (toggle_on, StoveBurner-4) -> Frames 45-50

### CRAFT 流程详细信息

   ✅ 加载了 51 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 51 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Pot)
         检查动作 4/10: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/10: (pick_up, Pot)
         检查动作 9/10: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pot must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:26
- **Failure Reason**: The robot failed to toggle on faucet.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [6/10] boilWater/boilWater-5

### 数据加载信息

✅ 加载了 67 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-5
   Action 1: (pick_up, Pot) -> Frames 6-12
   Action 2: (navigate_to_obj, Sink) -> Frames 13-19
   Action 3: (put_in, Pot, Sink) -> Frames 20-25
   Action 4: (toggle_on, Faucet) -> Frames 26-32
   Action 5: (toggle_off, Faucet) -> Frames 33-39
   Action 6: (pick_up, Pot) -> Frames 40-45
   Action 7: (navigate_to_obj, StoveBurner-4) -> Frames 46-52
   Action 8: (put_on, Pot, StoveBurner-4) -> Frames 53-59
   Action 9: (toggle_on, StoveBurner-4) -> Frames 60-66

### CRAFT 流程详细信息

   ✅ 加载了 67 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 67 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Pot)
         检查动作 4/10: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/10: (pick_up, Pot)
         检查动作 9/10: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pot must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong perception
- **Failure Step**: 00:24
- **Failure Reason**: The robot mis-identified pan as pot, and picked up the pan instead of the pot.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [7/10] boilWater/boilWater-6

### 数据加载信息

✅ 加载了 47 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(pick_up, Pot)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-4
   Action 1: (pick_up, Pot) -> Frames 5-10
   Action 2: (navigate_to_obj, Sink) -> Frames 11-16
   Action 3: (put_in, Pot, Sink) -> Frames 17-22
   Action 4: (pick_up, Pot) -> Frames 23-28
   Action 5: (navigate_to_obj, StoveBurner-4) -> Frames 29-34
   Action 6: (put_on, Pot, StoveBurner-4) -> Frames 35-40
   Action 7: (toggle_on, StoveBurner-4) -> Frames 41-46

### CRAFT 流程详细信息

   ✅ 加载了 47 个 events
   ✅ 建立了动作-帧映射: 8 个动作
       ✅ 加载了 47 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(pick_up, Pot)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=8, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/8: (pick_up, Pot)
         检查动作 4/8: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/8: (pick_up, Pot)
         检查动作 7/8: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 7-14 内未满足): Pot must be on top of StoveBurner-4
         检查动作 8/8: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:25', '00:26', '00:27', '00:28']
- **Failure Reason**: The robot never executed the actions to toggle on and off the faucet, as a result, the pot was never filled with water.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [8-12]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [8/10] boilWater/boilWater-7

### 数据加载信息

✅ 加载了 46 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-4
   Action 1: (pick_up, Pot) -> Frames 5-9
   Action 2: (navigate_to_obj, Sink) -> Frames 10-14
   Action 3: (put_in, Pot, Sink) -> Frames 15-19
   Action 4: (toggle_on, Faucet) -> Frames 20-24
   Action 5: (toggle_off, Faucet) -> Frames 25-29
   Action 6: (navigate_to_obj, StoveBurner-4) -> Frames 30-34
   Action 7: (put_on, Pot, StoveBurner-4) -> Frames 35-39
   Action 8: (toggle_on, StoveBurner-4) -> Frames 40-45

### CRAFT 流程详细信息

   ✅ 加载了 46 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 46 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 15 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 8/9: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Pot must be on top of StoveBurner-4
         检查动作 9/9: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 9 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:31', '00:32', '00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39', '00:40', '00:41']
- **Failure Reason**: The robot forgot to pick up the pot from sink, as a result, nothing was placed on the fourth stove burner.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [9-13]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [9/10] boilWater/boilWater-8

### 数据加载信息

✅ 加载了 59 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Bowl)', '(pick_up, Bowl)', '(navigate_to_obj, Sink)', '(put_in, Bowl, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Bowl) -> Frames 0-4
   Action 1: (pick_up, Bowl) -> Frames 5-10
   Action 2: (navigate_to_obj, Sink) -> Frames 11-16
   Action 3: (put_in, Bowl, Sink) -> Frames 17-22
   Action 4: (toggle_on, Faucet) -> Frames 23-28
   Action 5: (toggle_off, Faucet) -> Frames 29-34
   Action 6: (pick_up, Bowl) -> Frames 35-40
   Action 7: (navigate_to_obj, StoveBurner-4) -> Frames 41-46
   Action 8: (put_on, Bowl, StoveBurner-4) -> Frames 47-52
   Action 9: (toggle_on, StoveBurner-4) -> Frames 53-58

### CRAFT 流程详细信息

   ✅ 加载了 59 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 59 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Bowl)', '(pick_up, Bowl)', '(navigate_to_obj, Sink)', '(put_in, Bowl, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Bowl)', '(pick_up, Bowl)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Bowl)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Bowl)
         检查动作 4/10: (put_in, Bowl, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Bowl must be inside Sink
         检查动作 5/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/10: (pick_up, Bowl)
         检查动作 9/10: (put_on, Bowl, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Bowl must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-4 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:20
- **Failure Reason**: The robot should use a pot instead of a bowl to boil water. The bowl cannot be put on the stove burner.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-4 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 4 is not toggled on
      Frame: Unknown frame


================================================================================

## [10/10] boilWater/boilWater-9

### 数据加载信息

✅ 加载了 51 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'chosen_failure', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-4
   Action 1: (pick_up, Pot) -> Frames 5-9
   Action 2: (navigate_to_obj, Sink) -> Frames 10-14
   Action 3: (put_in, Pot, Sink) -> Frames 15-19
   Action 4: (toggle_on, Faucet) -> Frames 20-24
   Action 5: (toggle_off, Faucet) -> Frames 25-29
   Action 6: (pick_up, Pot) -> Frames 30-34
   Action 7: (navigate_to_obj, StoveBurner) -> Frames 35-39
   Action 8: (put_on, Pot, StoveBurner-4) -> Frames 40-44
   Action 9: (toggle_on, StoveBurner-2) -> Frames 45-50

### CRAFT 流程详细信息

   ✅ 加载了 51 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 51 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'chosen_failure', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 18 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/10: (pick_up, Pot)
         检查动作 4/10: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/10: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/10: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/10: (pick_up, Pot)
         检查动作 9/10: (put_on, Pot, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pot must be on top of StoveBurner-4
         检查动作 10/10: (toggle_on, StoveBurner-2)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): StoveBurner-2 must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: ambiguous_plan
- **Failure Step**: ['00:46', '00:47', '00:48', '00:49']
- **Failure Reason**: The robot put the pot on the fourth stove burner but toggled on the second stove burner (instead of the fourth stove burner).

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: StoveBurner-2 must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: 2 is not toggled on
      Frame: Unknown frame


================================================================================
