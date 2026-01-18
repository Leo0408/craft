# CRAFT 批量评估详细日志
生成时间: 2026-01-18 03:27:27
配置: LLM分析=False, GPT模型=gpt-3.5-turbo, 实例过滤=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
数据集数量: 100

================================================================================

## [1/100] boilWater/boilWater-1

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

## [2/100] boilWater/boilWater-10

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

## [3/100] boilWater/boilWater-2

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

## [4/100] boilWater/boilWater-3

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

## [5/100] boilWater/boilWater-4

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

## [6/100] boilWater/boilWater-5

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

## [7/100] boilWater/boilWater-6

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

## [8/100] boilWater/boilWater-7

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

## [9/100] boilWater/boilWater-8

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

## [10/100] boilWater/boilWater-9

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

## [11/100] cookEgg/cookEgg-1

### 数据加载信息

✅ 加载了 69 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-3
   Action 1: (open_obj, Fridge) -> Frames 4-7
   Action 2: (pick_up, Egg) -> Frames 8-11
   Action 3: (close_obj, Fridge) -> Frames 12-16
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 17-20
   Action 5: (toggle_on, StoveBurner-1) -> Frames 21-24
   Action 6: (navigate_to_obj, Pan) -> Frames 25-29
   Action 7: (put_on, Egg, CounterTop) -> Frames 30-33
   Action 8: (pick_up, Pan) -> Frames 34-37
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 38-42
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 43-46
   Action 11: (navigate_to_obj, Egg) -> Frames 47-50
   Action 12: (pick_up, Egg) -> Frames 51-55
   Action 13: (navigate_to_obj, Pan) -> Frames 56-59
   Action 14: (crack_obj, Egg) -> Frames 60-63
   Action 15: (put_in, EggCracked, Pan) -> Frames 64-68

### CRAFT 流程详细信息

   ✅ 加载了 69 个 events
   ✅ 建立了动作-帧映射: 16 个动作
       ✅ 加载了 69 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=16, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/16: (pick_up, Egg)
         检查动作 6/16: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/16: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/16: (pick_up, Pan)
         检查动作 11/16: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/16: (pick_up, Egg)
         检查动作 16/16: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 16-23 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:35
- **Failure Reason**: Dropped Egg

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [16-23]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [12/100] cookEgg/cookEgg-10

### 数据加载信息

✅ 加载了 87 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene', 'object_list', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-4
   Action 1: (open_obj, Fridge) -> Frames 5-9
   Action 2: (pick_up, Egg) -> Frames 10-15
   Action 3: (close_obj, Fridge) -> Frames 16-20
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 21-26
   Action 5: (toggle_on, StoveBurner-4) -> Frames 27-31
   Action 6: (navigate_to_obj, Pan) -> Frames 32-37
   Action 7: (put_on, Egg, CounterTop) -> Frames 38-42
   Action 8: (pick_up, Pan) -> Frames 43-47
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 48-53
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 54-58
   Action 11: (navigate_to_obj, Egg) -> Frames 59-64
   Action 12: (pick_up, Egg) -> Frames 65-69
   Action 13: (navigate_to_obj, Pan) -> Frames 70-75
   Action 14: (crack_obj, Egg) -> Frames 76-80
   Action 15: (put_in, EggCracked, Pan) -> Frames 81-86

### CRAFT 流程详细信息

   ✅ 加载了 87 个 events
   ✅ 建立了动作-帧映射: 16 个动作
       ✅ 加载了 87 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene']
       🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=16, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/16: (pick_up, Egg)
         检查动作 6/16: (toggle_on, StoveBurner-4)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-4 must be toggled on
         检查动作 8/16: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/16: (pick_up, Pan)
         检查动作 11/16: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/16: (pick_up, Egg)
         检查动作 16/16: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 16-23 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: ambiguous_plan
- **Failure Step**: ['00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39', '00:40', '00:41', '00:42', '00:43', '00:44', '00:45', '00:46', '00:47', '00:48', '00:49', '00:50', '00:51', '00:52', '00:53', '00:54', '00:55', '00:56', '00:57', '00:58', '00:59', '01:00']
- **Failure Reason**: The robot toggled on the fourth stove burner but put the pan on the first stove burner instead.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [16-23]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [13/100] cookEgg/cookEgg-2

### 数据加载信息

✅ 加载了 79 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-3
   Action 1: (open_obj, Fridge) -> Frames 4-8
   Action 2: (pick_up, Egg) -> Frames 9-13
   Action 3: (close_obj, Fridge) -> Frames 14-18
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 19-23
   Action 5: (toggle_on, StoveBurner-1) -> Frames 24-28
   Action 6: (navigate_to_obj, Pan) -> Frames 29-33
   Action 7: (put_on, Egg, CounterTop) -> Frames 34-38
   Action 8: (pick_up, Pan) -> Frames 39-43
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 44-48
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 49-53
   Action 11: (navigate_to_obj, Egg) -> Frames 54-58
   Action 12: (pick_up, Egg) -> Frames 59-63
   Action 13: (navigate_to_obj, Pan) -> Frames 64-68
   Action 14: (crack_obj, Egg) -> Frames 69-73
   Action 15: (put_in, EggCracked, Pan) -> Frames 74-78

### CRAFT 流程详细信息

   ✅ 加载了 79 个 events
   ✅ 建立了动作-帧映射: 16 个动作
       ✅ 加载了 79 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=16, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/16: (pick_up, Egg)
         检查动作 6/16: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/16: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/16: (pick_up, Pan)
         检查动作 11/16: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/16: (pick_up, Egg)
         检查动作 16/16: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 16-23 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: occupied
- **Failure Step**: 00:58
- **Failure Reason**: The robot failed to put the pan on the first stove burner because there was already a pot on it.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [16-23]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [14/100] cookEgg/cookEgg-3

### 数据加载信息

✅ 加载了 87 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene', 'object_list', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-4
   Action 1: (open_obj, Fridge) -> Frames 5-9
   Action 2: (pick_up, Egg) -> Frames 10-15
   Action 3: (close_obj, Fridge) -> Frames 16-20
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 21-26
   Action 5: (toggle_on, StoveBurner-1) -> Frames 27-31
   Action 6: (navigate_to_obj, Pan) -> Frames 32-37
   Action 7: (put_on, Egg, CounterTop) -> Frames 38-42
   Action 8: (pick_up, Pan) -> Frames 43-47
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 48-53
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 54-58
   Action 11: (navigate_to_obj, Egg) -> Frames 59-64
   Action 12: (pick_up, Egg) -> Frames 65-69
   Action 13: (navigate_to_obj, Pan) -> Frames 70-75
   Action 14: (crack_obj, Egg) -> Frames 76-80
   Action 15: (put_in, EggCracked, Pan) -> Frames 81-86

### CRAFT 流程详细信息

   ✅ 加载了 87 个 events
   ✅ 建立了动作-帧映射: 16 个动作
       ✅ 加载了 87 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene']
       🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=16, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/16: (pick_up, Egg)
         检查动作 6/16: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/16: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/16: (pick_up, Pan)
         检查动作 11/16: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/16: (pick_up, Egg)
         检查动作 16/16: (put_in, EggCracked, Pan)
           ❌ Precondition 违反: Pan must be empty
           ❌ Postcondition 违反 (窗口 16-23 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 5 个违反, 5 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: occupied
- **Failure Step**: ['00:47', '00:48', '00:49', '00:50', '00:51', '00:52', '00:53', '00:54', '00:55', '00:56', '00:57', '00:58', '00:59', '01:00']
- **Failure Reason**: A potato is inside the pan at the beginning of the task execution, and the robot never removed it from the pan.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Pan must be empty
      Reason: Container 'pan' contains 1 object(s): Potato_4dee147d
      Frame: Unknown frame


================================================================================

## [15/100] cookEgg/cookEgg-4

### 数据加载信息

✅ 加载了 119 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-6
   Action 1: (open_obj, Fridge) -> Frames 7-13
   Action 2: (pick_up, Egg) -> Frames 14-21
   Action 3: (close_obj, Fridge) -> Frames 22-28
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 29-36
   Action 5: (toggle_on, StoveBurner-1) -> Frames 37-43
   Action 6: (navigate_to_obj, Pan) -> Frames 44-51
   Action 7: (put_on, Egg, CounterTop) -> Frames 52-58
   Action 8: (pick_up, Pan) -> Frames 59-65
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 66-73
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 74-80
   Action 11: (navigate_to_obj, Egg) -> Frames 81-88
   Action 12: (pick_up, Egg) -> Frames 89-95
   Action 13: (navigate_to_obj, Pan) -> Frames 96-103
   Action 14: (crack_obj, Egg) -> Frames 104-110
   Action 15: (put_in, EggCracked, Pan) -> Frames 111-118

### CRAFT 流程详细信息

   ✅ 加载了 119 个 events
   ✅ 建立了动作-帧映射: 16 个动作
       ✅ 加载了 119 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=16, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/16: (pick_up, Egg)
         检查动作 6/16: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/16: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/16: (pick_up, Pan)
         检查动作 11/16: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/16: (pick_up, Egg)
         检查动作 16/16: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 16-23 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:05
- **Failure Reason**: The robot failed to open the fridge.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [16-23]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [16/100] cookEgg/cookEgg-5

### 数据加载信息

✅ 加载了 120 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-6
   Action 1: (open_obj, Fridge) -> Frames 7-14
   Action 2: (pick_up, Egg) -> Frames 15-21
   Action 3: (close_obj, Fridge) -> Frames 22-29
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 30-36
   Action 5: (toggle_on, StoveBurner-1) -> Frames 37-44
   Action 6: (navigate_to_obj, Pan) -> Frames 45-51
   Action 7: (put_on, Egg, CounterTop) -> Frames 52-59
   Action 8: (pick_up, Pan) -> Frames 60-66
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 67-74
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 75-81
   Action 11: (navigate_to_obj, Egg) -> Frames 82-89
   Action 12: (pick_up, Egg) -> Frames 90-96
   Action 13: (navigate_to_obj, Pan) -> Frames 97-104
   Action 14: (crack_obj, Egg) -> Frames 105-111
   Action 15: (put_in, EggCracked, Pan) -> Frames 112-119

### CRAFT 流程详细信息

   ✅ 加载了 120 个 events
   ✅ 建立了动作-帧映射: 16 个动作
       ✅ 加载了 120 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=16, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/16: (pick_up, Egg)
         检查动作 6/16: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/16: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/16: (pick_up, Pan)
         检查动作 11/16: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/16: (pick_up, Egg)
         检查动作 16/16: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 16-23 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong perception
- **Failure Step**: 01:01
- **Failure Reason**: The robot mis-identified book as pan, and picked up the book instead of the pan. The book cannot be put on the stove burner.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [16-23]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [17/100] cookEgg/cookEgg-6

### 数据加载信息

✅ 加载了 82 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=15, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-4
   Action 1: (open_obj, Fridge) -> Frames 5-9
   Action 2: (pick_up, Egg) -> Frames 10-15
   Action 3: (close_obj, Fridge) -> Frames 16-20
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 21-26
   Action 5: (toggle_on, StoveBurner-1) -> Frames 27-31
   Action 6: (navigate_to_obj, Pan) -> Frames 32-37
   Action 7: (put_on, Egg, CounterTop) -> Frames 38-42
   Action 8: (pick_up, Pan) -> Frames 43-48
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 49-53
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 54-59
   Action 11: (navigate_to_obj, Egg) -> Frames 60-64
   Action 12: (pick_up, Egg) -> Frames 65-70
   Action 13: (navigate_to_obj, Pan) -> Frames 71-75
   Action 14: (put_in, EggCracked, Pan) -> Frames 76-81

### CRAFT 流程详细信息

   ✅ 加载了 82 个 events
   ✅ 建立了动作-帧映射: 15 个动作
       ✅ 加载了 82 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=15, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=15, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/15: (pick_up, Egg)
         检查动作 6/15: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/15: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/15: (pick_up, Pan)
         检查动作 11/15: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/15: (pick_up, Egg)
         检查动作 15/15: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 15-22 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['01:10', '01:11', '01:12', '01:13', '01:14', '01:15', '01:16', '01:17', '01:18', '01:19', '01:20']
- **Failure Reason**: The robot never cracked the egg and put an uncracked egg in the pan.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [15-22]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [18/100] cookEgg/cookEgg-7

### 数据加载信息

✅ 加载了 115 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, StoveBurner-1)', '(toggle_on, StoveBurner-1)', '(navigate_to_obj, Pan)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-7
   Action 1: (pick_up, Egg) -> Frames 8-15
   Action 2: (navigate_to_obj, StoveBurner-1) -> Frames 16-23
   Action 3: (toggle_on, StoveBurner-1) -> Frames 24-31
   Action 4: (navigate_to_obj, Pan) -> Frames 32-40
   Action 5: (put_on, Egg, CounterTop) -> Frames 41-48
   Action 6: (pick_up, Pan) -> Frames 49-56
   Action 7: (navigate_to_obj, StoveBurner-1) -> Frames 57-64
   Action 8: (put_on, Pan, StoveBurner-1) -> Frames 65-72
   Action 9: (navigate_to_obj, Egg) -> Frames 73-81
   Action 10: (pick_up, Egg) -> Frames 82-89
   Action 11: (navigate_to_obj, Pan) -> Frames 90-97
   Action 12: (crack_obj, Egg) -> Frames 98-105
   Action 13: (put_in, EggCracked, Pan) -> Frames 106-114

### CRAFT 流程详细信息

   ✅ 加载了 115 个 events
   ✅ 建立了动作-帧映射: 14 个动作
       ✅ 加载了 115 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, StoveBurner-1)', '(toggle_on, StoveBurner-1)', '(navigate_to_obj, Pan)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=14, 前3个动作=['(navigate_to_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, StoveBurner-1)']
       🔍 调试：动作 2 ((pick_up, Egg)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/14: (pick_up, Egg)
         检查动作 4/14: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 4-8 内未满足): StoveBurner-1 must be toggled on
         检查动作 6/14: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Egg must be on top of CounterTop
         检查动作 7/14: (pick_up, Pan)
         检查动作 9/14: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pan must be on top of StoveBurner-1
         检查动作 11/14: (pick_up, Egg)
         检查动作 14/14: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 14-21 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:01', '00:02', '00:03', '00:04', '00:05', '00:06', '00:07', '00:08', '00:09', '00:10', '00:11', '00:12', '00:13', '00:14']
- **Failure Reason**: The robot never opened the fridge, as a result, it could not retrieve the egg from fridge.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [14-21]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [19/100] cookEgg/cookEgg-8

### 数据加载信息

✅ 加载了 93 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'scene', 'object_list', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-4
   Action 1: (open_obj, Fridge) -> Frames 5-10
   Action 2: (pick_up, Egg) -> Frames 11-16
   Action 3: (close_obj, Fridge) -> Frames 17-22
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 23-28
   Action 5: (toggle_on, StoveBurner-1) -> Frames 29-33
   Action 6: (navigate_to_obj, Pan) -> Frames 34-39
   Action 7: (put_on, Egg, CounterTop) -> Frames 40-45
   Action 8: (pick_up, Pan) -> Frames 46-51
   Action 9: (navigate_to_obj, StoveBurner-1) -> Frames 52-57
   Action 10: (put_on, Pan, StoveBurner-1) -> Frames 58-62
   Action 11: (navigate_to_obj, Egg) -> Frames 63-68
   Action 12: (pick_up, Egg) -> Frames 69-74
   Action 13: (navigate_to_obj, Pan) -> Frames 75-80
   Action 14: (crack_obj, Egg) -> Frames 81-86
   Action 15: (put_in, EggCracked, Pan) -> Frames 87-92

### CRAFT 流程详细信息

   ✅ 加载了 93 个 events
   ✅ 建立了动作-帧映射: 16 个动作
       ✅ 加载了 93 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'scene', 'object_list']
       🔍 调试：actions 数量=16, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=16, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 3/16: (pick_up, Egg)
         检查动作 6/16: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/16: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Egg must be on top of CounterTop
         检查动作 9/16: (pick_up, Pan)
         检查动作 11/16: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Pan must be on top of StoveBurner-1
         检查动作 13/16: (pick_up, Egg)
         检查动作 16/16: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 16-23 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:47', '00:48', '00:49', '00:50', '00:51', '00:52', '00:53', '00:54', '00:55', '00:56', '00:57', '00:58', '00:59', '01:00']
- **Failure Reason**: The pan is dirty at the beginning of the task execution, and the robot never cleaned the pan.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [16-23]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [20/100] cookEgg/cookEgg-9

### 数据加载信息

✅ 加载了 73 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'gt_failure_reason', 'gt_failure_step', 'scene', 'object_list', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=15, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-3
   Action 1: (open_obj, Fridge) -> Frames 4-8
   Action 2: (pick_up, Egg) -> Frames 9-13
   Action 3: (close_obj, Fridge) -> Frames 14-18
   Action 4: (navigate_to_obj, StoveBurner-1) -> Frames 19-23
   Action 5: (toggle_on, StoveBurner-1) -> Frames 24-28
   Action 6: (navigate_to_obj, Pan) -> Frames 29-33
   Action 7: (pick_up, Pan) -> Frames 34-37
   Action 8: (navigate_to_obj, StoveBurner-1) -> Frames 38-42
   Action 9: (put_on, Pan, StoveBurner-1) -> Frames 43-47
   Action 10: (navigate_to_obj, Egg) -> Frames 48-52
   Action 11: (pick_up, Egg) -> Frames 53-57
   Action 12: (navigate_to_obj, Pan) -> Frames 58-62
   Action 13: (crack_obj, Egg) -> Frames 63-67
   Action 14: (put_in, EggCracked, Pan) -> Frames 68-72

### CRAFT 流程详细信息

   ✅ 加载了 73 个 events
   ✅ 建立了动作-帧映射: 15 个动作
       ✅ 加载了 73 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'gt_failure_reason', 'gt_failure_step', 'scene', 'object_list', 'actions']
       🔍 调试：actions 数量=15, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(close_obj, Fridge)', '(navigate_to_obj, StoveBurner-1)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=15, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 17 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/15: (pick_up, Egg)
         检查动作 6/15: (toggle_on, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 6-10 内未满足): StoveBurner-1 must be toggled on
         检查动作 8/15: (pick_up, Pan)
         检查动作 10/15: (put_on, Pan, StoveBurner-1)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Pan must be on top of StoveBurner-1
         检查动作 12/15: (pick_up, Egg)
         检查动作 15/15: (put_in, EggCracked, Pan)
           ❌ Postcondition 违反 (窗口 15-22 内未满足): EggCracked must be inside Pan
       ✅ 检测完成: 3 个违反, 3 个真实错误, 13 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39', '00:40', '00:41', '00:42', '00:43', '00:44']
- **Failure Reason**: The robot did not put down the egg in its gripper before trying to pick up the pan.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: EggCracked must be inside Pan
      Reason: Postcondition not satisfied in temporal window [15-22]. Last reason: eggcracked is not inside pan (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [21/100] heatPotato/heatPotato-1

### 数据加载信息

✅ 加载了 83 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-6
   Action 1: (pick_up, Potato) -> Frames 7-14
   Action 2: (put_on, Potato, Plate) -> Frames 15-21
   Action 3: (pick_up, Plate) -> Frames 22-29
   Action 4: (navigate_to_obj, Microwave) -> Frames 30-36
   Action 5: (put_on, Plate, CounterTop) -> Frames 37-44
   Action 6: (open_obj, Microwave) -> Frames 45-51
   Action 7: (pick_up, Plate) -> Frames 52-59
   Action 8: (put_in, Plate, Microwave) -> Frames 60-66
   Action 9: (close_obj, Microwave) -> Frames 67-74
   Action 10: (toggle_on, Microwave) -> Frames 75-82

### CRAFT 流程详细信息

   ✅ 加载了 83 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 83 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 4/11: (pick_up, Plate)
         检查动作 6/11: (put_on, Plate, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Plate must be on top of CounterTop
         检查动作 8/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:44
- **Failure Reason**: Dropped Plate

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [11-15]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [22/100] heatPotato/heatPotato-10

### 数据加载信息

✅ 加载了 123 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-10
   Action 1: (pick_up, Potato) -> Frames 11-21
   Action 2: (put_on, Potato, Plate) -> Frames 22-32
   Action 3: (pick_up, Plate) -> Frames 33-43
   Action 4: (navigate_to_obj, Microwave) -> Frames 44-54
   Action 5: (put_on, Plate, CounterTop) -> Frames 55-66
   Action 6: (toggle_on, Microwave) -> Frames 67-77
   Action 7: (open_obj, Microwave) -> Frames 78-88
   Action 8: (pick_up, Plate) -> Frames 89-99
   Action 9: (put_in, Plate, Microwave) -> Frames 100-110
   Action 10: (close_obj, Microwave) -> Frames 111-122

### CRAFT 流程详细信息

   ✅ 加载了 123 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 123 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 4/11: (pick_up, Plate)
         检查动作 6/11: (put_on, Plate, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Plate must be on top of CounterTop
         检查动作 7/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 7-11 内未满足): Microwave must be toggled on
         检查动作 9/11: (pick_up, Plate)
         检查动作 10/11: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Plate must be inside Microwave
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong order
- **Failure Step**: ['01:31', '01:32', '01:33', '01:34']
- **Failure Reason**: The robot should not toggle on the microwave before trying to open it. As a result, the robot cannot open a microwave that is turned on.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Plate must be inside Microwave
      Reason: Postcondition not satisfied in temporal window [10-17]. Last reason: plate is not inside microwave (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [23/100] heatPotato/heatPotato-2

### 数据加载信息

✅ 加载了 123 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-10
   Action 1: (pick_up, Potato) -> Frames 11-21
   Action 2: (put_on, Potato, Plate) -> Frames 22-32
   Action 3: (pick_up, Plate) -> Frames 33-43
   Action 4: (navigate_to_obj, Microwave) -> Frames 44-54
   Action 5: (put_on, Plate, CounterTop) -> Frames 55-66
   Action 6: (open_obj, Microwave) -> Frames 67-77
   Action 7: (pick_up, Plate) -> Frames 78-88
   Action 8: (put_in, Plate, Microwave) -> Frames 89-99
   Action 9: (close_obj, Microwave) -> Frames 100-110
   Action 10: (toggle_on, Microwave) -> Frames 111-122

### CRAFT 流程详细信息

   ✅ 加载了 123 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 123 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 4/11: (pick_up, Plate)
         检查动作 6/11: (put_on, Plate, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Plate must be on top of CounterTop
         检查动作 8/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
           ❌ Precondition 违反: Microwave must be empty
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
       ✅ 检测完成: 5 个违反, 5 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: occupied_put
- **Failure Step**: 01:55
- **Failure Reason**: A bowl is already inside the microwave, as a result, the plate cannot be put inside the microwave due to limited space.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be empty
      Reason: Container 'microwave' contains 1 object(s): Bowl_c7b0b2d2
      Frame: Unknown frame

    Derived Violations (派生失败, 1 个):
      这些失败是由根失败导致的级联失败，不单独分析


================================================================================

## [24/100] heatPotato/heatPotato-3

### 数据加载信息

✅ 加载了 121 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-10
   Action 1: (pick_up, Potato) -> Frames 11-21
   Action 2: (put_on, Potato, Plate) -> Frames 22-32
   Action 3: (pick_up, Plate) -> Frames 33-43
   Action 4: (navigate_to_obj, Microwave) -> Frames 44-54
   Action 5: (put_on, Plate, CounterTop) -> Frames 55-65
   Action 6: (open_obj, Microwave) -> Frames 66-76
   Action 7: (pick_up, Plate) -> Frames 77-87
   Action 8: (put_in, Plate, Microwave) -> Frames 88-98
   Action 9: (close_obj, Microwave) -> Frames 99-109
   Action 10: (toggle_on, Microwave) -> Frames 110-120

### CRAFT 流程详细信息

   ✅ 加载了 121 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 121 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 4/11: (pick_up, Plate)
         检查动作 6/11: (put_on, Plate, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Plate must be on top of CounterTop
         检查动作 8/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: occupied
- **Failure Step**: ['00:37', '00:38', '00:39', '00:40']
- **Failure Reason**: An apple is inside the plate already and the robot never removed it, as a result, the potato cannot be put on top of the plate due to limited plate.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [11-15]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [25/100] heatPotato/heatPotato-4

### 数据加载信息

✅ 加载了 135 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(navigate_to_obj, Microwave)', '(open_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-11
   Action 1: (pick_up, Potato) -> Frames 12-23
   Action 2: (put_on, Potato, Plate) -> Frames 24-35
   Action 3: (navigate_to_obj, Microwave) -> Frames 36-48
   Action 4: (open_obj, Microwave) -> Frames 49-60
   Action 5: (navigate_to_obj, Plate) -> Frames 61-72
   Action 6: (pick_up, Plate) -> Frames 73-84
   Action 7: (navigate_to_obj, Microwave) -> Frames 85-97
   Action 8: (put_in, Plate, Microwave) -> Frames 98-109
   Action 9: (close_obj, Microwave) -> Frames 110-121
   Action 10: (toggle_on, Microwave) -> Frames 122-134

### CRAFT 流程详细信息

   ✅ 加载了 135 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 135 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(navigate_to_obj, Microwave)', '(open_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 7/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 01:07
- **Failure Reason**: The robot failed to open the microwave.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [11-15]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [26/100] heatPotato/heatPotato-5

### 数据加载信息

✅ 加载了 74 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=7, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(navigate_to_obj, Plate)', '(put_on, Potato, Plate)', '(pick_up, Plate)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-9
   Action 1: (pick_up, Potato) -> Frames 10-20
   Action 2: (navigate_to_obj, Plate) -> Frames 21-30
   Action 3: (put_on, Potato, Plate) -> Frames 31-41
   Action 4: (pick_up, Plate) -> Frames 42-51
   Action 5: (navigate_to_obj, Microwave) -> Frames 52-62
   Action 6: (put_in, Plate, Microwave) -> Frames 63-73

### CRAFT 流程详细信息

   ✅ 加载了 74 个 events
   ✅ 建立了动作-帧映射: 7 个动作
       ✅ 加载了 74 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=7, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(navigate_to_obj, Plate)', '(put_on, Potato, Plate)', '(pick_up, Plate)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=7, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(navigate_to_obj, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 12 个约束
       组织约束...
       ✅ 约束分组: 4 个动作有约束, 0 个目标约束
         检查动作 2/7: (pick_up, Potato)
         检查动作 4/7: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Potato must be on top of Plate
         检查动作 5/7: (pick_up, Plate)
         检查动作 7/7: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 7-14 内未满足): Plate must be inside Microwave
       ✅ 检测完成: 2 个违反, 2 个真实错误, 9 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:41', '00:42', '00:43', '00:44', '00:45', '00:46', '00:47', '00:48', '00:49', '00:50', '00:51', '00:52', '00:53', '00:54', '00:55', '00:56', '00:57', '00:58', '00:59', '01:00', '01:01', '01:02', '01:03', '01:04', '01:05', '01:06', '01:07', '01:08', '01:09', '01:10', '01:11', '01:12']
- **Failure Reason**: The robot never opened the microwave, as a result, the plate cannot be put inside the microwave.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Plate must be inside Microwave
      Reason: Postcondition not satisfied in temporal window [7-14]. Last reason: plate is not inside microwave (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [27/100] heatPotato/heatPotato-6

### 数据加载信息

✅ 加载了 112 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-9
   Action 1: (pick_up, Potato) -> Frames 10-19
   Action 2: (put_on, Potato, Plate) -> Frames 20-29
   Action 3: (pick_up, Plate) -> Frames 30-39
   Action 4: (navigate_to_obj, Microwave) -> Frames 40-49
   Action 5: (put_on, Plate, CounterTop) -> Frames 50-60
   Action 6: (open_obj, Microwave) -> Frames 61-70
   Action 7: (pick_up, Plate) -> Frames 71-80
   Action 8: (put_in, Plate, Microwave) -> Frames 81-90
   Action 9: (close_obj, Microwave) -> Frames 91-100
   Action 10: (toggle_on, Microwave) -> Frames 101-111

### CRAFT 流程详细信息

   ✅ 加载了 112 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 112 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'scene', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 4/11: (pick_up, Plate)
         检查动作 6/11: (put_on, Plate, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Plate must be on top of CounterTop
         检查动作 8/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong perception
- **Failure Step**: 00:23
- **Failure Reason**: The robot mis-identified apple as potato, and heated an apple instead of a potato.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [11-15]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [28/100] heatPotato/heatPotato-7

### 数据加载信息

✅ 加载了 123 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-10
   Action 1: (pick_up, Potato) -> Frames 11-21
   Action 2: (put_on, Potato, Plate) -> Frames 22-32
   Action 3: (pick_up, Plate) -> Frames 33-43
   Action 4: (navigate_to_obj, Microwave) -> Frames 44-54
   Action 5: (put_on, Plate, CounterTop) -> Frames 55-66
   Action 6: (open_obj, Microwave) -> Frames 67-77
   Action 7: (pick_up, Plate) -> Frames 78-88
   Action 8: (put_in, Plate, Microwave) -> Frames 89-99
   Action 9: (close_obj, Microwave) -> Frames 100-110
   Action 10: (toggle_on, Microwave) -> Frames 111-122

### CRAFT 流程详细信息

   ✅ 加载了 123 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 123 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 4/11: (pick_up, Plate)
         检查动作 6/11: (put_on, Plate, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Plate must be on top of CounterTop
         检查动作 8/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:19', '00:20', '00:21', '00:22', '00:23', '00:24', '00:25', '00:26', '00:27', '00:28', '00:29', '00:30', '00:31', '00:32', '00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39']
- **Failure Reason**: The plate is dirty at the beginning of the task execution, and the robot never cleaned it.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [11-15]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [29/100] heatPotato/heatPotato-8

### 数据加载信息

✅ 加载了 113 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Pan)', '(pick_up, Pan)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-9
   Action 1: (pick_up, Potato) -> Frames 10-19
   Action 2: (put_on, Potato, Pan) -> Frames 20-29
   Action 3: (pick_up, Pan) -> Frames 30-40
   Action 4: (navigate_to_obj, Microwave) -> Frames 41-50
   Action 5: (put_on, Pan, CounterTop) -> Frames 51-60
   Action 6: (open_obj, Microwave) -> Frames 61-70
   Action 7: (pick_up, Pan) -> Frames 71-81
   Action 8: (put_in, Pan, Microwave) -> Frames 82-91
   Action 9: (close_obj, Microwave) -> Frames 92-101
   Action 10: (toggle_on, Microwave) -> Frames 102-112

### CRAFT 流程详细信息

   ✅ 加载了 113 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 113 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Pan)', '(pick_up, Pan)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Pan)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Pan)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be inside Pan
         检查动作 4/11: (pick_up, Pan)
         检查动作 6/11: (put_on, Pan, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Pan must be on top of CounterTop
         检查动作 8/11: (pick_up, Pan)
         检查动作 9/11: (put_in, Pan, Microwave)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Pan must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:34
- **Failure Reason**: The robot should use a microwave-safe container (e.g. Plate) to heat the potato instead of a pan.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [11-15]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [30/100] heatPotato/heatPotato-9

### 数据加载信息

✅ 加载了 123 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Potato) -> Frames 0-10
   Action 1: (pick_up, Potato) -> Frames 11-21
   Action 2: (put_on, Potato, Plate) -> Frames 22-32
   Action 3: (pick_up, Plate) -> Frames 33-43
   Action 4: (navigate_to_obj, Microwave) -> Frames 44-54
   Action 5: (put_on, Plate, CounterTop) -> Frames 55-66
   Action 6: (open_obj, Microwave) -> Frames 67-77
   Action 7: (pick_up, Plate) -> Frames 78-88
   Action 8: (put_in, Plate, Microwave) -> Frames 89-99
   Action 9: (toggle_on, Microwave) -> Frames 100-110
   Action 10: (close_obj, Microwave) -> Frames 111-122

### CRAFT 流程详细信息

   ✅ 加载了 123 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 123 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)', '(pick_up, Plate)', '(navigate_to_obj, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Potato)', '(pick_up, Potato)', '(put_on, Potato, Plate)']
       🔍 调试：动作 2 ((pick_up, Potato)) 生成了 3 个约束
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 7 个动作有约束, 0 个目标约束
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 4/11: (pick_up, Plate)
         检查动作 6/11: (put_on, Plate, CounterTop)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Plate must be on top of CounterTop
         检查动作 8/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 10/11: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): Microwave must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong order
- **Failure Step**: ['01:58', '01:59', '02:00', '02:01']
- **Failure Reason**: The robot should have closed the microwave before trying to toggle it on. As a result, the robot could not toggle on a microwave that is open.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [10-14]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [31/100] makeCoffee/makeCoffee-1

### 数据加载信息

✅ 加载了 50 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-2
   Action 1: (pick_up, Mug) -> Frames 3-6
   Action 2: (navigate_to_obj, Sink) -> Frames 7-9
   Action 3: (put_on, Mug, SinkBasin) -> Frames 10-13
   Action 4: (toggle_on, Faucet) -> Frames 14-16
   Action 5: (toggle_off, Faucet) -> Frames 17-20
   Action 6: (pick_up, Mug) -> Frames 21-24
   Action 7: (pour, Mug, Sink) -> Frames 25-27
   Action 8: (navigate_to_obj, CoffeeMachine) -> Frames 28-31
   Action 9: (put_in, Mug, CoffeeMachine) -> Frames 32-34
   Action 10: (toggle_on, CoffeeMachine) -> Frames 35-38
   Action 11: (toggle_off, CoffeeMachine) -> Frames 39-41
   Action 12: (pick_up, Mug) -> Frames 42-45
   Action 13: (put_on, Mug, CounterTop) -> Frames 46-49

### CRAFT 流程详细信息

   ✅ 加载了 50 个 events
   ✅ 建立了动作-帧映射: 14 个动作
       ✅ 加载了 50 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps']
       🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=14, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 11 个动作有约束, 0 个目标约束
         检查动作 2/14: (pick_up, Mug)
         检查动作 4/14: (put_on, Mug, SinkBasin)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside SinkBasin
         检查动作 5/14: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/14: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/14: (pick_up, Mug)
         检查动作 8/14: (pour, Mug, Sink)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
         检查动作 10/14: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Mug must be inside CoffeeMachine
         检查动作 11/14: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): CoffeeMachine must be toggled on
         检查动作 12/14: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 12 (窗口 12-16) 满足): CoffeeMachine must be toggled off
         检查动作 13/14: (pick_up, Mug)
         检查动作 14/14: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 14-21 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 6 个违反, 6 个真实错误, 19 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:19
- **Failure Reason**: Dropped Mug

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [14-21]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [32/100] makeCoffee/makeCoffee-10

### 数据加载信息

✅ 加载了 62 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-3
   Action 1: (pick_up, Mug) -> Frames 4-7
   Action 2: (navigate_to_obj, Sink) -> Frames 8-12
   Action 3: (put_on, Mug, SinkBasin) -> Frames 13-16
   Action 4: (toggle_on, Faucet) -> Frames 17-21
   Action 5: (toggle_off, Faucet) -> Frames 22-25
   Action 6: (pick_up, Mug) -> Frames 26-30
   Action 7: (pour, Mug, Sink) -> Frames 31-34
   Action 8: (navigate_to_obj, CoffeeMachine) -> Frames 35-38
   Action 9: (toggle_on, CoffeeMachine) -> Frames 39-43
   Action 10: (toggle_off, CoffeeMachine) -> Frames 44-47
   Action 11: (put_in, Mug, CoffeeMachine) -> Frames 48-52
   Action 12: (pick_up, Mug) -> Frames 53-56
   Action 13: (put_on, Mug, CounterTop) -> Frames 57-61

### CRAFT 流程详细信息

   ✅ 加载了 62 个 events
   ✅ 建立了动作-帧映射: 14 个动作
       ✅ 加载了 62 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'preactions']
       🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=14, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 11 个动作有约束, 0 个目标约束
         检查动作 2/14: (pick_up, Mug)
         检查动作 4/14: (put_on, Mug, SinkBasin)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside SinkBasin
         检查动作 5/14: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/14: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/14: (pick_up, Mug)
         检查动作 8/14: (pour, Mug, Sink)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
         检查动作 10/14: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): CoffeeMachine must be toggled on
         检查动作 11/14: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 11 (窗口 11-15) 满足): CoffeeMachine must be toggled off
         检查动作 12/14: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside CoffeeMachine
         检查动作 13/14: (pick_up, Mug)
         检查动作 14/14: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 14-21 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 6 个违反, 6 个真实错误, 19 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong order
- **Failure Step**: ['00:48', '00:49', '00:50', '00:51', '00:52', '00:53', '00:54']
- **Failure Reason**: The robot put the mug inside the coffee machine after the coffee machine was turned off, as a result, the mug remained empty.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [14-21]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [33/100] makeCoffee/makeCoffee-2

### 数据加载信息

✅ 加载了 56 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-3
   Action 1: (pick_up, Mug) -> Frames 4-7
   Action 2: (navigate_to_obj, Sink) -> Frames 8-11
   Action 3: (put_on, Mug, SinkBasin) -> Frames 12-15
   Action 4: (toggle_on, Faucet) -> Frames 16-19
   Action 5: (toggle_off, Faucet) -> Frames 20-23
   Action 6: (pick_up, Mug) -> Frames 24-27
   Action 7: (pour, Mug, Sink) -> Frames 28-31
   Action 8: (navigate_to_obj, CoffeeMachine) -> Frames 32-35
   Action 9: (put_in, Mug, CoffeeMachine) -> Frames 36-39
   Action 10: (toggle_on, CoffeeMachine) -> Frames 40-43
   Action 11: (toggle_off, CoffeeMachine) -> Frames 44-47
   Action 12: (pick_up, Mug) -> Frames 48-51
   Action 13: (put_on, Mug, CounterTop) -> Frames 52-55

### CRAFT 流程详细信息

   ✅ 加载了 56 个 events
   ✅ 建立了动作-帧映射: 14 个动作
       ✅ 加载了 56 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps']
       🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=14, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 11 个动作有约束, 0 个目标约束
         检查动作 2/14: (pick_up, Mug)
         检查动作 4/14: (put_on, Mug, SinkBasin)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside SinkBasin
         检查动作 5/14: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/14: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/14: (pick_up, Mug)
         检查动作 8/14: (pour, Mug, Sink)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
         检查动作 10/14: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Mug must be inside CoffeeMachine
         检查动作 11/14: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): CoffeeMachine must be toggled on
         检查动作 12/14: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 12 (窗口 12-16) 满足): CoffeeMachine must be toggled off
         检查动作 13/14: (pick_up, Mug)
         检查动作 14/14: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 14-21 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 6 个违反, 6 个真实错误, 19 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:38
- **Failure Reason**: Dropped Mug

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [14-21]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [34/100] makeCoffee/makeCoffee-3

### 数据加载信息

✅ 加载了 78 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'chosen_failure', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-4
   Action 1: (pick_up, Mug) -> Frames 5-10
   Action 2: (navigate_to_obj, Sink) -> Frames 11-15
   Action 3: (put_on, Mug, SinkBasin) -> Frames 16-21
   Action 4: (toggle_on, Faucet) -> Frames 22-26
   Action 5: (toggle_off, Faucet) -> Frames 27-32
   Action 6: (pick_up, Mug) -> Frames 33-38
   Action 7: (pour, Mug, Sink) -> Frames 39-43
   Action 8: (navigate_to_obj, CoffeeMachine) -> Frames 44-49
   Action 9: (put_in, Mug, CoffeeMachine) -> Frames 50-54
   Action 10: (toggle_on, CoffeeMachine) -> Frames 55-60
   Action 11: (toggle_off, CoffeeMachine) -> Frames 61-65
   Action 12: (pick_up, Mug) -> Frames 66-71
   Action 13: (put_on, Mug, CounterTop) -> Frames 72-77

### CRAFT 流程详细信息

   ✅ 加载了 78 个 events
   ✅ 建立了动作-帧映射: 14 个动作
       ✅ 加载了 78 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'chosen_failure', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions']
       🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=14, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 11 个动作有约束, 0 个目标约束
         检查动作 2/14: (pick_up, Mug)
         检查动作 4/14: (put_on, Mug, SinkBasin)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside SinkBasin
         检查动作 5/14: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/14: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/14: (pick_up, Mug)
         检查动作 8/14: (pour, Mug, Sink)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
         检查动作 10/14: (put_in, Mug, CoffeeMachine)
           ❌ Precondition 违反: CoffeeMachine must be empty
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Mug must be inside CoffeeMachine
         检查动作 11/14: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): CoffeeMachine must be toggled on
         检查动作 12/14: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 12 (窗口 12-16) 满足): CoffeeMachine must be toggled off
         检查动作 13/14: (pick_up, Mug)
         检查动作 14/14: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 14-21 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 7 个违反, 7 个真实错误, 19 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: occupied
- **Failure Step**: 0
- **Failure Reason**: The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: CoffeeMachine must be empty
      Reason: Container 'coffeemachine' contains 1 object(s): Cup_dd30ad4b
      Frame: Unknown frame

    Derived Violations (派生失败, 2 个):
      这些失败是由根失败导致的级联失败，不单独分析


================================================================================

## [35/100] makeCoffee/makeCoffee-4

### 数据加载信息

✅ 加载了 44 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, CoffeeMachine)', '(put_in, Mug, CoffeeMachine)', '(toggle_on, CoffeeMachine)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-4
   Action 1: (pick_up, Mug) -> Frames 5-10
   Action 2: (navigate_to_obj, CoffeeMachine) -> Frames 11-15
   Action 3: (put_in, Mug, CoffeeMachine) -> Frames 16-21
   Action 4: (toggle_on, CoffeeMachine) -> Frames 22-26
   Action 5: (toggle_off, CoffeeMachine) -> Frames 27-32
   Action 6: (pick_up, Mug) -> Frames 33-37
   Action 7: (put_on, Mug, CounterTop) -> Frames 38-43

### CRAFT 流程详细信息

   ✅ 加载了 44 个 events
   ✅ 建立了动作-帧映射: 8 个动作
       ✅ 加载了 44 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions']
       🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, CoffeeMachine)', '(put_in, Mug, CoffeeMachine)', '(toggle_on, CoffeeMachine)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=8, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, CoffeeMachine)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/8: (pick_up, Mug)
         检查动作 4/8: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside CoffeeMachine
         检查动作 5/8: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): CoffeeMachine must be toggled on
         检查动作 6/8: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): CoffeeMachine must be toggled off
         检查动作 7/8: (pick_up, Mug)
         检查动作 8/8: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 3 个违反, 3 个真实错误, 11 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:17', '00:18', '00:19', '00:20', '00:21', '00:22', '00:23', '00:24', '00:25', '00:26', '00:27', '00:28', '00:29', '00:30']
- **Failure Reason**: The mug was already filled with water at the beginning of the task execution, and the robot never emptied it.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [8-15]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [36/100] makeCoffee/makeCoffee-5

### 数据加载信息

✅ 加载了 58 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-3
   Action 1: (pick_up, Mug) -> Frames 4-7
   Action 2: (navigate_to_obj, Sink) -> Frames 8-11
   Action 3: (put_on, Mug, SinkBasin) -> Frames 12-15
   Action 4: (toggle_on, Faucet) -> Frames 16-19
   Action 5: (toggle_off, Faucet) -> Frames 20-23
   Action 6: (pick_up, Mug) -> Frames 24-28
   Action 7: (pour, Mug, Sink) -> Frames 29-32
   Action 8: (navigate_to_obj, CoffeeMachine) -> Frames 33-36
   Action 9: (put_in, Mug, CoffeeMachine) -> Frames 37-40
   Action 10: (toggle_on, CoffeeMachine) -> Frames 41-44
   Action 11: (toggle_off, CoffeeMachine) -> Frames 45-48
   Action 12: (pick_up, Mug) -> Frames 49-52
   Action 13: (put_on, Mug, CounterTop) -> Frames 53-57

### CRAFT 流程详细信息

   ✅ 加载了 58 个 events
   ✅ 建立了动作-帧映射: 14 个动作
       ✅ 加载了 58 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps']
       🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=14, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 11 个动作有约束, 0 个目标约束
         检查动作 2/14: (pick_up, Mug)
         检查动作 4/14: (put_on, Mug, SinkBasin)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside SinkBasin
         检查动作 5/14: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/14: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/14: (pick_up, Mug)
         检查动作 8/14: (pour, Mug, Sink)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
         检查动作 10/14: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Mug must be inside CoffeeMachine
         检查动作 11/14: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): CoffeeMachine must be toggled on
         检查动作 12/14: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 12 (窗口 12-16) 满足): CoffeeMachine must be toggled off
         检查动作 13/14: (pick_up, Mug)
         检查动作 14/14: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 14-21 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 6 个违反, 6 个真实错误, 19 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:22
- **Failure Reason**: The robot failed to put the mug inside the sink (or on top of the sink basin).

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [14-21]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [37/100] makeCoffee/makeCoffee-6

### 数据加载信息

✅ 加载了 82 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-4
   Action 1: (pick_up, Mug) -> Frames 5-10
   Action 2: (navigate_to_obj, Sink) -> Frames 11-16
   Action 3: (put_on, Mug, SinkBasin) -> Frames 17-22
   Action 4: (toggle_on, Faucet) -> Frames 23-28
   Action 5: (toggle_off, Faucet) -> Frames 29-34
   Action 6: (pick_up, Mug) -> Frames 35-40
   Action 7: (pour, Mug, Sink) -> Frames 41-45
   Action 8: (navigate_to_obj, CoffeeMachine) -> Frames 46-51
   Action 9: (put_in, Mug, CoffeeMachine) -> Frames 52-57
   Action 10: (toggle_on, CoffeeMachine) -> Frames 58-63
   Action 11: (toggle_off, CoffeeMachine) -> Frames 64-69
   Action 12: (pick_up, Mug) -> Frames 70-75
   Action 13: (put_on, Mug, CounterTop) -> Frames 76-81

### CRAFT 流程详细信息

   ✅ 加载了 82 个 events
   ✅ 建立了动作-帧映射: 14 个动作
       ✅ 加载了 82 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'preactions']
       🔍 调试：actions 数量=14, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=14, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 11 个动作有约束, 0 个目标约束
         检查动作 2/14: (pick_up, Mug)
         检查动作 4/14: (put_on, Mug, SinkBasin)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside SinkBasin
         检查动作 5/14: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/14: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/14: (pick_up, Mug)
         检查动作 8/14: (pour, Mug, Sink)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
         检查动作 10/14: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Mug must be inside CoffeeMachine
         检查动作 11/14: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 11-15 内未满足): CoffeeMachine must be toggled on
         检查动作 12/14: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 12 (窗口 12-16) 满足): CoffeeMachine must be toggled off
         检查动作 13/14: (pick_up, Mug)
         检查动作 14/14: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 14-21 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 6 个违反, 6 个真实错误, 19 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong perception
- **Failure Step**: 00:23
- **Failure Reason**: The robot mis-identified bowl as mug, as a result, the bowl cannot be put inside the coffee machine.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [14-21]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [38/100] makeCoffee/makeCoffee-7

### 数据加载信息

✅ 加载了 61 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=13, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-3
   Action 1: (pick_up, Mug) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-13
   Action 3: (put_on, Mug, SinkBasin) -> Frames 14-17
   Action 4: (toggle_on, Faucet) -> Frames 18-22
   Action 5: (toggle_off, Faucet) -> Frames 23-27
   Action 6: (pick_up, Mug) -> Frames 28-31
   Action 7: (navigate_to_obj, CoffeeMachine) -> Frames 32-36
   Action 8: (put_in, Mug, CoffeeMachine) -> Frames 37-41
   Action 9: (toggle_on, CoffeeMachine) -> Frames 42-45
   Action 10: (toggle_off, CoffeeMachine) -> Frames 46-50
   Action 11: (pick_up, Mug) -> Frames 51-55
   Action 12: (put_on, Mug, CounterTop) -> Frames 56-60

### CRAFT 流程详细信息

   ✅ 加载了 61 个 events
   ✅ 建立了动作-帧映射: 13 个动作
       ✅ 加载了 61 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'actions', 'specified_missing_steps']
       🔍 调试：actions 数量=13, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_on, Mug, SinkBasin)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=13, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 25 个约束
       组织约束...
       ✅ 约束分组: 10 个动作有约束, 0 个目标约束
         检查动作 2/13: (pick_up, Mug)
         检查动作 4/13: (put_on, Mug, SinkBasin)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside SinkBasin
         检查动作 5/13: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/13: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/13: (pick_up, Mug)
         检查动作 9/13: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be inside CoffeeMachine
         检查动作 10/13: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): CoffeeMachine must be toggled on
         检查动作 11/13: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 11 (窗口 11-15) 满足): CoffeeMachine must be toggled off
         检查动作 12/13: (pick_up, Mug)
         检查动作 13/13: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 13-20 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 5 个违反, 5 个真实错误, 17 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:32', '00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39', '00:40', '00:41', '00:42', '00:43', '00:44', '00:45', '00:46', '00:47']
- **Failure Reason**: The robot never executed the action to pour water from mug after cleaning it. As a result, the mug cannot be filled with coffee.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [13-20]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [39/100] makeCoffee/makeCoffee-8

### 数据加载信息

✅ 加载了 44 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, CoffeeMachine)', '(put_in, Mug, CoffeeMachine)', '(toggle_on, CoffeeMachine)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-4
   Action 1: (pick_up, Mug) -> Frames 5-10
   Action 2: (navigate_to_obj, CoffeeMachine) -> Frames 11-15
   Action 3: (put_in, Mug, CoffeeMachine) -> Frames 16-21
   Action 4: (toggle_on, CoffeeMachine) -> Frames 22-26
   Action 5: (toggle_off, CoffeeMachine) -> Frames 27-32
   Action 6: (pick_up, Mug) -> Frames 33-37
   Action 7: (put_on, Mug, CounterTop) -> Frames 38-43

### CRAFT 流程详细信息

   ✅ 加载了 44 个 events
   ✅ 建立了动作-帧映射: 8 个动作
       ✅ 加载了 44 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions']
       🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, CoffeeMachine)', '(put_in, Mug, CoffeeMachine)', '(toggle_on, CoffeeMachine)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=8, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, CoffeeMachine)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/8: (pick_up, Mug)
         检查动作 4/8: (put_in, Mug, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside CoffeeMachine
         检查动作 5/8: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): CoffeeMachine must be toggled on
         检查动作 6/8: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): CoffeeMachine must be toggled off
         检查动作 7/8: (pick_up, Mug)
         检查动作 8/8: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Mug must be on top of CounterTop
       ✅ 检测完成: 3 个违反, 3 个真实错误, 11 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:17', '00:18', '00:19', '00:20', '00:21', '00:22', '00:23', '00:24', '00:25', '00:26', '00:27', '00:28', '00:29', '00:30']
- **Failure Reason**: The mug is dirty at the beginning of the task execution, and the robot never cleaned the mug.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [8-15]. Last reason: mug is not on top of countertop
      Frame: Unknown frame


================================================================================

## [40/100] makeCoffee/makeCoffee-9

### 数据加载信息

✅ 加载了 64 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=7, actions=['(pick_up, Bowl)', '(navigate_to_obj, CoffeeMachine)', '(put_in, Bowl, CoffeeMachine)', '(toggle_on, CoffeeMachine)', '(toggle_off, CoffeeMachine)']

### 动作-帧映射

   Action 0: (pick_up, Bowl) -> Frames 0-8
   Action 1: (navigate_to_obj, CoffeeMachine) -> Frames 9-17
   Action 2: (put_in, Bowl, CoffeeMachine) -> Frames 18-26
   Action 3: (toggle_on, CoffeeMachine) -> Frames 27-35
   Action 4: (toggle_off, CoffeeMachine) -> Frames 36-44
   Action 5: (pick_up, Bowl) -> Frames 45-53
   Action 6: (put_on, Bowl, CounterTop) -> Frames 54-63

### CRAFT 流程详细信息

   ✅ 加载了 64 个 events
   ✅ 建立了动作-帧映射: 7 个动作
       ✅ 加载了 64 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=7, actions=['(pick_up, Bowl)', '(navigate_to_obj, CoffeeMachine)', '(put_in, Bowl, CoffeeMachine)', '(toggle_on, CoffeeMachine)', '(toggle_off, CoffeeMachine)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=7, 前3个动作=['(pick_up, Bowl)', '(navigate_to_obj, CoffeeMachine)', '(put_in, Bowl, CoffeeMachine)']
       🔍 调试：动作 1 ((pick_up, Bowl)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 1/7: (pick_up, Bowl)
         检查动作 3/7: (put_in, Bowl, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Bowl must be inside CoffeeMachine
         检查动作 4/7: (toggle_on, CoffeeMachine)
           ❌ Postcondition 违反 (窗口 4-8 内未满足): CoffeeMachine must be toggled on
         检查动作 5/7: (toggle_off, CoffeeMachine)
           ✅ Postcondition 满足 (在 帧 5 (窗口 5-9) 满足): CoffeeMachine must be toggled off
         检查动作 6/7: (pick_up, Bowl)
         检查动作 7/7: (put_on, Bowl, CounterTop)
           ❌ Postcondition 违反 (窗口 7-14 内未满足): Bowl must be on top of CounterTop
       ✅ 检测完成: 3 个违反, 3 个真实错误, 11 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:23
- **Failure Reason**: The robot plan should not use a bowl instead of a mug or cup to make coffee. As a result, the bowl cannot be put inside the coffee machine.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Bowl must be on top of CounterTop
      Reason: Postcondition not satisfied in temporal window [7-14]. Last reason: bowl is not on top of countertop
      Frame: Unknown frame


================================================================================

## [41/100] makeSalad/makeSalad-1

### 数据加载信息

✅ 加载了 211 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-12
   Action 2: (navigate_to_obj, Bowl) -> Frames 13-18
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 19-25
   Action 4: (navigate_to_obj, Tomato) -> Frames 26-31
   Action 5: (pick_up, Tomato) -> Frames 32-38
   Action 6: (navigate_to_obj, Bowl) -> Frames 39-45
   Action 7: (put_on, Tomato, CounterTop) -> Frames 46-51
   Action 8: (navigate_to_obj, Potato) -> Frames 52-58
   Action 9: (pick_up, Potato) -> Frames 59-64
   Action 10: (navigate_to_obj, Bowl) -> Frames 65-71
   Action 11: (put_on, Potato, CounterTop) -> Frames 72-78
   Action 12: (navigate_to_obj, Knife) -> Frames 79-84
   Action 13: (pick_up, Knife) -> Frames 85-91
   Action 14: (navigate_to_obj, Bowl) -> Frames 92-97
   Action 15: (slice_obj, Lettuce) -> Frames 98-104
   Action 16: (slice_obj, Tomato) -> Frames 105-111
   Action 17: (put_on, Knife, CounterTop) -> Frames 112-117
   Action 18: (slice_obj, Potato) -> Frames 118-124
   Action 19: (pick_up, LettuceSliced) -> Frames 125-130
   Action 20: (put_in, LettuceSliced, Bowl) -> Frames 131-137
   Action 21: (pick_up, TomatoSliced) -> Frames 138-144
   Action 22: (put_in, TomatoSliced, Bowl) -> Frames 145-150
   Action 23: (pick_up, PotatoSliced) -> Frames 151-157
   Action 24: (put_in, PotatoSliced, Bowl) -> Frames 158-163
   Action 25: (navigate_to_obj, Fridge) -> Frames 164-170
   Action 26: (open_obj, Fridge) -> Frames 171-177
   Action 27: (navigate_to_obj, Bowl) -> Frames 178-183
   Action 28: (pick_up, Bowl) -> Frames 184-190
   Action 29: (navigate_to_obj, Fridge) -> Frames 191-196
   Action 30: (put_in, Bowl, Fridge) -> Frames 197-203
   Action 31: (close_obj, Fridge) -> Frames 204-210

### CRAFT 流程详细信息

   ✅ 加载了 211 个 events
   ✅ 建立了动作-帧映射: 32 个动作
       ✅ 加载了 211 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=32, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/32: (pick_up, Lettuce)
         检查动作 4/32: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/32: (pick_up, Tomato)
         检查动作 8/32: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/32: (pick_up, Potato)
         检查动作 12/32: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/32: (pick_up, Knife)
         检查动作 18/32: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 18-25 内未满足): Knife must be on top of CounterTop
         检查动作 20/32: (pick_up, LettuceSliced)
         检查动作 21/32: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): LettuceSliced must be inside Bowl
         检查动作 22/32: (pick_up, TomatoSliced)
         检查动作 23/32: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): TomatoSliced must be inside Bowl
         检查动作 24/32: (pick_up, PotatoSliced)
         检查动作 25/32: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 25-32 内未满足): PotatoSliced must be inside Bowl
         检查动作 29/32: (pick_up, Bowl)
         检查动作 31/32: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 31-38 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['02:04', '02:05']
- **Failure Reason**: Wrong order - knife is put on countertop before slicing tomato

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [42/100] makeSalad/makeSalad-10

### 数据加载信息

✅ 加载了 217 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-12
   Action 2: (navigate_to_obj, Bowl) -> Frames 13-19
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 20-26
   Action 4: (navigate_to_obj, Tomato) -> Frames 27-32
   Action 5: (pick_up, Tomato) -> Frames 33-39
   Action 6: (navigate_to_obj, Bowl) -> Frames 40-46
   Action 7: (put_on, Tomato, CounterTop) -> Frames 47-53
   Action 8: (navigate_to_obj, Potato) -> Frames 54-60
   Action 9: (pick_up, Potato) -> Frames 61-66
   Action 10: (navigate_to_obj, Bowl) -> Frames 67-73
   Action 11: (put_on, Potato, CounterTop) -> Frames 74-80
   Action 12: (navigate_to_obj, Knife) -> Frames 81-87
   Action 13: (pick_up, Knife) -> Frames 88-93
   Action 14: (navigate_to_obj, Bowl) -> Frames 94-100
   Action 15: (slice_obj, Lettuce) -> Frames 101-107
   Action 16: (slice_obj, Potato) -> Frames 108-114
   Action 17: (slice_obj, Tomato) -> Frames 115-121
   Action 18: (put_on, Knife, CounterTop) -> Frames 122-127
   Action 19: (pick_up, LettuceSliced) -> Frames 128-134
   Action 20: (put_in, LettuceSliced, Bowl) -> Frames 135-141
   Action 21: (pick_up, PotatoSliced) -> Frames 142-148
   Action 22: (put_in, PotatoSliced, Bowl) -> Frames 149-154
   Action 23: (pick_up, TomatoSliced) -> Frames 155-161
   Action 24: (put_in, TomatoSliced, Bowl) -> Frames 162-168
   Action 25: (navigate_to_obj, Fridge) -> Frames 169-175
   Action 26: (open_obj, Fridge) -> Frames 176-182
   Action 27: (navigate_to_obj, Bowl) -> Frames 183-188
   Action 28: (pick_up, Bowl) -> Frames 189-195
   Action 29: (navigate_to_obj, Fridge) -> Frames 196-202
   Action 30: (put_in, Bowl, Fridge) -> Frames 203-209
   Action 31: (close_obj, Fridge) -> Frames 210-216

### CRAFT 流程详细信息

   ✅ 加载了 217 个 events
   ✅ 建立了动作-帧映射: 32 个动作
       ✅ 加载了 217 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=32, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/32: (pick_up, Lettuce)
         检查动作 4/32: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/32: (pick_up, Tomato)
         检查动作 8/32: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/32: (pick_up, Potato)
         检查动作 12/32: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/32: (pick_up, Knife)
         检查动作 19/32: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Knife must be on top of CounterTop
         检查动作 20/32: (pick_up, LettuceSliced)
         检查动作 21/32: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): LettuceSliced must be inside Bowl
         检查动作 22/32: (pick_up, PotatoSliced)
         检查动作 23/32: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): PotatoSliced must be inside Bowl
         检查动作 24/32: (pick_up, TomatoSliced)
         检查动作 25/32: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 25-32 内未满足): TomatoSliced must be inside Bowl
         检查动作 29/32: (pick_up, Bowl)
         检查动作 31/32: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 31-38 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['02:45']
- **Failure Reason**: Failed to successfully execute (pick_up, TomatoSliced)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [43/100] makeSalad/makeSalad-2

⚠️ 加载数据失败: pickle data was truncated
### CRAFT 流程详细信息

       ❌ 加载数据失败: pickle data was truncated

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['02:31']
- **Failure Reason**: Wrong plan - potato slice is put in pan instead of the bowl

### 约束检查日志

  ✅ 所有约束都满足，未发现违反

### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    ⚠️  未找到根失败


================================================================================

## [44/100] makeSalad/makeSalad-3

### 数据加载信息

✅ 加载了 202 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Apple)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-11
   Action 2: (navigate_to_obj, Bowl) -> Frames 12-17
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 18-24
   Action 4: (navigate_to_obj, Apple) -> Frames 25-30
   Action 5: (pick_up, Apple) -> Frames 31-36
   Action 6: (navigate_to_obj, Bowl) -> Frames 37-43
   Action 7: (put_on, Apple, CounterTop) -> Frames 44-49
   Action 8: (navigate_to_obj, Potato) -> Frames 50-55
   Action 9: (pick_up, Potato) -> Frames 56-62
   Action 10: (navigate_to_obj, Bowl) -> Frames 63-68
   Action 11: (put_on, Potato, CounterTop) -> Frames 69-74
   Action 12: (navigate_to_obj, Knife) -> Frames 75-81
   Action 13: (pick_up, Knife) -> Frames 82-87
   Action 14: (navigate_to_obj, Bowl) -> Frames 88-93
   Action 15: (slice_obj, Lettuce) -> Frames 94-100
   Action 16: (slice_obj, Potato) -> Frames 101-106
   Action 17: (slice_obj, Apple) -> Frames 107-112
   Action 18: (put_on, Knife, CounterTop) -> Frames 113-118
   Action 19: (pick_up, LettuceSliced) -> Frames 119-125
   Action 20: (put_in, LettuceSliced, Bowl) -> Frames 126-131
   Action 21: (pick_up, PotatoSliced) -> Frames 132-137
   Action 22: (put_in, PotatoSliced, Bowl) -> Frames 138-144
   Action 23: (pick_up, AppleSliced) -> Frames 145-150
   Action 24: (put_in, AppleSliced, Bowl) -> Frames 151-156
   Action 25: (navigate_to_obj, Fridge) -> Frames 157-163
   Action 26: (open_obj, Fridge) -> Frames 164-169
   Action 27: (navigate_to_obj, Bowl) -> Frames 170-175
   Action 28: (pick_up, Bowl) -> Frames 176-182
   Action 29: (navigate_to_obj, Fridge) -> Frames 183-188
   Action 30: (put_in, Bowl, Fridge) -> Frames 189-194
   Action 31: (close_obj, Fridge) -> Frames 195-201

### CRAFT 流程详细信息

   ✅ 加载了 202 个 events
   ✅ 建立了动作-帧映射: 32 个动作
       ✅ 加载了 202 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Apple)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=32, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/32: (pick_up, Lettuce)
         检查动作 4/32: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/32: (pick_up, Apple)
         检查动作 8/32: (put_on, Apple, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Apple must be on top of CounterTop
         检查动作 10/32: (pick_up, Potato)
         检查动作 12/32: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/32: (pick_up, Knife)
         检查动作 19/32: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Knife must be on top of CounterTop
         检查动作 20/32: (pick_up, LettuceSliced)
         检查动作 21/32: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): LettuceSliced must be inside Bowl
         检查动作 22/32: (pick_up, PotatoSliced)
         检查动作 23/32: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): PotatoSliced must be inside Bowl
         检查动作 24/32: (pick_up, AppleSliced)
         检查动作 25/32: (put_in, AppleSliced, Bowl)
           ❌ Postcondition 违反 (窗口 25-32 内未满足): AppleSliced must be inside Bowl
         检查动作 29/32: (pick_up, Bowl)
         检查动作 31/32: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 31-38 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:46', '01:56', '02:20']
- **Failure Reason**: Wrong plan - apple instead of tomato

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [45/100] makeSalad/makeSalad-4

### 数据加载信息

✅ 加载了 209 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-12
   Action 2: (navigate_to_obj, Bowl) -> Frames 13-18
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 19-25
   Action 4: (navigate_to_obj, Tomato) -> Frames 26-31
   Action 5: (pick_up, Tomato) -> Frames 32-38
   Action 6: (navigate_to_obj, Bowl) -> Frames 39-44
   Action 7: (put_on, Tomato, CounterTop) -> Frames 45-51
   Action 8: (navigate_to_obj, Potato) -> Frames 52-57
   Action 9: (pick_up, Potato) -> Frames 58-64
   Action 10: (navigate_to_obj, Bowl) -> Frames 65-70
   Action 11: (put_on, Potato, CounterTop) -> Frames 71-77
   Action 12: (navigate_to_obj, Knife) -> Frames 78-83
   Action 13: (pick_up, Knife) -> Frames 84-90
   Action 14: (navigate_to_obj, Bowl) -> Frames 91-96
   Action 15: (slice_obj, Lettuce) -> Frames 97-103
   Action 16: (slice_obj, Potato) -> Frames 104-110
   Action 17: (slice_obj, Tomato) -> Frames 111-116
   Action 18: (put_on, Knife, CounterTop) -> Frames 117-123
   Action 19: (pick_up, LettuceSliced) -> Frames 124-129
   Action 20: (put_in, LettuceSliced, Bowl) -> Frames 130-136
   Action 21: (pick_up, TomatoSliced) -> Frames 137-142
   Action 22: (put_in, TomatoSliced, Bowl) -> Frames 143-149
   Action 23: (pick_up, PotatoSliced) -> Frames 150-155
   Action 24: (put_in, PotatoSliced, Bowl) -> Frames 156-162
   Action 25: (navigate_to_obj, Fridge) -> Frames 163-168
   Action 26: (open_obj, Fridge) -> Frames 169-175
   Action 27: (navigate_to_obj, Bowl) -> Frames 176-181
   Action 28: (pick_up, Bowl) -> Frames 182-188
   Action 29: (navigate_to_obj, Fridge) -> Frames 189-194
   Action 30: (put_in, Bowl, Fridge) -> Frames 195-201
   Action 31: (close_obj, Fridge) -> Frames 202-208

### CRAFT 流程详细信息

   ✅ 加载了 209 个 events
   ✅ 建立了动作-帧映射: 32 个动作
       ✅ 加载了 209 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params']
       🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=32, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/32: (pick_up, Lettuce)
         检查动作 4/32: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/32: (pick_up, Tomato)
         检查动作 8/32: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/32: (pick_up, Potato)
         检查动作 12/32: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/32: (pick_up, Knife)
         检查动作 19/32: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Knife must be on top of CounterTop
         检查动作 20/32: (pick_up, LettuceSliced)
         检查动作 21/32: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): LettuceSliced must be inside Bowl
         检查动作 22/32: (pick_up, TomatoSliced)
         检查动作 23/32: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): TomatoSliced must be inside Bowl
         检查动作 24/32: (pick_up, PotatoSliced)
         检查动作 25/32: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 25-32 内未满足): PotatoSliced must be inside Bowl
         检查动作 29/32: (pick_up, Bowl)
         检查动作 31/32: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 31-38 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong perception
- **Failure Step**: ['03:08']
- **Failure Reason**: Wrong perception: pan and bowl

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [46/100] makeSalad/makeSalad-5

### 数据加载信息

✅ 加载了 213 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=31, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-12
   Action 2: (navigate_to_obj, Bowl) -> Frames 13-19
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 20-26
   Action 4: (navigate_to_obj, Tomato) -> Frames 27-33
   Action 5: (pick_up, Tomato) -> Frames 34-40
   Action 6: (navigate_to_obj, Bowl) -> Frames 41-47
   Action 7: (put_on, Tomato, CounterTop) -> Frames 48-53
   Action 8: (navigate_to_obj, Potato) -> Frames 54-60
   Action 9: (pick_up, Potato) -> Frames 61-67
   Action 10: (navigate_to_obj, Bowl) -> Frames 68-74
   Action 11: (put_on, Potato, CounterTop) -> Frames 75-81
   Action 12: (navigate_to_obj, Knife) -> Frames 82-88
   Action 13: (pick_up, Knife) -> Frames 89-95
   Action 14: (navigate_to_obj, Bowl) -> Frames 96-102
   Action 15: (slice_obj, Lettuce) -> Frames 103-108
   Action 16: (slice_obj, Tomato) -> Frames 109-115
   Action 17: (put_on, Knife, CounterTop) -> Frames 116-122
   Action 18: (pick_up, LettuceSliced) -> Frames 123-129
   Action 19: (put_in, LettuceSliced, Bowl) -> Frames 130-136
   Action 20: (pick_up, TomatoSliced) -> Frames 137-143
   Action 21: (put_in, TomatoSliced, Bowl) -> Frames 144-150
   Action 22: (pick_up, Potato) -> Frames 151-157
   Action 23: (put_in, Potato, Bowl) -> Frames 158-163
   Action 24: (navigate_to_obj, Fridge) -> Frames 164-170
   Action 25: (open_obj, Fridge) -> Frames 171-177
   Action 26: (navigate_to_obj, Bowl) -> Frames 178-184
   Action 27: (pick_up, Bowl) -> Frames 185-191
   Action 28: (navigate_to_obj, Fridge) -> Frames 192-198
   Action 29: (put_in, Bowl, Fridge) -> Frames 199-205
   Action 30: (close_obj, Fridge) -> Frames 206-212

### CRAFT 流程详细信息

   ✅ 加载了 213 个 events
   ✅ 建立了动作-帧映射: 31 个动作
       ✅ 加载了 213 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=31, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=31, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/31: (pick_up, Lettuce)
         检查动作 4/31: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/31: (pick_up, Tomato)
         检查动作 8/31: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/31: (pick_up, Potato)
         检查动作 12/31: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/31: (pick_up, Knife)
         检查动作 18/31: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 18-25 内未满足): Knife must be on top of CounterTop
         检查动作 19/31: (pick_up, LettuceSliced)
         检查动作 20/31: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 20-27 内未满足): LettuceSliced must be inside Bowl
         检查动作 21/31: (pick_up, TomatoSliced)
         检查动作 22/31: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 22-29 内未满足): TomatoSliced must be inside Bowl
         检查动作 23/31: (pick_up, Potato)
         检查动作 24/31: (put_in, Potato, Bowl)
           ❌ Postcondition 违反 (窗口 24-31 内未满足): Potato must be inside Bowl
         检查动作 28/31: (pick_up, Bowl)
         检查动作 30/31: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 30-37 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['02:42']
- **Failure Reason**: Missing step - slice potato

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [47/100] makeSalad/makeSalad-6

### 数据加载信息

✅ 加载了 199 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-11
   Action 2: (navigate_to_obj, Bowl) -> Frames 12-17
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 18-23
   Action 4: (navigate_to_obj, Tomato) -> Frames 24-30
   Action 5: (pick_up, Tomato) -> Frames 31-36
   Action 6: (navigate_to_obj, Bowl) -> Frames 37-42
   Action 7: (put_on, Tomato, CounterTop) -> Frames 43-48
   Action 8: (navigate_to_obj, Potato) -> Frames 49-54
   Action 9: (pick_up, Potato) -> Frames 55-61
   Action 10: (navigate_to_obj, Bowl) -> Frames 62-67
   Action 11: (put_on, Potato, CounterTop) -> Frames 68-73
   Action 12: (navigate_to_obj, Knife) -> Frames 74-79
   Action 13: (pick_up, Knife) -> Frames 80-86
   Action 14: (navigate_to_obj, Bowl) -> Frames 87-92
   Action 15: (slice_obj, Lettuce) -> Frames 93-98
   Action 16: (slice_obj, Potato) -> Frames 99-104
   Action 17: (slice_obj, Tomato) -> Frames 105-110
   Action 18: (put_on, Knife, CounterTop) -> Frames 111-117
   Action 19: (pick_up, LettuceSliced) -> Frames 118-123
   Action 20: (put_in, LettuceSliced, Bowl) -> Frames 124-129
   Action 21: (pick_up, PotatoSliced) -> Frames 130-135
   Action 22: (put_in, PotatoSliced, Bowl) -> Frames 136-142
   Action 23: (pick_up, TomatoSliced) -> Frames 143-148
   Action 24: (put_in, TomatoSliced, Bowl) -> Frames 149-154
   Action 25: (navigate_to_obj, Fridge) -> Frames 155-160
   Action 26: (open_obj, Fridge) -> Frames 161-166
   Action 27: (navigate_to_obj, Bowl) -> Frames 167-173
   Action 28: (pick_up, Bowl) -> Frames 174-179
   Action 29: (navigate_to_obj, Fridge) -> Frames 180-185
   Action 30: (put_in, Bowl, Fridge) -> Frames 186-191
   Action 31: (close_obj, Fridge) -> Frames 192-198

### CRAFT 流程详细信息

   ✅ 加载了 199 个 events
   ✅ 建立了动作-帧映射: 32 个动作
       ✅ 加载了 199 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=32, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/32: (pick_up, Lettuce)
         检查动作 4/32: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/32: (pick_up, Tomato)
         检查动作 8/32: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/32: (pick_up, Potato)
         检查动作 12/32: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/32: (pick_up, Knife)
         检查动作 19/32: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Knife must be on top of CounterTop
         检查动作 20/32: (pick_up, LettuceSliced)
         检查动作 21/32: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): LettuceSliced must be inside Bowl
         检查动作 22/32: (pick_up, PotatoSliced)
         检查动作 23/32: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): PotatoSliced must be inside Bowl
         检查动作 24/32: (pick_up, TomatoSliced)
         检查动作 25/32: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 25-32 内未满足): TomatoSliced must be inside Bowl
         检查动作 29/32: (pick_up, Bowl)
         检查动作 31/32: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 31-38 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['180']
- **Failure Reason**: Failed to successfully execute (pick_up, Bowl)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [48/100] makeSalad/makeSalad-7

### 数据加载信息

✅ 加载了 192 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=30, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-11
   Action 2: (navigate_to_obj, Bowl) -> Frames 12-18
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 19-24
   Action 4: (navigate_to_obj, Tomato) -> Frames 25-31
   Action 5: (pick_up, Tomato) -> Frames 32-37
   Action 6: (navigate_to_obj, Bowl) -> Frames 38-43
   Action 7: (put_on, Tomato, CounterTop) -> Frames 44-50
   Action 8: (navigate_to_obj, Potato) -> Frames 51-56
   Action 9: (pick_up, Potato) -> Frames 57-63
   Action 10: (navigate_to_obj, Bowl) -> Frames 64-69
   Action 11: (put_on, Potato, CounterTop) -> Frames 70-75
   Action 12: (navigate_to_obj, Knife) -> Frames 76-82
   Action 13: (pick_up, Knife) -> Frames 83-88
   Action 14: (navigate_to_obj, Bowl) -> Frames 89-95
   Action 15: (slice_obj, Lettuce) -> Frames 96-101
   Action 16: (slice_obj, Potato) -> Frames 102-107
   Action 17: (slice_obj, Tomato) -> Frames 108-114
   Action 18: (put_on, Knife, CounterTop) -> Frames 115-120
   Action 19: (pick_up, PotatoSliced) -> Frames 121-127
   Action 20: (put_in, PotatoSliced, Bowl) -> Frames 128-133
   Action 21: (pick_up, TomatoSliced) -> Frames 134-139
   Action 22: (put_in, TomatoSliced, Bowl) -> Frames 140-146
   Action 23: (navigate_to_obj, Fridge) -> Frames 147-152
   Action 24: (open_obj, Fridge) -> Frames 153-159
   Action 25: (navigate_to_obj, Bowl) -> Frames 160-165
   Action 26: (pick_up, Bowl) -> Frames 166-171
   Action 27: (navigate_to_obj, Fridge) -> Frames 172-178
   Action 28: (put_in, Bowl, Fridge) -> Frames 179-184
   Action 29: (close_obj, Fridge) -> Frames 185-191

### CRAFT 流程详细信息

   ✅ 加载了 192 个 events
   ✅ 建立了动作-帧映射: 30 个动作
       ✅ 加载了 192 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=30, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=30, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 41 个约束
       组织约束...
       ✅ 约束分组: 14 个动作有约束, 0 个目标约束
         检查动作 2/30: (pick_up, Lettuce)
         检查动作 4/30: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/30: (pick_up, Tomato)
         检查动作 8/30: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/30: (pick_up, Potato)
         检查动作 12/30: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/30: (pick_up, Knife)
         检查动作 19/30: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Knife must be on top of CounterTop
         检查动作 20/30: (pick_up, PotatoSliced)
         检查动作 21/30: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): PotatoSliced must be inside Bowl
         检查动作 22/30: (pick_up, TomatoSliced)
         检查动作 23/30: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): TomatoSliced must be inside Bowl
         检查动作 27/30: (pick_up, Bowl)
         检查动作 29/30: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 29-36 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 8 个违反, 8 个真实错误, 31 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['187']
- **Failure Reason**: Missing (pick_up, LettuceSliced), (put_in, LettuceSliced, Bowl)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [49/100] makeSalad/makeSalad-8

### 数据加载信息

✅ 加载了 231 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-6
   Action 1: (pick_up, Lettuce) -> Frames 7-13
   Action 2: (navigate_to_obj, Bowl) -> Frames 14-20
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 21-27
   Action 4: (navigate_to_obj, Tomato) -> Frames 28-35
   Action 5: (pick_up, Tomato) -> Frames 36-42
   Action 6: (navigate_to_obj, Bowl) -> Frames 43-49
   Action 7: (put_on, Tomato, CounterTop) -> Frames 50-56
   Action 8: (navigate_to_obj, Potato) -> Frames 57-63
   Action 9: (pick_up, Potato) -> Frames 64-71
   Action 10: (navigate_to_obj, Bowl) -> Frames 72-78
   Action 11: (put_on, Potato, CounterTop) -> Frames 79-85
   Action 12: (navigate_to_obj, Knife) -> Frames 86-92
   Action 13: (pick_up, Knife) -> Frames 93-100
   Action 14: (navigate_to_obj, Bowl) -> Frames 101-107
   Action 15: (slice_obj, Lettuce) -> Frames 108-114
   Action 16: (slice_obj, Potato) -> Frames 115-121
   Action 17: (slice_obj, Tomato) -> Frames 122-128
   Action 18: (put_on, Knife, CounterTop) -> Frames 129-136
   Action 19: (pick_up, LettuceSliced) -> Frames 137-143
   Action 20: (put_in, LettuceSliced, Bowl) -> Frames 144-150
   Action 21: (pick_up, PotatoSliced) -> Frames 151-157
   Action 22: (put_in, PotatoSliced, Bowl) -> Frames 158-165
   Action 23: (pick_up, TomatoSliced) -> Frames 166-172
   Action 24: (put_in, TomatoSliced, Bowl) -> Frames 173-179
   Action 25: (navigate_to_obj, Fridge) -> Frames 180-186
   Action 26: (open_obj, Fridge) -> Frames 187-193
   Action 27: (navigate_to_obj, Bowl) -> Frames 194-201
   Action 28: (pick_up, Bowl) -> Frames 202-208
   Action 29: (navigate_to_obj, Fridge) -> Frames 209-215
   Action 30: (put_in, Bowl, Fridge) -> Frames 216-222
   Action 31: (close_obj, Fridge) -> Frames 223-230

### CRAFT 流程详细信息

   ✅ 加载了 231 个 events
   ✅ 建立了动作-帧映射: 32 个动作
       ✅ 加载了 231 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=32, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/32: (pick_up, Lettuce)
         检查动作 4/32: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/32: (pick_up, Tomato)
         检查动作 8/32: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/32: (pick_up, Potato)
         检查动作 12/32: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/32: (pick_up, Knife)
         检查动作 19/32: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Knife must be on top of CounterTop
         检查动作 20/32: (pick_up, LettuceSliced)
         检查动作 21/32: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): LettuceSliced must be inside Bowl
         检查动作 22/32: (pick_up, PotatoSliced)
         检查动作 23/32: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): PotatoSliced must be inside Bowl
         检查动作 24/32: (pick_up, TomatoSliced)
         检查动作 25/32: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 25-32 内未满足): TomatoSliced must be inside Bowl
         检查动作 29/32: (pick_up, Bowl)
         检查动作 31/32: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 31-38 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['01:19']
- **Failure Reason**: Failed to successfully execute (put_on, Tomato, CounterTop)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [50/100] makeSalad/makeSalad-9

### 数据加载信息

✅ 加载了 219 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Lettuce) -> Frames 0-5
   Action 1: (pick_up, Lettuce) -> Frames 6-12
   Action 2: (navigate_to_obj, Bowl) -> Frames 13-19
   Action 3: (put_on, Lettuce, CounterTop) -> Frames 20-26
   Action 4: (navigate_to_obj, Tomato) -> Frames 27-33
   Action 5: (pick_up, Tomato) -> Frames 34-40
   Action 6: (navigate_to_obj, Bowl) -> Frames 41-46
   Action 7: (put_on, Tomato, CounterTop) -> Frames 47-53
   Action 8: (navigate_to_obj, Potato) -> Frames 54-60
   Action 9: (pick_up, Potato) -> Frames 61-67
   Action 10: (navigate_to_obj, Bowl) -> Frames 68-74
   Action 11: (put_on, Potato, CounterTop) -> Frames 75-81
   Action 12: (navigate_to_obj, Knife) -> Frames 82-87
   Action 13: (pick_up, Knife) -> Frames 88-94
   Action 14: (navigate_to_obj, Bowl) -> Frames 95-101
   Action 15: (slice_obj, Lettuce) -> Frames 102-108
   Action 16: (slice_obj, Potato) -> Frames 109-115
   Action 17: (slice_obj, Tomato) -> Frames 116-122
   Action 18: (put_on, Knife, CounterTop) -> Frames 123-129
   Action 19: (pick_up, LettuceSliced) -> Frames 130-135
   Action 20: (put_in, LettuceSliced, Bowl) -> Frames 136-142
   Action 21: (pick_up, PotatoSliced) -> Frames 143-149
   Action 22: (put_in, PotatoSliced, Bowl) -> Frames 150-156
   Action 23: (pick_up, TomatoSliced) -> Frames 157-163
   Action 24: (put_in, TomatoSliced, Bowl) -> Frames 164-170
   Action 25: (navigate_to_obj, Fridge) -> Frames 171-176
   Action 26: (open_obj, Fridge) -> Frames 177-183
   Action 27: (navigate_to_obj, Bowl) -> Frames 184-190
   Action 28: (pick_up, Bowl) -> Frames 191-197
   Action 29: (navigate_to_obj, Fridge) -> Frames 198-204
   Action 30: (put_in, Bowl, Fridge) -> Frames 205-211
   Action 31: (close_obj, Fridge) -> Frames 212-218

### CRAFT 流程详细信息

   ✅ 加载了 219 个 events
   ✅ 建立了动作-帧映射: 32 个动作
       ✅ 加载了 219 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'specified_missing_steps']
       🔍 调试：actions 数量=32, actions=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)', '(put_on, Lettuce, CounterTop)', '(navigate_to_obj, Tomato)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=32, 前3个动作=['(navigate_to_obj, Lettuce)', '(pick_up, Lettuce)', '(navigate_to_obj, Bowl)']
       🔍 调试：动作 2 ((pick_up, Lettuce)) 生成了 3 个约束
       ✅ 生成了 48 个约束
       组织约束...
       ✅ 约束分组: 16 个动作有约束, 0 个目标约束
         检查动作 2/32: (pick_up, Lettuce)
         检查动作 4/32: (put_on, Lettuce, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Lettuce must be on top of CounterTop
         检查动作 6/32: (pick_up, Tomato)
         检查动作 8/32: (put_on, Tomato, CounterTop)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): Tomato must be on top of CounterTop
         检查动作 10/32: (pick_up, Potato)
         检查动作 12/32: (put_on, Potato, CounterTop)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Potato must be on top of CounterTop
         检查动作 14/32: (pick_up, Knife)
         检查动作 19/32: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Knife must be on top of CounterTop
         检查动作 20/32: (pick_up, LettuceSliced)
         检查动作 21/32: (put_in, LettuceSliced, Bowl)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): LettuceSliced must be inside Bowl
         检查动作 22/32: (pick_up, PotatoSliced)
         检查动作 23/32: (put_in, PotatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 23-30 内未满足): PotatoSliced must be inside Bowl
         检查动作 24/32: (pick_up, TomatoSliced)
         检查动作 25/32: (put_in, TomatoSliced, Bowl)
           ❌ Postcondition 违反 (窗口 25-32 内未满足): TomatoSliced must be inside Bowl
         检查动作 29/32: (pick_up, Bowl)
         检查动作 31/32: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 31-38 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 9 个违反, 9 个真实错误, 36 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['03:22']
- **Failure Reason**: Dropped Bowl

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 1 object(s): Egg_9b434f5f
      Frame: Unknown frame


================================================================================

## [51/100] storeEgg/storeEgg-1

### 数据加载信息

✅ 加载了 146 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-7
   Action 1: (open_obj, Fridge) -> Frames 8-15
   Action 2: (pick_up, Egg) -> Frames 16-23
   Action 3: (navigate_to_obj, CounterTop) -> Frames 24-31
   Action 4: (put_on, Egg, CounterTop) -> Frames 32-39
   Action 5: (navigate_to_obj, Fridge) -> Frames 40-47
   Action 6: (close_obj, Fridge) -> Frames 48-55
   Action 7: (navigate_to_obj, Egg) -> Frames 56-63
   Action 8: (pick_up, Egg) -> Frames 64-72
   Action 9: (navigate_to_obj, Bowl) -> Frames 73-80
   Action 10: (put_in, Egg, Bowl) -> Frames 81-88
   Action 11: (navigate_to_obj, Fridge) -> Frames 89-96
   Action 12: (open_obj, Fridge) -> Frames 97-104
   Action 13: (navigate_to_obj, Bowl) -> Frames 105-112
   Action 14: (pick_up, Pan) -> Frames 113-120
   Action 15: (navigate_to_obj, Fridge) -> Frames 121-128
   Action 16: (put_in, Pan, Fridge) -> Frames 129-136
   Action 17: (close_obj, Fridge) -> Frames 137-145

### CRAFT 流程详细信息

   ✅ 加载了 146 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 146 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/18: (pick_up, Egg)
         检查动作 5/18: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Egg must be on top of CounterTop
         检查动作 9/18: (pick_up, Egg)
         检查动作 11/18: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Egg must be inside Bowl
         检查动作 15/18: (pick_up, Pan)
         检查动作 17/18: (put_in, Pan, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Pan must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['02:07', '02:21']
- **Failure Reason**: Wrong plan - robot puts pan instead of bowl in the fridge

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [52/100] storeEgg/storeEgg-10

### 数据加载信息

✅ 加载了 125 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=17, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)', '(navigate_to_obj, Fridge)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-6
   Action 1: (open_obj, Fridge) -> Frames 7-13
   Action 2: (navigate_to_obj, CounterTop) -> Frames 14-21
   Action 3: (put_on, Egg, CounterTop) -> Frames 22-28
   Action 4: (navigate_to_obj, Fridge) -> Frames 29-35
   Action 5: (close_obj, Fridge) -> Frames 36-43
   Action 6: (navigate_to_obj, Egg) -> Frames 44-50
   Action 7: (pick_up, Egg) -> Frames 51-57
   Action 8: (navigate_to_obj, Bowl) -> Frames 58-65
   Action 9: (put_in, Egg, Bowl) -> Frames 66-72
   Action 10: (navigate_to_obj, Fridge) -> Frames 73-79
   Action 11: (open_obj, Fridge) -> Frames 80-87
   Action 12: (navigate_to_obj, Bowl) -> Frames 88-94
   Action 13: (pick_up, Bowl) -> Frames 95-101
   Action 14: (navigate_to_obj, Fridge) -> Frames 102-109
   Action 15: (put_in, Bowl, Fridge) -> Frames 110-116
   Action 16: (close_obj, Fridge) -> Frames 117-124

### CRAFT 流程详细信息

   ✅ 加载了 125 个 events
   ✅ 建立了动作-帧映射: 17 个动作
       ✅ 加载了 125 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=17, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)', '(navigate_to_obj, Fridge)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=17, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(navigate_to_obj, CounterTop)']
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 4/17: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Egg must be on top of CounterTop
         检查动作 8/17: (pick_up, Egg)
         检查动作 10/17: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): Egg must be inside Bowl
         检查动作 14/17: (pick_up, Bowl)
         检查动作 16/17: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 16-23 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 11 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:35']
- **Failure Reason**: Missing (pick_up, Egg)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [53/100] storeEgg/storeEgg-2

### 数据加载信息

✅ 加载了 126 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-6
   Action 1: (open_obj, Fridge) -> Frames 7-13
   Action 2: (pick_up, Egg) -> Frames 14-20
   Action 3: (navigate_to_obj, CounterTop) -> Frames 21-27
   Action 4: (put_on, Egg, CounterTop) -> Frames 28-34
   Action 5: (navigate_to_obj, Fridge) -> Frames 35-41
   Action 6: (close_obj, Fridge) -> Frames 42-48
   Action 7: (navigate_to_obj, Egg) -> Frames 49-55
   Action 8: (pick_up, Egg) -> Frames 56-62
   Action 9: (navigate_to_obj, Bowl) -> Frames 63-69
   Action 10: (put_in, Egg, Bowl) -> Frames 70-76
   Action 11: (navigate_to_obj, Fridge) -> Frames 77-83
   Action 12: (open_obj, Fridge) -> Frames 84-90
   Action 13: (navigate_to_obj, Bowl) -> Frames 91-97
   Action 14: (pick_up, Bowl) -> Frames 98-104
   Action 15: (navigate_to_obj, Fridge) -> Frames 105-111
   Action 16: (put_in, Bowl, Fridge) -> Frames 112-118
   Action 17: (close_obj, Fridge) -> Frames 119-125

### CRAFT 流程详细信息

   ✅ 加载了 126 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 126 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/18: (pick_up, Egg)
         检查动作 5/18: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Egg must be on top of CounterTop
         检查动作 9/18: (pick_up, Egg)
         检查动作 11/18: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Egg must be inside Bowl
         检查动作 15/18: (pick_up, Bowl)
         检查动作 17/18: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: blocking
- **Failure Step**: ['00:16']
- **Failure Reason**: lettuce is blocking the egg

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [54/100] storeEgg/storeEgg-3

⚠️ 加载数据失败: pickle data was truncated
### CRAFT 流程详细信息

       ❌ 加载数据失败: pickle data was truncated

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['01:42', '02:23']
- **Failure Reason**: Wrong plan - robot puts egg in the pan instead of bowl

### 约束检查日志

  ✅ 所有约束都满足，未发现违反

### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    ⚠️  未找到根失败


================================================================================

## [55/100] storeEgg/storeEgg-4

### 数据加载信息

✅ 加载了 148 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-7
   Action 1: (open_obj, Fridge) -> Frames 8-15
   Action 2: (pick_up, Egg) -> Frames 16-23
   Action 3: (navigate_to_obj, CounterTop) -> Frames 24-31
   Action 4: (put_on, Egg, CounterTop) -> Frames 32-40
   Action 5: (navigate_to_obj, Fridge) -> Frames 41-48
   Action 6: (close_obj, Fridge) -> Frames 49-56
   Action 7: (navigate_to_obj, Egg) -> Frames 57-64
   Action 8: (pick_up, Egg) -> Frames 65-73
   Action 9: (navigate_to_obj, Bowl) -> Frames 74-81
   Action 10: (put_in, Egg, Container) -> Frames 82-89
   Action 11: (navigate_to_obj, Fridge) -> Frames 90-97
   Action 12: (open_obj, Fridge) -> Frames 98-105
   Action 13: (navigate_to_obj, Bowl) -> Frames 106-114
   Action 14: (pick_up, Bowl) -> Frames 115-122
   Action 15: (navigate_to_obj, Fridge) -> Frames 123-130
   Action 16: (put_in, Bowl, Fridge) -> Frames 131-138
   Action 17: (close_obj, Fridge) -> Frames 139-147

### CRAFT 流程详细信息

   ✅ 加载了 148 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 148 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/18: (pick_up, Egg)
         检查动作 5/18: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Egg must be on top of CounterTop
         检查动作 9/18: (pick_up, Egg)
         检查动作 11/18: (put_in, Egg, Container)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Egg must be inside Container
         检查动作 15/18: (pick_up, Bowl)
         检查动作 17/18: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: ambiguous_plan
- **Failure Step**: ['01:41']
- **Failure Reason**: Ambiguous plan - says some container (maps to pan)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [56/100] storeEgg/storeEgg-5

### 数据加载信息

✅ 加载了 148 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-7
   Action 1: (open_obj, Fridge) -> Frames 8-15
   Action 2: (pick_up, Egg) -> Frames 16-23
   Action 3: (navigate_to_obj, CounterTop) -> Frames 24-31
   Action 4: (put_on, Egg, CounterTop) -> Frames 32-40
   Action 5: (navigate_to_obj, Fridge) -> Frames 41-48
   Action 6: (close_obj, Fridge) -> Frames 49-56
   Action 7: (navigate_to_obj, Egg) -> Frames 57-64
   Action 8: (pick_up, Egg) -> Frames 65-73
   Action 9: (navigate_to_obj, Bowl) -> Frames 74-81
   Action 10: (put_in, Egg, Bowl) -> Frames 82-89
   Action 11: (navigate_to_obj, Fridge) -> Frames 90-97
   Action 12: (open_obj, Fridge) -> Frames 98-105
   Action 13: (navigate_to_obj, Bowl) -> Frames 106-114
   Action 14: (pick_up, Bowl) -> Frames 115-122
   Action 15: (navigate_to_obj, Fridge) -> Frames 123-130
   Action 16: (put_in, Bowl, Fridge) -> Frames 131-138
   Action 17: (close_obj, Fridge) -> Frames 139-147

### CRAFT 流程详细信息

   ✅ 加载了 148 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 148 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/18: (pick_up, Egg)
         检查动作 5/18: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Egg must be on top of CounterTop
         检查动作 9/18: (pick_up, Egg)
         检查动作 11/18: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Egg must be inside Bowl
         检查动作 15/18: (pick_up, Bowl)
         检查动作 17/18: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: execution_failure
- **Failure Step**: ['01:41']
- **Failure Reason**: Wrong execution - policy puts egg in pan instead of bowl

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [57/100] storeEgg/storeEgg-6

### 数据加载信息

✅ 加载了 148 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-7
   Action 1: (open_obj, Fridge) -> Frames 8-15
   Action 2: (pick_up, Egg) -> Frames 16-23
   Action 3: (navigate_to_obj, CounterTop) -> Frames 24-31
   Action 4: (put_on, Egg, CounterTop) -> Frames 32-40
   Action 5: (navigate_to_obj, Fridge) -> Frames 41-48
   Action 6: (close_obj, Fridge) -> Frames 49-56
   Action 7: (navigate_to_obj, Egg) -> Frames 57-64
   Action 8: (pick_up, Egg) -> Frames 65-73
   Action 9: (navigate_to_obj, Bowl) -> Frames 74-81
   Action 10: (put_in, Egg, Bowl) -> Frames 82-89
   Action 11: (navigate_to_obj, Fridge) -> Frames 90-97
   Action 12: (open_obj, Fridge) -> Frames 98-105
   Action 13: (navigate_to_obj, Bowl) -> Frames 106-114
   Action 14: (pick_up, Bowl) -> Frames 115-122
   Action 15: (navigate_to_obj, Fridge) -> Frames 123-130
   Action 16: (put_in, Bowl, Fridge) -> Frames 131-138
   Action 17: (close_obj, Fridge) -> Frames 139-147

### CRAFT 流程详细信息

   ✅ 加载了 148 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 148 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/18: (pick_up, Egg)
         检查动作 5/18: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Egg must be on top of CounterTop
         检查动作 9/18: (pick_up, Egg)
         检查动作 11/18: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Egg must be inside Bowl
         检查动作 15/18: (pick_up, Bowl)
         检查动作 17/18: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong perception
- **Failure Step**: ['01:27']
- **Failure Reason**: Wrong perception - potato detected as egg

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [58/100] storeEgg/storeEgg-7

### 数据加载信息

✅ 加载了 74 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(pick_up, Egg)', '(navigate_to_obj, Bowl)', '(put_in, Egg, Bowl)', '(navigate_to_obj, Fridge)', '(open_obj, Fridge)']

### 动作-帧映射

   Action 0: (pick_up, Egg) -> Frames 0-6
   Action 1: (navigate_to_obj, Bowl) -> Frames 7-13
   Action 2: (put_in, Egg, Bowl) -> Frames 14-21
   Action 3: (navigate_to_obj, Fridge) -> Frames 22-28
   Action 4: (open_obj, Fridge) -> Frames 29-36
   Action 5: (navigate_to_obj, Bowl) -> Frames 37-43
   Action 6: (pick_up, Bowl) -> Frames 44-50
   Action 7: (navigate_to_obj, Fridge) -> Frames 51-58
   Action 8: (put_in, Bowl, Fridge) -> Frames 59-65
   Action 9: (close_obj, Fridge) -> Frames 66-73

### CRAFT 流程详细信息

   ✅ 加载了 74 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 74 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=10, actions=['(pick_up, Egg)', '(navigate_to_obj, Bowl)', '(put_in, Egg, Bowl)', '(navigate_to_obj, Fridge)', '(open_obj, Fridge)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(pick_up, Egg)', '(navigate_to_obj, Bowl)', '(put_in, Egg, Bowl)']
       🔍 调试：动作 1 ((pick_up, Egg)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 4 个动作有约束, 0 个目标约束
         检查动作 1/10: (pick_up, Egg)
         检查动作 3/10: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 3-10 内未满足): Egg must be inside Bowl
         检查动作 7/10: (pick_up, Bowl)
         检查动作 9/10: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:15']
- **Failure Reason**: Missing step of open fridge before pick up the egg

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [59/100] storeEgg/storeEgg-8

### 数据加载信息

✅ 加载了 160 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-7
   Action 1: (open_obj, Fridge) -> Frames 8-16
   Action 2: (pick_up, Egg) -> Frames 17-25
   Action 3: (navigate_to_obj, CounterTop) -> Frames 26-34
   Action 4: (put_on, Egg, CounterTop) -> Frames 35-43
   Action 5: (navigate_to_obj, Fridge) -> Frames 44-52
   Action 6: (close_obj, Fridge) -> Frames 53-61
   Action 7: (navigate_to_obj, Egg) -> Frames 62-70
   Action 8: (pick_up, Egg) -> Frames 71-79
   Action 9: (navigate_to_obj, Bowl) -> Frames 80-87
   Action 10: (put_in, Egg, Bowl) -> Frames 88-96
   Action 11: (navigate_to_obj, Fridge) -> Frames 97-105
   Action 12: (open_obj, Fridge) -> Frames 106-114
   Action 13: (navigate_to_obj, Bowl) -> Frames 115-123
   Action 14: (pick_up, Bowl) -> Frames 124-132
   Action 15: (navigate_to_obj, Fridge) -> Frames 133-141
   Action 16: (put_in, Bowl, Fridge) -> Frames 142-150
   Action 17: (close_obj, Fridge) -> Frames 151-159

### CRAFT 流程详细信息

   ✅ 加载了 160 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 160 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/18: (pick_up, Egg)
         检查动作 5/18: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Egg must be on top of CounterTop
         检查动作 9/18: (pick_up, Egg)
         检查动作 11/18: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Egg must be inside Bowl
         检查动作 15/18: (pick_up, Bowl)
         检查动作 17/18: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['02:11']
- **Failure Reason**: Dropped Bowl

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [60/100] storeEgg/storeEgg-9

### 数据加载信息

✅ 加载了 145 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Fridge) -> Frames 0-7
   Action 1: (open_obj, Fridge) -> Frames 8-15
   Action 2: (pick_up, Egg) -> Frames 16-23
   Action 3: (navigate_to_obj, CounterTop) -> Frames 24-31
   Action 4: (put_on, Egg, CounterTop) -> Frames 32-39
   Action 5: (navigate_to_obj, Fridge) -> Frames 40-47
   Action 6: (close_obj, Fridge) -> Frames 48-55
   Action 7: (navigate_to_obj, Egg) -> Frames 56-63
   Action 8: (pick_up, Egg) -> Frames 64-71
   Action 9: (navigate_to_obj, Bowl) -> Frames 72-79
   Action 10: (put_in, Egg, Bowl) -> Frames 80-87
   Action 11: (navigate_to_obj, Fridge) -> Frames 88-95
   Action 12: (open_obj, Fridge) -> Frames 96-103
   Action 13: (navigate_to_obj, Bowl) -> Frames 104-111
   Action 14: (pick_up, Bowl) -> Frames 112-119
   Action 15: (navigate_to_obj, Fridge) -> Frames 120-127
   Action 16: (put_in, Bowl, Fridge) -> Frames 128-135
   Action 17: (close_obj, Fridge) -> Frames 136-144

### CRAFT 流程详细信息

   ✅ 加载了 145 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 145 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)', '(navigate_to_obj, CounterTop)', '(put_on, Egg, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Fridge)', '(open_obj, Fridge)', '(pick_up, Egg)']
       ✅ 生成了 19 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 3/18: (pick_up, Egg)
         检查动作 5/18: (put_on, Egg, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Egg must be on top of CounterTop
         检查动作 9/18: (pick_up, Egg)
         检查动作 11/18: (put_in, Egg, Bowl)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): Egg must be inside Bowl
         检查动作 15/18: (pick_up, Bowl)
         检查动作 17/18: (put_in, Bowl, Fridge)
           ❌ Precondition 违反: Fridge must be empty
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Bowl must be inside Fridge
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:15']
- **Failure Reason**: Failed to successfully execute (open_obj, Fridge)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Fridge must be empty
      Reason: Container 'fridge' contains 2 object(s): Egg_2744cca4, Lettuce_c532b8b7
      Frame: Unknown frame


================================================================================

## [61/100] switchDevices/switchDevices-1

### 数据加载信息

✅ 加载了 74 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(toggle_on, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-6
   Action 1: (toggle_on, Laptop) -> Frames 7-13
   Action 2: (pick_up, Laptop) -> Frames 14-21
   Action 3: (navigate_to_obj, TVStand) -> Frames 22-28
   Action 4: (put_on, Laptop, TVStand) -> Frames 29-36
   Action 5: (navigate_to_obj, RemoteControl) -> Frames 37-43
   Action 6: (pick_up, RemoteControl) -> Frames 44-50
   Action 7: (navigate_to_obj, Television) -> Frames 51-58
   Action 8: (toggle_off, Television) -> Frames 59-65
   Action 9: (put_on, RemoteControl, TVStand) -> Frames 66-73

### CRAFT 流程详细信息

   ✅ 加载了 74 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 74 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(toggle_on, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Laptop)', '(toggle_on, Laptop)', '(pick_up, Laptop)']
       🔍 调试：动作 2 ((toggle_on, Laptop)) 生成了 2 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/10: (toggle_on, Laptop)
           ❌ Postcondition 违反 (窗口 2-6 内未满足): Laptop must be toggled on
         检查动作 3/10: (pick_up, Laptop)
         检查动作 5/10: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/10: (pick_up, RemoteControl)
         检查动作 9/10: (toggle_off, Television)
           ✅ Postcondition 满足 (在 帧 9 (窗口 9-13) 满足): Television must be toggled off
         检查动作 10/10: (put_on, RemoteControl, TVStand)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): RemoteControl must be on top of TVStand
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:19', '01:07']
- **Failure Reason**: Wrong Plan: TV switched off and laptop turned on

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: RemoteControl must be on top of TVStand
      Reason: Postcondition not satisfied in temporal window [10-17]. Last reason: remotecontrol is not on top of tvstand
      Frame: Unknown frame


================================================================================

## [62/100] switchDevices/switchDevices-10

### 数据加载信息

✅ 加载了 66 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-7
   Action 1: (close_obj, Laptop) -> Frames 8-15
   Action 2: (pick_up, Laptop) -> Frames 16-23
   Action 3: (navigate_to_obj, TVStand) -> Frames 24-32
   Action 4: (put_on, Laptop, TVStand) -> Frames 33-40
   Action 5: (navigate_to_obj, RemoteControl) -> Frames 41-48
   Action 6: (navigate_to_obj, Television) -> Frames 49-56
   Action 7: (toggle_on, Television) -> Frames 57-65

### CRAFT 流程详细信息

   ✅ 加载了 66 个 events
   ✅ 建立了动作-帧映射: 8 个动作
       ✅ 加载了 66 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason']
       🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=8, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 7 个约束
       组织约束...
       ✅ 约束分组: 3 个动作有约束, 0 个目标约束
         检查动作 3/8: (pick_up, Laptop)
         检查动作 5/8: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 8/8: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Television must be toggled on
       ✅ 检测完成: 2 个违反, 2 个真实错误, 5 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['01:04']
- **Failure Reason**: Missing (pick_up, RemoteControl)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Television must be toggled on
      Reason: Postcondition not satisfied in temporal window [8-12]. Last reason: television is not toggled on
      Frame: Unknown frame


================================================================================

## [63/100] switchDevices/switchDevices-2

### 数据加载信息

✅ 加载了 67 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'failure_injection_params', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-5
   Action 1: (close_obj, Laptop) -> Frames 6-12
   Action 2: (pick_up, Laptop) -> Frames 13-19
   Action 3: (navigate_to_obj, TVStand) -> Frames 20-25
   Action 4: (put_on, Laptop, TVStand) -> Frames 26-32
   Action 5: (navigate_to_obj, Television) -> Frames 33-39
   Action 6: (toggle_on, Television) -> Frames 40-45
   Action 7: (navigate_to_obj, RemoteControl) -> Frames 46-52
   Action 8: (pick_up, RemoteControl) -> Frames 53-59
   Action 9: (put_on, RemoteControl, TVStand) -> Frames 60-66

### CRAFT 流程详细信息

   ✅ 加载了 67 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 67 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'failure_injection_params', 'gt_failure_reason', 'gt_failure_step']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 12 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 3/10: (pick_up, Laptop)
         检查动作 5/10: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/10: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 7-11 内未满足): Television must be toggled on
         检查动作 9/10: (pick_up, RemoteControl)
         检查动作 10/10: (put_on, RemoteControl, TVStand)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): RemoteControl must be on top of TVStand
       ✅ 检测完成: 3 个违反, 3 个真实错误, 9 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong_plan
- **Failure Step**: ['00:55']
- **Failure Reason**: Wrong Order: of pick up remote control and toggle on television

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: RemoteControl must be on top of TVStand
      Reason: Postcondition not satisfied in temporal window [10-17]. Last reason: remotecontrol is not on top of tvstand
      Frame: Unknown frame


================================================================================

## [64/100] switchDevices/switchDevices-3

### 数据加载信息

✅ 加载了 77 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-6
   Action 1: (close_obj, Laptop) -> Frames 7-13
   Action 2: (pick_up, Laptop) -> Frames 14-20
   Action 3: (navigate_to_obj, TVStand) -> Frames 21-27
   Action 4: (put_on, Laptop, TVStand) -> Frames 28-34
   Action 5: (open_obj, Laptop) -> Frames 35-41
   Action 6: (navigate_to_obj, RemoteControl) -> Frames 42-48
   Action 7: (pick_up, RemoteControl) -> Frames 49-55
   Action 8: (navigate_to_obj, Television) -> Frames 56-62
   Action 9: (toggle_on, Television) -> Frames 63-69
   Action 10: (put_on, RemoteControl, TVStand) -> Frames 70-76

### CRAFT 流程详细信息

   ✅ 加载了 77 个 events
   ✅ 建立了动作-帧映射: 11 个动作
       ✅ 加载了 77 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=11, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=11, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 12 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 3/11: (pick_up, Laptop)
         检查动作 5/11: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 8/11: (pick_up, RemoteControl)
         检查动作 10/11: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): Television must be toggled on
         检查动作 11/11: (put_on, RemoteControl, TVStand)
           ❌ Postcondition 违反 (窗口 11-18 内未满足): RemoteControl must be on top of TVStand
       ✅ 检测完成: 3 个违反, 3 个真实错误, 9 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:47']
- **Failure Reason**: Wrong Plan: Opens the laptop again

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: RemoteControl must be on top of TVStand
      Reason: Postcondition not satisfied in temporal window [11-18]. Last reason: remotecontrol is not on top of tvstand
      Frame: Unknown frame


================================================================================

## [65/100] switchDevices/switchDevices-4

### 数据加载信息

✅ 加载了 101 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-10
   Action 1: (close_obj, Laptop) -> Frames 11-21
   Action 2: (pick_up, Laptop) -> Frames 22-32
   Action 3: (navigate_to_obj, TVStand) -> Frames 33-43
   Action 4: (put_on, Laptop, TVStand) -> Frames 44-55
   Action 5: (navigate_to_obj, RemoteControl) -> Frames 56-66
   Action 6: (pick_up, RemoteControl) -> Frames 67-77
   Action 7: (navigate_to_obj, FloorLamp) -> Frames 78-88
   Action 8: (toggle_on, FloorLamp) -> Frames 89-100

### CRAFT 流程详细信息

   ✅ 加载了 101 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 101 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 10 个约束
       组织约束...
       ✅ 约束分组: 4 个动作有约束, 0 个目标约束
         检查动作 3/9: (pick_up, Laptop)
         检查动作 5/9: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/9: (pick_up, RemoteControl)
         检查动作 9/9: (toggle_on, FloorLamp)
           ✅ Postcondition 满足 (在 帧 9 (窗口 9-13) 满足): FloorLamp must be toggled on
       ✅ 检测完成: 1 个违反, 1 个真实错误, 8 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['01:39']
- **Failure Reason**: Wrong Plan: Floorlamp turned on instead of television

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Laptop must be on top of TVStand
      Reason: Postcondition not satisfied in temporal window [5-12]. Last reason: laptop is not on top of tvstand
      Frame: Unknown frame


================================================================================

## [66/100] switchDevices/switchDevices-5

### 数据加载信息

✅ 加载了 80 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, GarbageCan)', '(put_in, Laptop, GarbageCan)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-7
   Action 1: (close_obj, Laptop) -> Frames 8-15
   Action 2: (pick_up, Laptop) -> Frames 16-23
   Action 3: (navigate_to_obj, GarbageCan) -> Frames 24-31
   Action 4: (put_in, Laptop, GarbageCan) -> Frames 32-39
   Action 5: (navigate_to_obj, RemoteControl) -> Frames 40-47
   Action 6: (pick_up, RemoteControl) -> Frames 48-55
   Action 7: (navigate_to_obj, Television) -> Frames 56-63
   Action 8: (toggle_on, Television) -> Frames 64-71
   Action 9: (put_on, RemoteControl, TVStand) -> Frames 72-79

### CRAFT 流程详细信息

   ✅ 加载了 80 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 80 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, GarbageCan)', '(put_in, Laptop, GarbageCan)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 3/10: (pick_up, Laptop)
         检查动作 5/10: (put_in, Laptop, GarbageCan)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be inside GarbageCan
         检查动作 7/10: (pick_up, RemoteControl)
         检查动作 9/10: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Television must be toggled on
         检查动作 10/10: (put_on, RemoteControl, TVStand)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): RemoteControl must be on top of TVStand
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:48']
- **Failure Reason**: Wrong Plan: Laptop is put in garbage can instead of TV stand

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: RemoteControl must be on top of TVStand
      Reason: Postcondition not satisfied in temporal window [10-17]. Last reason: remotecontrol is not on top of tvstand
      Frame: Unknown frame


================================================================================

## [67/100] switchDevices/switchDevices-6

### 数据加载信息

✅ 加载了 66 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_in, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-5
   Action 1: (close_obj, Laptop) -> Frames 6-12
   Action 2: (pick_up, Laptop) -> Frames 13-18
   Action 3: (navigate_to_obj, TVStand) -> Frames 19-25
   Action 4: (put_in, Laptop, TVStand) -> Frames 26-32
   Action 5: (navigate_to_obj, RemoteControl) -> Frames 33-38
   Action 6: (pick_up, RemoteControl) -> Frames 39-45
   Action 7: (navigate_to_obj, Television) -> Frames 46-51
   Action 8: (toggle_on, Television) -> Frames 52-58
   Action 9: (put_on, RemoteControl, TVStand) -> Frames 59-65

### CRAFT 流程详细信息

   ✅ 加载了 66 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 66 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_in, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=10, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 3/10: (pick_up, Laptop)
         检查动作 5/10: (put_in, Laptop, TVStand)
           ❌ Precondition 违反: TVStand must be empty
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be inside TVStand
         检查动作 7/10: (pick_up, RemoteControl)
         检查动作 9/10: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Television must be toggled on
         检查动作 10/10: (put_on, RemoteControl, TVStand)
           ❌ Postcondition 违反 (窗口 10-17 内未满足): RemoteControl must be on top of TVStand
       ✅ 检测完成: 4 个违反, 4 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: blocking
- **Failure Step**: ['00:51']
- **Failure Reason**: Book is blocking remote control

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: TVStand must be empty
      Reason: Container 'tvstand' contains 3 object(s): KeyChain_82853271, Television_3896d693, Watch_d584f93f
      Frame: Unknown frame

    Derived Violations (派生失败, 2 个):
      这些失败是由根失败导致的级联失败，不单独分析


================================================================================

## [68/100] switchDevices/switchDevices-7

### 数据加载信息

✅ 加载了 71 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)', '(navigate_to_obj, RemoteControl)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-6
   Action 1: (pick_up, Laptop) -> Frames 7-14
   Action 2: (navigate_to_obj, TVStand) -> Frames 15-22
   Action 3: (put_on, Laptop, TVStand) -> Frames 23-30
   Action 4: (navigate_to_obj, RemoteControl) -> Frames 31-38
   Action 5: (pick_up, RemoteControl) -> Frames 39-46
   Action 6: (navigate_to_obj, Television) -> Frames 47-54
   Action 7: (toggle_on, Television) -> Frames 55-62
   Action 8: (put_on, RemoteControl, TVStand) -> Frames 63-70

### CRAFT 流程详细信息

   ✅ 加载了 71 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 71 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)', '(navigate_to_obj, RemoteControl)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)']
       🔍 调试：动作 2 ((pick_up, Laptop)) 生成了 3 个约束
       ✅ 生成了 12 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Laptop)
         检查动作 4/9: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Laptop must be on top of TVStand
         检查动作 6/9: (pick_up, RemoteControl)
         检查动作 8/9: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Television must be toggled on
         检查动作 9/9: (put_on, RemoteControl, TVStand)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): RemoteControl must be on top of TVStand
       ✅ 检测完成: 3 个违反, 3 个真实错误, 9 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:41']
- **Failure Reason**: Miss step of close laptop

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: RemoteControl must be on top of TVStand
      Reason: Postcondition not satisfied in temporal window [9-16]. Last reason: remotecontrol is not on top of tvstand
      Frame: Unknown frame


================================================================================

## [69/100] switchDevices/switchDevices-8

### 数据加载信息

✅ 加载了 69 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-6
   Action 1: (close_obj, Laptop) -> Frames 7-14
   Action 2: (pick_up, Laptop) -> Frames 15-22
   Action 3: (navigate_to_obj, TVStand) -> Frames 23-29
   Action 4: (put_on, Laptop, TVStand) -> Frames 30-37
   Action 5: (navigate_to_obj, RemoteControl) -> Frames 38-45
   Action 6: (pick_up, RemoteControl) -> Frames 46-52
   Action 7: (navigate_to_obj, Television) -> Frames 53-60
   Action 8: (toggle_on, Television) -> Frames 61-68

### CRAFT 流程详细信息

   ✅ 加载了 69 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 69 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 10 个约束
       组织约束...
       ✅ 约束分组: 4 个动作有约束, 0 个目标约束
         检查动作 3/9: (pick_up, Laptop)
         检查动作 5/9: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/9: (pick_up, RemoteControl)
         检查动作 9/9: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Television must be toggled on
       ✅ 检测完成: 2 个违反, 2 个真实错误, 8 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:59']
- **Failure Reason**: Dropped RemoteControl

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Television must be toggled on
      Reason: Postcondition not satisfied in temporal window [9-13]. Last reason: television is not toggled on
      Frame: Unknown frame


================================================================================

## [70/100] switchDevices/switchDevices-9

### 数据加载信息

✅ 加载了 68 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Laptop) -> Frames 0-6
   Action 1: (close_obj, Laptop) -> Frames 7-14
   Action 2: (pick_up, Laptop) -> Frames 15-21
   Action 3: (navigate_to_obj, TVStand) -> Frames 22-29
   Action 4: (put_on, Laptop, TVStand) -> Frames 30-36
   Action 5: (navigate_to_obj, RemoteControl) -> Frames 37-44
   Action 6: (pick_up, RemoteControl) -> Frames 45-51
   Action 7: (navigate_to_obj, Television) -> Frames 52-59
   Action 8: (toggle_on, Television) -> Frames 60-67

### CRAFT 流程详细信息

   ✅ 加载了 68 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 68 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'specified_missing_steps', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)', '(navigate_to_obj, TVStand)', '(put_on, Laptop, TVStand)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Laptop)', '(close_obj, Laptop)', '(pick_up, Laptop)']
       ✅ 生成了 10 个约束
       组织约束...
       ✅ 约束分组: 4 个动作有约束, 0 个目标约束
         检查动作 3/9: (pick_up, Laptop)
         检查动作 5/9: (put_on, Laptop, TVStand)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/9: (pick_up, RemoteControl)
         检查动作 9/9: (toggle_on, Television)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Television must be toggled on
       ✅ 检测完成: 2 个违反, 2 个真实错误, 8 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:40']
- **Failure Reason**: Failed to successfully execute (put_on, Laptop, TVStand)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Television must be toggled on
      Reason: Postcondition not satisfied in temporal window [9-13]. Last reason: television is not toggled on
      Frame: Unknown frame


================================================================================

## [71/100] toastBread/toastBread-1

### 数据加载信息

✅ 加载了 50 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-4
   Action 1: (pick_up, Knife) -> Frames 5-10
   Action 2: (navigate_to_obj, Bread) -> Frames 11-15
   Action 3: (slice_obj, Bread) -> Frames 16-21
   Action 4: (put_on, Knife, CounterTop) -> Frames 22-26
   Action 5: (pick_up, BreadSliced) -> Frames 27-32
   Action 6: (navigate_to_obj, Toaster) -> Frames 33-37
   Action 7: (put_in, BreadSliced, Toaster) -> Frames 38-43
   Action 8: (toggle_on, Toaster) -> Frames 44-49

### CRAFT 流程详细信息

   ✅ 加载了 50 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 50 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:43
- **Failure Reason**: Dropped BreadSliced

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [9-13]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [72/100] toastBread/toastBread-10

### 数据加载信息

✅ 加载了 47 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-4
   Action 1: (pick_up, Knife) -> Frames 5-10
   Action 2: (navigate_to_obj, Bread) -> Frames 11-16
   Action 3: (slice_obj, Bread) -> Frames 17-22
   Action 4: (put_on, Knife, CounterTop) -> Frames 23-28
   Action 5: (navigate_to_obj, Toaster) -> Frames 29-34
   Action 6: (put_in, BreadSliced, Toaster) -> Frames 35-40
   Action 7: (toggle_on, Toaster) -> Frames 41-46

### CRAFT 流程详细信息

   ✅ 加载了 47 个 events
   ✅ 建立了动作-帧映射: 8 个动作
       ✅ 加载了 47 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=8, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 11 个约束
       组织约束...
       ✅ 约束分组: 4 个动作有约束, 0 个目标约束
         检查动作 2/8: (pick_up, Knife)
         检查动作 5/8: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 7/8: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 7-14 内未满足): BreadSliced must be inside Toaster
         检查动作 8/8: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Toaster must be toggled on
       ✅ 检测完成: 3 个违反, 3 个真实错误, 7 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:36', '00:37', '00:38', '00:39', '00:40', '00:41', '00:42']
- **Failure Reason**: The robot forgot to pick up the bread slice from the countertop before moving to the toaster.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [8-12]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [73/100] toastBread/toastBread-2

### 数据加载信息

✅ 加载了 48 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-4
   Action 1: (pick_up, Knife) -> Frames 5-9
   Action 2: (navigate_to_obj, Bread) -> Frames 10-15
   Action 3: (slice_obj, Bread) -> Frames 16-20
   Action 4: (put_on, Knife, CounterTop) -> Frames 21-25
   Action 5: (pick_up, BreadSliced) -> Frames 26-31
   Action 6: (navigate_to_obj, Toaster) -> Frames 32-36
   Action 7: (put_in, BreadSliced, Toaster) -> Frames 37-41
   Action 8: (toggle_on, Toaster) -> Frames 42-47

### CRAFT 流程详细信息

   ✅ 加载了 48 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 48 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: blocking
- **Failure Step**: 00:23
- **Failure Reason**: The robot cannot pick up knife due to the pot occluding the knife.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [9-13]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [74/100] toastBread/toastBread-3

### 数据加载信息

✅ 加载了 52 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene', 'object_list', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-4
   Action 1: (pick_up, Knife) -> Frames 5-10
   Action 2: (navigate_to_obj, Bread) -> Frames 11-16
   Action 3: (slice_obj, Bread) -> Frames 17-22
   Action 4: (put_on, Knife, CounterTop) -> Frames 23-27
   Action 5: (pick_up, BreadSliced) -> Frames 28-33
   Action 6: (navigate_to_obj, Toaster) -> Frames 34-39
   Action 7: (put_in, BreadSliced, Toaster) -> Frames 40-45
   Action 8: (toggle_on, Toaster) -> Frames 46-51

### CRAFT 流程详细信息

   ✅ 加载了 52 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 52 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'scene']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
           ❌ Precondition 违反: Toaster must be empty
           ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
       ✅ 检测完成: 4 个违反, 4 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: blocking
- **Failure Step**: 00:47
- **Failure Reason**: The robot cannot put bread slice inside the toaster due to a cellphone on top of the toaster, blocking the slots.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be empty
      Reason: Container 'toaster' contains 1 object(s): CellPhone_aedef2ad
      Frame: Unknown frame

    Derived Violations (派生失败, 1 个):
      这些失败是由根失败导致的级联失败，不单独分析


================================================================================

## [75/100] toastBread/toastBread-4

### 数据加载信息

✅ 加载了 50 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-4
   Action 1: (pick_up, Knife) -> Frames 5-10
   Action 2: (navigate_to_obj, Bread) -> Frames 11-15
   Action 3: (slice_obj, Bread) -> Frames 16-21
   Action 4: (put_on, Knife, CounterTop) -> Frames 22-26
   Action 5: (pick_up, BreadSliced) -> Frames 27-32
   Action 6: (navigate_to_obj, Toaster) -> Frames 33-37
   Action 7: (put_in, BreadSliced, Toaster) -> Frames 38-43
   Action 8: (toggle_on, Toaster) -> Frames 44-49

### CRAFT 流程详细信息

   ✅ 加载了 50 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 50 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:31
- **Failure Reason**: The robot failed to execute the slice bread action.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [9-13]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [76/100] toastBread/toastBread-5

### 数据加载信息

✅ 加载了 50 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-4
   Action 1: (pick_up, Knife) -> Frames 5-10
   Action 2: (navigate_to_obj, Bread) -> Frames 11-15
   Action 3: (slice_obj, Bread) -> Frames 16-21
   Action 4: (put_on, Knife, CounterTop) -> Frames 22-26
   Action 5: (pick_up, BreadSliced) -> Frames 27-32
   Action 6: (navigate_to_obj, Toaster) -> Frames 33-37
   Action 7: (put_in, BreadSliced, Toaster) -> Frames 38-43
   Action 8: (toggle_on, Toaster) -> Frames 44-49

### CRAFT 流程详细信息

   ✅ 加载了 50 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 50 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: 00:45
- **Failure Reason**: The robot failed to put bread slice inside toaster.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [9-13]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [77/100] toastBread/toastBread-6

### 数据加载信息

✅ 加载了 44 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=7, actions=['(navigate_to_obj, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(pick_up, BreadSliced)', '(navigate_to_obj, Toaster)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-5
   Action 1: (navigate_to_obj, Bread) -> Frames 6-11
   Action 2: (slice_obj, Bread) -> Frames 12-17
   Action 3: (pick_up, BreadSliced) -> Frames 18-24
   Action 4: (navigate_to_obj, Toaster) -> Frames 25-30
   Action 5: (put_in, BreadSliced, Toaster) -> Frames 31-36
   Action 6: (toggle_on, Toaster) -> Frames 37-43

### CRAFT 流程详细信息

   ✅ 加载了 44 个 events
   ✅ 建立了动作-帧映射: 7 个动作
       ✅ 加载了 44 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=7, actions=['(navigate_to_obj, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(pick_up, BreadSliced)', '(navigate_to_obj, Toaster)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=7, 前3个动作=['(navigate_to_obj, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)']
       ✅ 生成了 9 个约束
       组织约束...
       ✅ 约束分组: 3 个动作有约束, 0 个目标约束
         检查动作 4/7: (pick_up, BreadSliced)
         检查动作 6/7: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): BreadSliced must be inside Toaster
         检查动作 7/7: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 7-11 内未满足): Toaster must be toggled on
       ✅ 检测完成: 2 个违反, 2 个真实错误, 6 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:01', '00:02', '00:03', '00:04', '00:05', '00:06', '00:07', '00:08', '00:09', '00:10', '00:11', '00:12', '00:13', '00:14', '00:15', '00:16', '00:17', '00:18', '00:19', '00:20', '00:21', '00:22', '00:23', '00:24', '00:25', '00:26', '00:27', '00:28']
- **Failure Reason**: The robot never picked up a knife and thus cannot slice the bread.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [7-11]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [78/100] toastBread/toastBread-7

### 数据加载信息

✅ 加载了 34 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=5, actions=['(navigate_to_obj, Bread)', '(pick_up, Bread)', '(navigate_to_obj, Toaster)', '(put_in, Bread, Toaster)', '(toggle_on, Toaster)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Bread) -> Frames 0-5
   Action 1: (pick_up, Bread) -> Frames 6-12
   Action 2: (navigate_to_obj, Toaster) -> Frames 13-19
   Action 3: (put_in, Bread, Toaster) -> Frames 20-26
   Action 4: (toggle_on, Toaster) -> Frames 27-33

### CRAFT 流程详细信息

   ✅ 加载了 34 个 events
   ✅ 建立了动作-帧映射: 5 个动作
       ✅ 加载了 34 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=5, actions=['(navigate_to_obj, Bread)', '(pick_up, Bread)', '(navigate_to_obj, Toaster)', '(put_in, Bread, Toaster)', '(toggle_on, Toaster)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=5, 前3个动作=['(navigate_to_obj, Bread)', '(pick_up, Bread)', '(navigate_to_obj, Toaster)']
       🔍 调试：动作 2 ((pick_up, Bread)) 生成了 3 个约束
       ✅ 生成了 9 个约束
       组织约束...
       ✅ 约束分组: 3 个动作有约束, 0 个目标约束
         检查动作 2/5: (pick_up, Bread)
         检查动作 4/5: (put_in, Bread, Toaster)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Bread must be inside Toaster
         检查动作 5/5: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Toaster must be toggled on
       ✅ 检测完成: 2 个违反, 2 个真实错误, 6 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:01', '00:02', '00:03', '00:04', '00:05', '00:06', '00:07', '00:08', '00:09', '00:10', '00:11', '00:12', '00:13', '00:14', '00:15', '00:16', '00:17', '00:18', '00:19', '00:20', '00:21']
- **Failure Reason**: The robot never sliced the bread, and the unsliced bread cannot be put inside the toaster.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [5-9]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [79/100] toastBread/toastBread-8

### 数据加载信息

✅ 加载了 63 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(pick_up, BreadSliced)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-6
   Action 1: (pick_up, Knife) -> Frames 7-14
   Action 2: (navigate_to_obj, Bread) -> Frames 15-22
   Action 3: (slice_obj, Bread) -> Frames 23-30
   Action 4: (pick_up, BreadSliced) -> Frames 31-38
   Action 5: (navigate_to_obj, Toaster) -> Frames 39-46
   Action 6: (put_in, BreadSliced, Toaster) -> Frames 47-54
   Action 7: (toggle_on, Toaster) -> Frames 55-62

### CRAFT 流程详细信息

   ✅ 加载了 63 个 events
   ✅ 建立了动作-帧映射: 8 个动作
       ✅ 加载了 63 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'specified_missing_steps', 'success_condition']
       🔍 调试：actions 数量=8, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(pick_up, BreadSliced)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=8, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 12 个约束
       组织约束...
       ✅ 约束分组: 4 个动作有约束, 0 个目标约束
         检查动作 2/8: (pick_up, Knife)
         检查动作 5/8: (pick_up, BreadSliced)
         检查动作 7/8: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 7-14 内未满足): BreadSliced must be inside Toaster
         检查动作 8/8: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Toaster must be toggled on
       ✅ 检测完成: 2 个违反, 2 个真实错误, 9 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:33', '00:34', '00:35', '00:36', '00:37', '00:38', '00:39', '00:40', '00:41', '00:42', '00:43', '00:44', '00:45', '00:46', '00:47', '00:48', '00:49', '00:50']
- **Failure Reason**: The robot did not put the knife down after slicing the bread. As a result, the gripper was occupied by the knife and it could not pick up the bread slice.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Toaster must be toggled on
      Reason: Postcondition not satisfied in temporal window [8-12]. Last reason: toaster is not toggled on
      Frame: Unknown frame


================================================================================

## [80/100] toastBread/toastBread-9

### 数据加载信息

✅ 加载了 52 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Knife) -> Frames 0-4
   Action 1: (pick_up, Knife) -> Frames 5-10
   Action 2: (navigate_to_obj, Bread) -> Frames 11-16
   Action 3: (slice_obj, Bread) -> Frames 17-22
   Action 4: (put_on, Knife, CounterTop) -> Frames 23-27
   Action 5: (pick_up, BreadSliced) -> Frames 28-33
   Action 6: (navigate_to_obj, Toaster) -> Frames 34-39
   Action 7: (toggle_on, Toaster) -> Frames 40-45
   Action 8: (put_in, BreadSliced, Toaster) -> Frames 46-51

### CRAFT 流程详细信息

   ✅ 加载了 52 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 52 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)', '(slice_obj, Bread)', '(put_on, Knife, CounterTop)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Knife)', '(pick_up, Knife)', '(navigate_to_obj, Bread)']
       🔍 调试：动作 2 ((pick_up, Knife)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 5 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (toggle_on, Toaster)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Toaster must be toggled on
         检查动作 9/9: (put_in, BreadSliced, Toaster)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): BreadSliced must be inside Toaster
       ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:47', '00:48', '00:49', '00:50']
- **Failure Reason**: The robot toggled on the toaster before trying to put the bread slice inside the toaster. As a result, the bread slice ended up on top of the toaster instead of inside it.

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: BreadSliced must be inside Toaster
      Reason: Postcondition not satisfied in temporal window [9-16]. Last reason: breadsliced is not inside toaster (checked both 'inside' and 'on_top_of' for container type)
      Frame: Unknown frame


================================================================================

## [81/100] warmWater/warmWater-1

### 数据加载信息

✅ 加载了 112 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-4
   Action 1: (pick_up, Mug) -> Frames 5-10
   Action 2: (navigate_to_obj, Sink) -> Frames 11-16
   Action 3: (put_in, Mug, Sink) -> Frames 17-22
   Action 4: (toggle_on, Faucet) -> Frames 23-28
   Action 5: (toggle_off, Faucet) -> Frames 29-34
   Action 6: (pick_up, Mug) -> Frames 35-40
   Action 7: (navigate_to_obj, Microwave) -> Frames 41-46
   Action 8: (put_on, Mug, CounterTop) -> Frames 47-52
   Action 9: (open_obj, Microwave) -> Frames 53-57
   Action 10: (pick_up, Mug) -> Frames 58-63
   Action 11: (put_in, Mug, Microwave) -> Frames 64-69
   Action 12: (toggle_on, Microwave) -> Frames 70-75
   Action 13: (close_obj, Microwave) -> Frames 76-81
   Action 14: (toggle_off, Microwave) -> Frames 82-87
   Action 15: (open_obj, Microwave) -> Frames 88-93
   Action 16: (pick_up, Mug) -> Frames 94-99
   Action 17: (navigate_to_obj, DiningTable) -> Frames 100-105
   Action 18: (put_on, Mug, DiningTable) -> Frames 106-111

### CRAFT 流程详细信息

   ✅ 加载了 112 个 events
   ✅ 建立了动作-帧映射: 19 个动作
       ✅ 加载了 112 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=19, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 32 个约束
       组织约束...
       ✅ 约束分组: 12 个动作有约束, 0 个目标约束
         检查动作 2/19: (pick_up, Mug)
         检查动作 4/19: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/19: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/19: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/19: (pick_up, Mug)
         检查动作 9/19: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/19: (pick_up, Mug)
         检查动作 12/19: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Microwave
         检查动作 13/19: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 13-17 内未满足): Microwave must be toggled on
         检查动作 15/19: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 17/19: (pick_up, Mug)
         检查动作 19/19: (put_on, Mug, DiningTable)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Mug must be on top of DiningTable
       ✅ 检测完成: 6 个违反, 6 个真实错误, 22 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong order
- **Failure Step**: ['01:04', '01:07']
- **Failure Reason**: Wrong order of toggle on microwave and close microwave

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [19-26]. Last reason: mug is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [82/100] warmWater/warmWater-10

### 数据加载信息

✅ 加载了 114 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-5
   Action 1: (pick_up, Mug) -> Frames 6-11
   Action 2: (navigate_to_obj, Sink) -> Frames 12-17
   Action 3: (put_in, Mug, Sink) -> Frames 18-23
   Action 4: (toggle_on, Faucet) -> Frames 24-29
   Action 5: (toggle_off, Faucet) -> Frames 30-35
   Action 6: (pick_up, Mug) -> Frames 36-41
   Action 7: (navigate_to_obj, Microwave) -> Frames 42-47
   Action 8: (put_on, Mug, CounterTop) -> Frames 48-53
   Action 9: (open_obj, Microwave) -> Frames 54-59
   Action 10: (pick_up, Mug) -> Frames 60-65
   Action 11: (put_in, Mug, Microwave) -> Frames 66-71
   Action 12: (close_obj, Microwave) -> Frames 72-77
   Action 13: (toggle_on, Microwave) -> Frames 78-83
   Action 14: (toggle_off, Microwave) -> Frames 84-89
   Action 15: (open_obj, Microwave) -> Frames 90-95
   Action 16: (pick_up, Mug) -> Frames 96-101
   Action 17: (navigate_to_obj, DiningTable) -> Frames 102-107
   Action 18: (put_on, Mug, DiningTable) -> Frames 108-113

### CRAFT 流程详细信息

   ✅ 加载了 114 个 events
   ✅ 建立了动作-帧映射: 19 个动作
       ✅ 加载了 114 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step']
       🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=19, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 32 个约束
       组织约束...
       ✅ 约束分组: 12 个动作有约束, 0 个目标约束
         检查动作 2/19: (pick_up, Mug)
         检查动作 4/19: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/19: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/19: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/19: (pick_up, Mug)
         检查动作 9/19: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/19: (pick_up, Mug)
         检查动作 12/19: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Microwave
         检查动作 14/19: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/19: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 17/19: (pick_up, Mug)
         检查动作 19/19: (put_on, Mug, DiningTable)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Mug must be on top of DiningTable
       ✅ 检测完成: 6 个违反, 6 个真实错误, 22 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:21']
- **Failure Reason**: Failed to successfully execute (toggle_on, Faucet)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [19-26]. Last reason: mug is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [83/100] warmWater/warmWater-2

### 数据加载信息

✅ 加载了 87 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=17, actions=['(navigate_to_obj, Microwave)', '(open_obj, Microwave)', '(navigate_to_obj, Mug)', '(pick_up, Mug)', '(put_in, Mug, Microwave)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Microwave) -> Frames 0-4
   Action 1: (open_obj, Microwave) -> Frames 5-9
   Action 2: (navigate_to_obj, Mug) -> Frames 10-14
   Action 3: (pick_up, Mug) -> Frames 15-19
   Action 4: (put_in, Mug, Microwave) -> Frames 20-24
   Action 5: (close_obj, Microwave) -> Frames 25-29
   Action 6: (toggle_on, Microwave) -> Frames 30-34
   Action 7: (toggle_off, Microwave) -> Frames 35-39
   Action 8: (open_obj, Microwave) -> Frames 40-45
   Action 9: (pick_up, Mug) -> Frames 46-50
   Action 10: (navigate_to_obj, Sink) -> Frames 51-55
   Action 11: (put_in, Mug, Sink) -> Frames 56-60
   Action 12: (toggle_on, Faucet) -> Frames 61-65
   Action 13: (toggle_off, Faucet) -> Frames 66-70
   Action 14: (pick_up, Mug) -> Frames 71-75
   Action 15: (navigate_to_obj, DiningTable) -> Frames 76-80
   Action 16: (put_on, Mug, DiningTable) -> Frames 81-86

### CRAFT 流程详细信息

   ✅ 加载了 87 个 events
   ✅ 建立了动作-帧映射: 17 个动作
       ✅ 加载了 87 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=17, actions=['(navigate_to_obj, Microwave)', '(open_obj, Microwave)', '(navigate_to_obj, Mug)', '(pick_up, Mug)', '(put_in, Mug, Microwave)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=17, 前3个动作=['(navigate_to_obj, Microwave)', '(open_obj, Microwave)', '(navigate_to_obj, Mug)']
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 10 个动作有约束, 0 个目标约束
         检查动作 4/17: (pick_up, Mug)
         检查动作 5/17: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 5-12 内未满足): Mug must be inside Microwave
         检查动作 7/17: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 7-11 内未满足): Microwave must be toggled on
         检查动作 8/17: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 8 (窗口 8-12) 满足): Microwave must be toggled off
         检查动作 10/17: (pick_up, Mug)
         检查动作 12/17: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Sink
         检查动作 13/17: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 13-17 内未满足): Faucet must be toggled on
         检查动作 14/17: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 14 (窗口 14-18) 满足): Faucet must be toggled off
         检查动作 15/17: (pick_up, Mug)
         检查动作 17/17: (put_on, Mug, DiningTable)
           ❌ Postcondition 违反 (窗口 17-24 内未满足): Mug must be on top of DiningTable
       ✅ 检测完成: 5 个违反, 5 个真实错误, 18 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong order
- **Failure Step**: ['00:21', '01:25']
- **Failure Reason**: Wrong order of filling water and microwaving

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [17-24]. Last reason: mug is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [84/100] warmWater/warmWater-3

### 数据加载信息

✅ 加载了 72 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=15, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-3
   Action 1: (pick_up, Mug) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-13
   Action 3: (put_in, Mug, Sink) -> Frames 14-18
   Action 4: (toggle_on, Faucet) -> Frames 19-23
   Action 5: (toggle_off, Faucet) -> Frames 24-27
   Action 6: (pick_up, Mug) -> Frames 28-32
   Action 7: (navigate_to_obj, Microwave) -> Frames 33-37
   Action 8: (put_on, Mug, CounterTop) -> Frames 38-42
   Action 9: (open_obj, Microwave) -> Frames 43-47
   Action 10: (pick_up, Mug) -> Frames 48-51
   Action 11: (put_in, Mug, Microwave) -> Frames 52-56
   Action 12: (close_obj, Microwave) -> Frames 57-61
   Action 13: (toggle_on, Microwave) -> Frames 62-66
   Action 14: (toggle_off, Microwave) -> Frames 67-71

### CRAFT 流程详细信息

   ✅ 加载了 72 个 events
   ✅ 建立了动作-帧映射: 15 个动作
       ✅ 加载了 72 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=15, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=15, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 27 个约束
       组织约束...
       ✅ 约束分组: 10 个动作有约束, 0 个目标约束
         检查动作 2/15: (pick_up, Mug)
         检查动作 4/15: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/15: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/15: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/15: (pick_up, Mug)
         检查动作 9/15: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/15: (pick_up, Mug)
         检查动作 12/15: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Microwave
         检查动作 14/15: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/15: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
       ✅ 检测完成: 5 个违反, 5 个真实错误, 18 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: incomplete_plan
- **Failure Step**: ['01:10']
- **Failure Reason**: Incomplete Plan: missed steps for serving the mug

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Microwave must be toggled on
      Reason: Postcondition not satisfied in temporal window [14-18]. Last reason: microwave is not toggled on
      Frame: Unknown frame


================================================================================

## [85/100] warmWater/warmWater-4

### 数据加载信息

✅ 加载了 120 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-5
   Action 1: (pick_up, Mug) -> Frames 6-12
   Action 2: (navigate_to_obj, Sink) -> Frames 13-19
   Action 3: (put_in, Mug, Sink) -> Frames 20-25
   Action 4: (toggle_on, Faucet) -> Frames 26-32
   Action 5: (toggle_off, Faucet) -> Frames 33-39
   Action 6: (pick_up, Mug) -> Frames 40-45
   Action 7: (navigate_to_obj, Microwave) -> Frames 46-52
   Action 8: (put_on, Mug, CounterTop) -> Frames 53-59
   Action 9: (open_obj, Microwave) -> Frames 60-65
   Action 10: (pick_up, Mug) -> Frames 66-72
   Action 11: (put_in, Mug, Microwave) -> Frames 73-79
   Action 12: (close_obj, Microwave) -> Frames 80-85
   Action 13: (toggle_on, Microwave) -> Frames 86-92
   Action 14: (toggle_off, Microwave) -> Frames 93-99
   Action 15: (pick_up, Cup) -> Frames 100-105
   Action 16: (navigate_to_obj, DiningTable) -> Frames 106-112
   Action 17: (put_on, Cup, DiningTable) -> Frames 113-119

### CRAFT 流程详细信息

   ✅ 加载了 120 个 events
   ✅ 建立了动作-帧映射: 18 个动作
       ✅ 加载了 120 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=18, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=18, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 32 个约束
       组织约束...
       ✅ 约束分组: 12 个动作有约束, 0 个目标约束
         检查动作 2/18: (pick_up, Mug)
         检查动作 4/18: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/18: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/18: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/18: (pick_up, Mug)
         检查动作 9/18: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/18: (pick_up, Mug)
         检查动作 12/18: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Microwave
         检查动作 14/18: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/18: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 16/18: (pick_up, Cup)
         检查动作 18/18: (put_on, Cup, DiningTable)
           ❌ Postcondition 违反 (窗口 18-25 内未满足): Cup must be on top of DiningTable
       ✅ 检测完成: 6 个违反, 6 个真实错误, 22 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong_plan
- **Failure Step**: ['01:40', '01:58']
- **Failure Reason**: Wrong Plan: serves glass instead of the mug with water

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Cup must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [18-25]. Last reason: cup is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [86/100] warmWater/warmWater-5

### 数据加载信息

✅ 加载了 145 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Cup)', '(pick_up, Cup)', '(navigate_to_obj, Sink)', '(put_in, Cup, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Cup) -> Frames 0-6
   Action 1: (pick_up, Cup) -> Frames 7-14
   Action 2: (navigate_to_obj, Sink) -> Frames 15-21
   Action 3: (put_in, Cup, Sink) -> Frames 22-29
   Action 4: (toggle_on, Faucet) -> Frames 30-37
   Action 5: (toggle_off, Faucet) -> Frames 38-44
   Action 6: (pick_up, Cup) -> Frames 45-52
   Action 7: (navigate_to_obj, Microwave) -> Frames 53-60
   Action 8: (put_on, Cup, CounterTop) -> Frames 61-67
   Action 9: (open_obj, Microwave) -> Frames 68-75
   Action 10: (pick_up, Cup) -> Frames 76-82
   Action 11: (put_in, Cup, Microwave) -> Frames 83-90
   Action 12: (close_obj, Microwave) -> Frames 91-98
   Action 13: (toggle_on, Microwave) -> Frames 99-105
   Action 14: (toggle_off, Microwave) -> Frames 106-113
   Action 15: (open_obj, Microwave) -> Frames 114-121
   Action 16: (pick_up, Cup) -> Frames 122-128
   Action 17: (navigate_to_obj, DiningTable) -> Frames 129-136
   Action 18: (put_on, Cup, DiningTable) -> Frames 137-144

### CRAFT 流程详细信息

   ✅ 加载了 145 个 events
   ✅ 建立了动作-帧映射: 19 个动作
       ✅ 加载了 145 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Cup)', '(pick_up, Cup)', '(navigate_to_obj, Sink)', '(put_in, Cup, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=19, 前3个动作=['(navigate_to_obj, Cup)', '(pick_up, Cup)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Cup)) 生成了 3 个约束
       ✅ 生成了 32 个约束
       组织约束...
       ✅ 约束分组: 12 个动作有约束, 0 个目标约束
         检查动作 2/19: (pick_up, Cup)
         检查动作 4/19: (put_in, Cup, Sink)
           ❌ Precondition 违反: Sink must be empty
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Cup must be inside Sink
         检查动作 5/19: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/19: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/19: (pick_up, Cup)
         检查动作 9/19: (put_on, Cup, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Cup must be on top of CounterTop
         检查动作 11/19: (pick_up, Cup)
         检查动作 12/19: (put_in, Cup, Microwave)
           ❌ Precondition 违反: Microwave must be empty
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Cup must be inside Microwave
         检查动作 14/19: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/19: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 17/19: (pick_up, Cup)
         检查动作 19/19: (put_on, Cup, DiningTable)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Cup must be on top of DiningTable
       ✅ 检测完成: 8 个违反, 8 个真实错误, 22 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: occupied_put
- **Failure Step**: ['01:35']
- **Failure Reason**: Microwave already occupied

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Sink must be empty
      Reason: Container 'sink' contains 1 object(s): Mug_0b3dbbd3
      Frame: Unknown frame

    Derived Violations (派生失败, 6 个):
      这些失败是由根失败导致的级联失败，不单独分析


================================================================================

## [87/100] warmWater/warmWater-6

### 数据加载信息

✅ 加载了 122 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-5
   Action 1: (pick_up, Mug) -> Frames 6-11
   Action 2: (navigate_to_obj, Sink) -> Frames 12-18
   Action 3: (put_in, Mug, Sink) -> Frames 19-24
   Action 4: (toggle_on, Faucet) -> Frames 25-31
   Action 5: (toggle_off, Faucet) -> Frames 32-37
   Action 6: (pick_up, Mug) -> Frames 38-43
   Action 7: (navigate_to_obj, Microwave) -> Frames 44-50
   Action 8: (put_on, Mug, CounterTop) -> Frames 51-56
   Action 9: (open_obj, Microwave) -> Frames 57-63
   Action 10: (pick_up, Mug) -> Frames 64-69
   Action 11: (put_in, Mug, Microwave) -> Frames 70-76
   Action 12: (close_obj, Microwave) -> Frames 77-82
   Action 13: (toggle_on, Microwave) -> Frames 83-88
   Action 14: (toggle_off, Microwave) -> Frames 89-95
   Action 15: (open_obj, Microwave) -> Frames 96-101
   Action 16: (pick_up, Mug) -> Frames 102-108
   Action 17: (navigate_to_obj, DiningTable) -> Frames 109-114
   Action 18: (put_on, Mug, DiningTable) -> Frames 115-121

### CRAFT 流程详细信息

   ✅ 加载了 122 个 events
   ✅ 建立了动作-帧映射: 19 个动作
       ✅ 加载了 122 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params']
       🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=19, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 32 个约束
       组织约束...
       ✅ 约束分组: 12 个动作有约束, 0 个目标约束
         检查动作 2/19: (pick_up, Mug)
         检查动作 4/19: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/19: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/19: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/19: (pick_up, Mug)
         检查动作 9/19: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/19: (pick_up, Mug)
         检查动作 12/19: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Microwave
         检查动作 14/19: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/19: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 17/19: (pick_up, Mug)
         检查动作 19/19: (put_on, Mug, DiningTable)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Mug must be on top of DiningTable
       ✅ 检测完成: 6 个违反, 6 个真实错误, 22 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong perception
- **Failure Step**: ['00:59']
- **Failure Reason**: Wrong perception: glass and mug

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [19-26]. Last reason: mug is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [88/100] warmWater/warmWater-7

### 数据加载信息

✅ 加载了 167 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-7
   Action 1: (pick_up, Mug) -> Frames 8-16
   Action 2: (navigate_to_obj, Sink) -> Frames 17-25
   Action 3: (put_in, Mug, Sink) -> Frames 26-34
   Action 4: (toggle_on, Faucet) -> Frames 35-42
   Action 5: (toggle_off, Faucet) -> Frames 43-51
   Action 6: (pick_up, Mug) -> Frames 52-60
   Action 7: (navigate_to_obj, Microwave) -> Frames 61-69
   Action 8: (put_on, Mug, CounterTop) -> Frames 70-78
   Action 9: (open_obj, Microwave) -> Frames 79-86
   Action 10: (pick_up, Cup) -> Frames 87-95
   Action 11: (put_in, Cup, Microwave) -> Frames 96-104
   Action 12: (close_obj, Microwave) -> Frames 105-113
   Action 13: (toggle_on, Microwave) -> Frames 114-122
   Action 14: (toggle_off, Microwave) -> Frames 123-130
   Action 15: (open_obj, Microwave) -> Frames 131-139
   Action 16: (pick_up, Mug) -> Frames 140-148
   Action 17: (navigate_to_obj, DiningTable) -> Frames 149-157
   Action 18: (put_on, Mug, DiningTable) -> Frames 158-166

### CRAFT 流程详细信息

   ✅ 加载了 167 个 events
   ✅ 建立了动作-帧映射: 19 个动作
       ✅ 加载了 167 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=19, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 32 个约束
       组织约束...
       ✅ 约束分组: 12 个动作有约束, 0 个目标约束
         检查动作 2/19: (pick_up, Mug)
         检查动作 4/19: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/19: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/19: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/19: (pick_up, Mug)
         检查动作 9/19: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/19: (pick_up, Cup)
         检查动作 12/19: (put_in, Cup, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Cup must be inside Microwave
         检查动作 14/19: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/19: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 17/19: (pick_up, Mug)
         检查动作 19/19: (put_on, Mug, DiningTable)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Mug must be on top of DiningTable
       ✅ 检测完成: 6 个违反, 6 个真实错误, 22 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['01:21', '01:51']
- **Failure Reason**: Wrong Plan: puts cup inside microwave instead of mug filled with water

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [19-26]. Last reason: mug is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [89/100] warmWater/warmWater-8

### 数据加载信息

✅ 加载了 116 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-5
   Action 1: (pick_up, Mug) -> Frames 6-11
   Action 2: (navigate_to_obj, Sink) -> Frames 12-17
   Action 3: (put_in, Mug, Sink) -> Frames 18-23
   Action 4: (toggle_on, Faucet) -> Frames 24-29
   Action 5: (toggle_off, Faucet) -> Frames 30-35
   Action 6: (pick_up, Mug) -> Frames 36-41
   Action 7: (navigate_to_obj, Microwave) -> Frames 42-47
   Action 8: (put_on, Mug, CounterTop) -> Frames 48-53
   Action 9: (open_obj, Microwave) -> Frames 54-60
   Action 10: (pick_up, Mug) -> Frames 61-66
   Action 11: (put_in, Mug, Microwave) -> Frames 67-72
   Action 12: (close_obj, Microwave) -> Frames 73-78
   Action 13: (toggle_on, Microwave) -> Frames 79-84
   Action 14: (toggle_off, Microwave) -> Frames 85-90
   Action 15: (open_obj, Microwave) -> Frames 91-96
   Action 16: (pick_up, Mug) -> Frames 97-102
   Action 17: (navigate_to_obj, DiningTable) -> Frames 103-108
   Action 18: (put_on, Mug, DiningTable) -> Frames 109-115

### CRAFT 流程详细信息

   ✅ 加载了 116 个 events
   ✅ 建立了动作-帧映射: 19 个动作
       ✅ 加载了 116 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'preactions', 'actions']
       🔍 调试：actions 数量=19, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=19, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 32 个约束
       组织约束...
       ✅ 约束分组: 12 个动作有约束, 0 个目标约束
         检查动作 2/19: (pick_up, Mug)
         检查动作 4/19: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/19: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/19: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/19: (pick_up, Mug)
         检查动作 9/19: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/19: (pick_up, Mug)
         检查动作 12/19: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Microwave
         检查动作 14/19: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/19: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 17/19: (pick_up, Mug)
         检查动作 19/19: (put_on, Mug, DiningTable)
           ❌ Postcondition 违反 (窗口 19-26 内未满足): Mug must be on top of DiningTable
       ✅ 检测完成: 6 个违反, 6 个真实错误, 22 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:20']
- **Failure Reason**: Missing step to pour wine out of the mug

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [19-26]. Last reason: mug is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [90/100] warmWater/warmWater-9

### 数据加载信息

✅ 加载了 120 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=21, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Mug) -> Frames 0-4
   Action 1: (pick_up, Mug) -> Frames 5-10
   Action 2: (navigate_to_obj, Sink) -> Frames 11-16
   Action 3: (put_in, Mug, Sink) -> Frames 17-21
   Action 4: (toggle_on, Faucet) -> Frames 22-27
   Action 5: (toggle_off, Faucet) -> Frames 28-33
   Action 6: (pick_up, Mug) -> Frames 34-39
   Action 7: (navigate_to_obj, Microwave) -> Frames 40-44
   Action 8: (put_on, Mug, CounterTop) -> Frames 45-50
   Action 9: (open_obj, Microwave) -> Frames 51-56
   Action 10: (pick_up, Mug) -> Frames 57-61
   Action 11: (put_in, Mug, Microwave) -> Frames 62-67
   Action 12: (close_obj, Microwave) -> Frames 68-73
   Action 13: (toggle_on, Microwave) -> Frames 74-79
   Action 14: (toggle_off, Microwave) -> Frames 80-84
   Action 15: (open_obj, Microwave) -> Frames 85-90
   Action 16: (pick_up, Mug) -> Frames 91-96
   Action 17: (navigate_to_obj, Sink) -> Frames 97-101
   Action 18: (pour, Mug, Sink) -> Frames 102-107
   Action 19: (navigate_to_obj, DiningTable) -> Frames 108-113
   Action 20: (put_on, Mug, DiningTable) -> Frames 114-119

### CRAFT 流程详细信息

   ✅ 加载了 120 个 events
   ✅ 建立了动作-帧映射: 21 个动作
       ✅ 加载了 120 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition']
       🔍 调试：actions 数量=21, actions=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)', '(put_in, Mug, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=21, 前3个动作=['(navigate_to_obj, Mug)', '(pick_up, Mug)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Mug)) 生成了 3 个约束
       ✅ 生成了 34 个约束
       组织约束...
       ✅ 约束分组: 13 个动作有约束, 0 个目标约束
         检查动作 2/21: (pick_up, Mug)
         检查动作 4/21: (put_in, Mug, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Mug must be inside Sink
         检查动作 5/21: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/21: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/21: (pick_up, Mug)
         检查动作 9/21: (put_on, Mug, CounterTop)
           ❌ Postcondition 违反 (窗口 9-16 内未满足): Mug must be on top of CounterTop
         检查动作 11/21: (pick_up, Mug)
         检查动作 12/21: (put_in, Mug, Microwave)
           ❌ Postcondition 违反 (窗口 12-19 内未满足): Mug must be inside Microwave
         检查动作 14/21: (toggle_on, Microwave)
           ❌ Postcondition 违反 (窗口 14-18 内未满足): Microwave must be toggled on
         检查动作 15/21: (toggle_off, Microwave)
           ✅ Postcondition 满足 (在 帧 15 (窗口 15-19) 满足): Microwave must be toggled off
         检查动作 17/21: (pick_up, Mug)
         检查动作 19/21: (pour, Mug, Sink)
           ❌ Postcondition 违反 (窗口 19-23 内未满足): Sink must be filled
         检查动作 21/21: (put_on, Mug, DiningTable)
           ❌ Postcondition 违反 (窗口 21-28 内未满足): Mug must be on top of DiningTable
       ✅ 检测完成: 7 个违反, 7 个真实错误, 24 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['01:29']
- **Failure Reason**: Wrong Plan: Pours water from mug after microwaving and before serving

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: Mug must be on top of DiningTable
      Reason: Postcondition not satisfied in temporal window [21-28]. Last reason: mug is not on top of diningtable
      Frame: Unknown frame


================================================================================

## [91/100] waterPlant/waterPlant-1

### 数据加载信息

✅ 加载了 41 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-12
   Action 3: (put_in, Pot, Sink) -> Frames 13-17
   Action 4: (toggle_on, Faucet) -> Frames 18-21
   Action 5: (toggle_off, Faucet) -> Frames 22-26
   Action 6: (pick_up, Pot) -> Frames 27-30
   Action 7: (navigate_to_obj, HousePlant) -> Frames 31-35
   Action 8: (pour, Pot, HousePlant) -> Frames 36-40

### CRAFT 流程详细信息

   ✅ 加载了 41 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 41 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'preactions', 'gt_failure_reason', 'gt_failure_step']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Pot)
         检查动作 9/9: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:25', '00:39']
- **Failure Reason**: Incomplete Plan - Pot is filled with wine already

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [92/100] waterPlant/waterPlant-10

### 数据加载信息

✅ 加载了 39 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-7
   Action 2: (navigate_to_obj, Sink) -> Frames 8-12
   Action 3: (put_in, Pot, Sink) -> Frames 13-16
   Action 4: (toggle_on, Faucet) -> Frames 17-20
   Action 5: (toggle_off, Faucet) -> Frames 21-25
   Action 6: (pick_up, Pot) -> Frames 26-29
   Action 7: (navigate_to_obj, HousePlant) -> Frames 30-33
   Action 8: (pour, Pot, HousePlant) -> Frames 34-38

### CRAFT 流程详细信息

   ✅ 加载了 39 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 39 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Pot)
         检查动作 9/9: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:32']
- **Failure Reason**: Failed to successfully execute (pick_up, Pot)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [93/100] waterPlant/waterPlant-2

### 数据加载信息

✅ 加载了 39 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_on, Pot, CounterTop)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-7
   Action 2: (navigate_to_obj, Sink) -> Frames 8-12
   Action 3: (put_on, Pot, CounterTop) -> Frames 13-16
   Action 4: (toggle_on, Faucet) -> Frames 17-20
   Action 5: (toggle_off, Faucet) -> Frames 21-25
   Action 6: (pick_up, Pot) -> Frames 26-29
   Action 7: (navigate_to_obj, HousePlant) -> Frames 30-33
   Action 8: (pour, Pot, HousePlant) -> Frames 34-38

### CRAFT 流程详细信息

   ✅ 加载了 39 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 39 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_on, Pot, CounterTop)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 14 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_on, Pot, CounterTop)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be on top of CounterTop
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Pot)
         检查动作 9/9: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 11 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:23', '00:37']
- **Failure Reason**: Wrong Plan - pot is put on countertop instead of inside sink

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [94/100] waterPlant/waterPlant-3

### 数据加载信息

✅ 加载了 45 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=10, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-12
   Action 3: (put_in, Pot, Sink) -> Frames 13-17
   Action 4: (toggle_on, Faucet) -> Frames 18-21
   Action 5: (toggle_off, Faucet) -> Frames 22-26
   Action 6: (pick_up, Pot) -> Frames 27-30
   Action 7: (pour, Pot, Sink) -> Frames 31-35
   Action 8: (navigate_to_obj, HousePlant) -> Frames 36-39
   Action 9: (pour, Pot, HousePlant) -> Frames 40-44

### CRAFT 流程详细信息

   ✅ 加载了 45 个 events
   ✅ 建立了动作-帧映射: 10 个动作
       ✅ 加载了 45 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
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
         检查动作 8/10: (pour, Pot, Sink)
           ❌ Postcondition 违反 (窗口 8-12 内未满足): Sink must be filled
         检查动作 10/10: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 10-14 内未满足): HousePlant must be filled
       ✅ 检测完成: 4 个违反, 4 个真实错误, 14 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:38']
- **Failure Reason**: Wrong Plan - wrong step: (pour, Pot, Sink)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [10-14]
      Frame: Unknown frame


================================================================================

## [95/100] waterPlant/waterPlant-4

### 数据加载信息

✅ 加载了 41 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(toggle_on, Faucet)', '(toggle_off, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-12
   Action 3: (toggle_on, Faucet) -> Frames 13-17
   Action 4: (toggle_off, Faucet) -> Frames 18-21
   Action 5: (put_in, Pot, Sink) -> Frames 22-26
   Action 6: (pick_up, Pot) -> Frames 27-30
   Action 7: (navigate_to_obj, HousePlant) -> Frames 31-35
   Action 8: (pour, Pot, HousePlant) -> Frames 36-40

### CRAFT 流程详细信息

   ✅ 加载了 41 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 41 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'actions']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(toggle_on, Faucet)', '(toggle_off, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 4-8 内未满足): Faucet must be toggled on
         检查动作 5/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 5 (窗口 5-9) 满足): Faucet must be toggled off
         检查动作 6/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 6-13 内未满足): Pot must be inside Sink
         检查动作 7/9: (pick_up, Pot)
         检查动作 9/9: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:28', '00:31']
- **Failure Reason**: Wrong order - faucet is turned on and off before pot is put in the sink

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [96/100] waterPlant/waterPlant-5

### 数据加载信息

✅ 加载了 41 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-12
   Action 3: (put_in, Pot, Sink) -> Frames 13-17
   Action 4: (toggle_on, Faucet) -> Frames 18-21
   Action 5: (toggle_off, Faucet) -> Frames 22-26
   Action 6: (pick_up, Kettle) -> Frames 27-30
   Action 7: (navigate_to_obj, HousePlant) -> Frames 31-35
   Action 8: (pour, Kettle, HousePlant) -> Frames 36-40

### CRAFT 流程详细信息

   ✅ 加载了 41 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 41 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Kettle)
         检查动作 9/9: (pour, Kettle, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: wrong_plan
- **Failure Step**: ['00:34']
- **Failure Reason**: Wrong plan - kettle is picked up instead of pot

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [97/100] waterPlant/waterPlant-6

### 数据加载信息

✅ 加载了 58 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-5
   Action 1: (pick_up, Pot) -> Frames 6-11
   Action 2: (navigate_to_obj, Sink) -> Frames 12-18
   Action 3: (put_in, Pot, Sink) -> Frames 19-24
   Action 4: (toggle_on, Faucet) -> Frames 25-31
   Action 5: (toggle_off, Faucet) -> Frames 32-37
   Action 6: (pick_up, Container) -> Frames 38-44
   Action 7: (navigate_to_obj, HousePlant) -> Frames 45-50
   Action 8: (pour, Bowl, HousePlant) -> Frames 51-57

### CRAFT 流程详细信息

   ✅ 加载了 58 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 58 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'gt_failure_reason', 'gt_failure_step', 'chosen_failure', 'failure_injection_params']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Container)
         检查动作 9/9: (pour, Bowl, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: ambiguous_plan
- **Failure Step**: ['00:44']
- **Failure Reason**: Ambiguous plan - some container

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [98/100] waterPlant/waterPlant-7

### 数据加载信息

✅ 加载了 38 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params', 'actions', 'success_condition', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-7
   Action 2: (navigate_to_obj, Sink) -> Frames 8-11
   Action 3: (put_in, Pot, Sink) -> Frames 12-15
   Action 4: (toggle_on, Faucet) -> Frames 16-20
   Action 5: (toggle_off, Faucet) -> Frames 21-24
   Action 6: (pick_up, Pot) -> Frames 25-28
   Action 7: (navigate_to_obj, HousePlant) -> Frames 29-32
   Action 8: (pour, Pot, HousePlant) -> Frames 33-37

### CRAFT 流程详细信息

   ✅ 加载了 38 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 38 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'chosen_failure', 'gt_failure_reason', 'gt_failure_step', 'failure_injection_params']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Pot)
         检查动作 9/9: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: True
- **Failure Type**: blocking
- **Failure Step**: ['00:16']
- **Failure Reason**: Pan is blocking the pot

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [99/100] waterPlant/waterPlant-8

### 数据加载信息

✅ 加载了 39 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-7
   Action 2: (navigate_to_obj, Sink) -> Frames 8-12
   Action 3: (put_in, Pot, Sink) -> Frames 13-16
   Action 4: (toggle_on, Faucet) -> Frames 17-20
   Action 5: (toggle_off, Faucet) -> Frames 21-25
   Action 6: (pick_up, Pot) -> Frames 26-29
   Action 7: (navigate_to_obj, HousePlant) -> Frames 30-33
   Action 8: (pour, Pot, HousePlant) -> Frames 34-38

### CRAFT 流程详细信息

   ✅ 加载了 39 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 39 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Pot)
         检查动作 9/9: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:26']
- **Failure Reason**: Failed to successfully execute (toggle_on, Faucet)

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================

## [100/100] waterPlant/waterPlant-9

### 数据加载信息

✅ 加载了 41 个 events
🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason', 'gt_failure_step', 'specific_folder_name', 'unity_name_map', 'sounds']
🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']

### 动作-帧映射

   Action 0: (navigate_to_obj, Pot) -> Frames 0-3
   Action 1: (pick_up, Pot) -> Frames 4-8
   Action 2: (navigate_to_obj, Sink) -> Frames 9-12
   Action 3: (put_in, Pot, Sink) -> Frames 13-17
   Action 4: (toggle_on, Faucet) -> Frames 18-21
   Action 5: (toggle_off, Faucet) -> Frames 22-26
   Action 6: (pick_up, Pot) -> Frames 27-30
   Action 7: (navigate_to_obj, HousePlant) -> Frames 31-35
   Action 8: (pour, Pot, HousePlant) -> Frames 36-40

### CRAFT 流程详细信息

   ✅ 加载了 41 个 events
   ✅ 建立了动作-帧映射: 9 个动作
       ✅ 加载了 41 个 events
       🔍 调试：task_info.keys()=['task_idx', 'failure_injection', 'reps', 'name', 'general_folder_name', 'scene', 'object_list', 'actions', 'success_condition', 'gt_failure_reason']
       🔍 调试：actions 数量=9, actions=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)', '(put_in, Pot, Sink)', '(toggle_on, Faucet)']
       生成场景图...
       生成约束...
       🔍 调试：actions 数量=9, 前3个动作=['(navigate_to_obj, Pot)', '(pick_up, Pot)', '(navigate_to_obj, Sink)']
       🔍 调试：动作 2 ((pick_up, Pot)) 生成了 3 个约束
       ✅ 生成了 16 个约束
       组织约束...
       ✅ 约束分组: 6 个动作有约束, 0 个目标约束
         检查动作 2/9: (pick_up, Pot)
         检查动作 4/9: (put_in, Pot, Sink)
           ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/9: (toggle_on, Faucet)
           ❌ Postcondition 违反 (窗口 5-9 内未满足): Faucet must be toggled on
         检查动作 6/9: (toggle_off, Faucet)
           ✅ Postcondition 满足 (在 帧 6 (窗口 6-10) 满足): Faucet must be toggled off
         检查动作 7/9: (pick_up, Pot)
         检查动作 9/9: (pour, Pot, HousePlant)
           ❌ Postcondition 违反 (窗口 9-13 内未满足): HousePlant must be filled
       ✅ 检测完成: 3 个违反, 3 个真实错误, 12 个跳过约束

### Ground Truth

- **Has Failure**: False
- **Failure Type**: N/A
- **Failure Step**: ['00:36']
- **Failure Reason**: Dropped Pot

### 约束检查日志


### 根因分析

  🔍 根因分析 (Root Cause Analysis):
    Root Violation (根失败):
      Step ?: Unknown
      Type: UNKNOWN
      Constraint: HousePlant must be filled
      Reason: Postcondition not satisfied in temporal window [9-13]
      Frame: Unknown frame


================================================================================
