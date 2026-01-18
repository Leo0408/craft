# CRAFT 批量评估详细日志
生成时间: 2026-01-18 01:03:12
配置: LLM分析=False, GPT模型=gpt-3.5-turbo, 实例过滤=[0]
数据集数量: 10

================================================================================

## [1/10] boilWater/boilWater-1

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

## [2/10] cookEgg/cookEgg-1

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

## [3/10] heatPotato/heatPotato-1

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

## [4/10] makeCoffee/makeCoffee-1

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

## [5/10] makeSalad/makeSalad-1

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

## [6/10] storeEgg/storeEgg-1

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

## [7/10] switchDevices/switchDevices-1

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

## [8/10] toastBread/toastBread-1

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

## [9/10] warmWater/warmWater-1

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

## [10/10] waterPlant/waterPlant-1

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
