# Step 3-5 详细分析总结

## 两个实例的动作和帧对应关系

### makeCoffee-1
- 动作数量: 14
- 事件数量: 50
- 动作-帧对应: 1:1 (Frame i 对应 Action i)

### boilWater-1
- 动作数量: 10
- 事件数量: 49
- 动作-帧对应: 1:1 (Frame i 对应 Action i)

## Precondition 和 Postcondition 的帧选择

### Precondition
- **使用的帧**: `events[action_idx]` (当前动作对应的帧，动作执行前)
- **说明**: 不是前一个动作的帧，而是当前动作执行前的帧

### Postcondition
- **起始帧**: `events[action_idx + 1]`
- **时间窗口大小 K** (根据动作类型):
  - toggle 动作: 5 帧
  - put_in / put_on 动作: 8 帧
  - pick_up 动作: 3 帧
  - 默认: 5 帧
- **结束帧**: `min(action_idx + 1 + K, len(events))`
- **说明**: 在时间窗口内检查，只要窗口内任何一帧满足，就认为满足

## 失败检测检测到的帧

### makeCoffee-1 可能检测到的帧
| 动作索引 | 动作 | Pre帧 | Post帧范围 | 窗口大小 |
|---------|------|-------|-----------|---------|
| 0 | (navigate_to_obj, Mug) | 0 | 1-5 | 5 |
| 1 | (pick_up, Mug) | 1 | 2-4 | 3 |
| 2 | (navigate_to_obj, Sink) | 2 | 3-7 | 5 |
| 3 | (put_on, Mug, SinkBasin) | 3 | 4-11 | 8 |
| 4 | (toggle_on, Faucet) | 4 | 5-9 | 5 |
| 5 | (toggle_off, Faucet) | 5 | 6-10 | 5 |
| 6 | (pick_up, Mug) | 6 | 7-9 | 3 |
| 7 | (pour, Mug, Sink) | 7 | 8-12 | 5 |
| 8 | (navigate_to_obj, CoffeeMachine) | 8 | 9-13 | 5 |
| 9 | (put_in, Mug, CoffeeMachine) | 9 | 10-17 | 8 |
| 10 | (toggle_on, CoffeeMachine) | 10 | 11-15 | 5 |
| 11 | (toggle_off, CoffeeMachine) | 11 | 12-16 | 5 |
| 12 | (pick_up, Mug) | 12 | 13-15 | 3 |
| 13 | (put_on, Mug, CounterTop) | 13 | 14-21 | 8 |

### boilWater-1 可能检测到的帧
| 动作索引 | 动作 | Pre帧 | Post帧范围 | 窗口大小 |
|---------|------|-------|-----------|---------|
| 0 | (navigate_to_obj, Pot) | 0 | 1-5 | 5 |
| 1 | (pick_up, Pot) | 1 | 2-4 | 3 |
| 2 | (navigate_to_obj, Sink) | 2 | 3-7 | 5 |
| 3 | (put_in, Pot, Sink) | 3 | 4-11 | 8 |
| 4 | (toggle_on, Faucet) | 4 | 5-9 | 5 |
| 5 | (toggle_off, Faucet) | 5 | 6-10 | 5 |
| 6 | (pick_up, Pot) | 6 | 7-9 | 3 |
| 7 | (navigate_to_obj, StoveBurner-4) | 7 | 8-12 | 5 |
| 8 | (put_on, Pot, StoveBurner-4) | 8 | 9-16 | 8 |
| 9 | (toggle_on, StoveBurner-4) | 9 | 10-14 | 5 |

## 关键点

1. **动作之间的帧数差距**: 如果使用关键帧，通常是 1 帧（1:1 对应）
2. **Precondition**: 使用当前动作对应的帧（不是前一个动作）
3. **Postcondition**: 使用动作执行后的时间窗口（多帧检查）
4. **实际检测结果**: 需要运行 Step 5 的完整失败检测流程才能看到实际检测到的失败帧
