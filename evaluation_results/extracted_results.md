## [1] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot did not pick up the pot from sink before moving to stove burner.",
"约束条件输出"
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

---

## [2] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(put_in, Pot, Sink)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot put the pot in sink after the faucet was turned off, as a result, the pot was not filled with water.",
"约束条件输出"
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

---

## [3] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot put the pot on the fourth stove burner but toggled on the second stove burner (instead of the fourth stove burner).",
"约束条件输出"
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

---

## [4] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(toggle_on, Faucet)",
        "(put_in, Pot, Sink)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "An apple is inside the pot at the beginning of the task execution, and the robot never removed it from the pot.",
"约束条件输出"
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

---

## [5] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot failed to toggle on faucet.",
"约束条件输出"
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

---

## [6] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot mis-identified pan as pot, and picked up the pan instead of the pot.",
"约束条件输出"
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

---

## [7] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot never executed the actions to toggle on and off the faucet, as a result, the pot was never filled with water.",
"约束条件输出"
         检查动作 2/8: (pick_up, Pot)
         检查动作 4/8: (put_in, Pot, Sink)
         ❌ Postcondition 违反 (窗口 4-11 内未满足): Pot must be inside Sink
         检查动作 5/8: (pick_up, Pot)
         检查动作 7/8: (put_on, Pot, StoveBurner-4)
         ❌ Postcondition 违反 (窗口 7-14 内未满足): Pot must be on top of StoveBurner-4
         检查动作 8/8: (toggle_on, StoveBurner-4)
         ❌ Postcondition 违反 (窗口 8-12 内未满足): StoveBurner-4 must be toggled on
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [8] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot forgot to pick up the pot from sink, as a result, nothing was placed on the fourth stove burner.",
"约束条件输出"
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

---

## [9] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Sink)",
        "(put_in, Bowl, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, StoveBurner-4)",
        "(put_on, Bowl, StoveBurner-4)",
        "(toggle_on, StoveBurner-4)"
      ],
"真实原因": "The robot should use a pot instead of a bowl to boil water. The bowl cannot be put on the stove burner.",
"约束条件输出"
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

---

## [10] boil water

"任务名称": "boil water",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, StoveBurner)",
        "(put_on, Pot, StoveBurner-4)",
        "(toggle_on, StoveBurner-2)"
      ],
"真实原因": "The robot put the pot on the fourth stove burner but toggled on the second stove burner (instead of the fourth stove burner).",
"约束条件输出"
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

---

## [11] cook an egg

"任务名称": "cook an egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "Dropped Egg",
"约束条件输出"
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

---

## [12] cook egg

"任务名称": "cook egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-4)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The robot toggled on the fourth stove burner but put the pan on the first stove burner instead.",
"约束条件输出"
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

---

## [13] cook an egg

"任务名称": "cook an egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The robot failed to put the pan on the first stove burner because there was already a pot on it.",
"约束条件输出"
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

---

## [14] cook egg

"任务名称": "cook egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "A potato is inside the pan at the beginning of the task execution, and the robot never removed it from the pan.",
"约束条件输出"
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

---

## [15] cook an egg

"任务名称": "cook an egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The robot failed to open the fridge.",
"约束条件输出"
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

---

## [16] cook an egg

"任务名称": "cook an egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The robot mis-identified book as pan, and picked up the book instead of the pan. The book cannot be put on the stove burner.",
"约束条件输出"
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

---

## [17] cook an egg

"任务名称": "cook an egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The robot never cracked the egg and put an uncracked egg in the pan.",
"约束条件输出"
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

---

## [18] cook an egg

"任务名称": "cook an egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The robot never opened the fridge, as a result, it could not retrieve the egg from fridge.",
"约束条件输出"
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

---

## [19] cook egg

"任务名称": "cook egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(put_on, Egg, CounterTop)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The pan is dirty at the beginning of the task execution, and the robot never cleaned the pan.",
"约束条件输出"
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

---

## [20] cook egg

"任务名称": "cook egg",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, StoveBurner-1)",
        "(toggle_on, StoveBurner-1)",
        "(navigate_to_obj, Pan)",
        "(pick_up, Pan)",
        "(navigate_to_obj, StoveBurner-1)",
        "(put_on, Pan, StoveBurner-1)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(crack_obj, Egg)",
        "(put_in, EggCracked, Pan)"
      ],
"真实原因": "The robot did not put down the egg in its gripper before trying to pick up the pan.",
"约束条件输出"
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

---

## [21] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Plate, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Plate)",
        "(put_in, Plate, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)"
      ],
"真实原因": "The robot should not toggle on the microwave before trying to open it. As a result, the robot cannot open a microwave that is turned on.",
"约束条件输出"
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

---

## [22] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Plate, CounterTop)",
        "(toggle_on, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Plate)",
        "(put_in, Plate, Microwave)",
        "(close_obj, Microwave)"
      ],
"真实原因": "The robot should not toggle on the microwave before trying to open it. As a result, the robot cannot open a microwave that is turned on.",
"约束条件输出"
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

---

## [23] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Plate, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Plate)",
        "(put_in, Plate, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)"
      ],
"真实原因": "A bowl is already inside the microwave, as a result, the plate cannot be put inside the microwave due to limited space.",
"约束条件输出"
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

---

## [24] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Plate, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Plate)",
        "(put_in, Plate, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)"
      ],
"真实原因": "An apple is inside the plate already and the robot never removed it, as a result, the potato cannot be put on top of the plate due to limited plate.",
"约束条件输出"
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

---

## [25] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(navigate_to_obj, Microwave)",
        "(open_obj, Microwave)",
        "(navigate_to_obj, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_in, Plate, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)"
      ],
"真实原因": "The robot failed to open the microwave.",
"约束条件输出"
         检查动作 2/11: (pick_up, Potato)
         检查动作 3/11: (put_on, Potato, Plate)
         ❌ Postcondition 违反 (窗口 3-10 内未满足): Potato must be on top of Plate
         检查动作 7/11: (pick_up, Plate)
         检查动作 9/11: (put_in, Plate, Microwave)
         ❌ Postcondition 违反 (窗口 9-16 内未满足): Plate must be inside Microwave
         检查动作 11/11: (toggle_on, Microwave)
         ❌ Postcondition 违反 (窗口 11-15 内未满足): Microwave must be toggled on
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [26] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Plate)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_in, Plate, Microwave)"
      ],
"真实原因": "The robot never opened the microwave, as a result, the plate cannot be put inside the microwave.",
"约束条件输出"
         检查动作 2/7: (pick_up, Potato)
         检查动作 4/7: (put_on, Potato, Plate)
         ❌ Postcondition 违反 (窗口 4-11 内未满足): Potato must be on top of Plate
         检查动作 5/7: (pick_up, Plate)
         检查动作 7/7: (put_in, Plate, Microwave)
         ❌ Postcondition 违反 (窗口 7-14 内未满足): Plate must be inside Microwave
         ✅ 检测完成: 2 个违反, 2 个真实错误, 9 个跳过约束

---

## [27] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Plate, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Plate)",
        "(put_in, Plate, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)"
      ],
"真实原因": "The robot mis-identified apple as potato, and heated an apple instead of a potato.",
"约束条件输出"
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

---

## [28] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Plate, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Plate)",
        "(put_in, Plate, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)"
      ],
"真实原因": "The plate is dirty at the beginning of the task execution, and the robot never cleaned it.",
"约束条件输出"
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

---

## [29] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Pan)",
        "(pick_up, Pan)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Pan, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Pan)",
        "(put_in, Pan, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)"
      ],
"真实原因": "The robot should use a microwave-safe container (e.g. Plate) to heat the potato instead of a pan.",
"约束条件输出"
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

---

## [30] heat potato

"任务名称": "heat potato",
"原计划动作": [
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(put_on, Potato, Plate)",
        "(pick_up, Plate)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Plate, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Plate)",
        "(put_in, Plate, Microwave)",
        "(toggle_on, Microwave)",
        "(close_obj, Microwave)"
      ],
"真实原因": "The robot should have closed the microwave before trying to toggle it on. As a result, the robot could not toggle on a microwave that is open.",
"约束条件输出"
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

---

## [31] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space.",
"约束条件输出"
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

---

## [32] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The robot put the mug inside the coffee machine after the coffee machine was turned off, as a result, the mug remained empty.",
"约束条件输出"
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

---

## [33] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "Dropped Mug",
"约束条件输出"
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

---

## [34] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The robot failed to put the mug inside the coffee machine because there was already a cup inside it, occupying the space.",
"约束条件输出"
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

---

## [35] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The mug was already filled with water at the beginning of the task execution, and the robot never emptied it.",
"约束条件输出"
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

---

## [36] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The robot failed to put the mug inside the sink (or on top of the sink basin).",
"约束条件输出"
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

---

## [37] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The robot mis-identified bowl as mug, as a result, the bowl cannot be put inside the coffee machine.",
"约束条件输出"
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

---

## [38] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_on, Mug, SinkBasin)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The robot never executed the action to pour water from mug after cleaning it. As a result, the mug cannot be filled with coffee.",
"约束条件输出"
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

---

## [39] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Mug, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Mug)",
        "(put_on, Mug, CounterTop)"
      ],
"真实原因": "The mug is dirty at the beginning of the task execution, and the robot never cleaned the mug.",
"约束条件输出"
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

---

## [40] make coffee

"任务名称": "make coffee",
"原计划动作": [
        "(pick_up, Bowl)",
        "(navigate_to_obj, CoffeeMachine)",
        "(put_in, Bowl, CoffeeMachine)",
        "(toggle_on, CoffeeMachine)",
        "(toggle_off, CoffeeMachine)",
        "(pick_up, Bowl)",
        "(put_on, Bowl, CounterTop)"
      ],
"真实原因": "The robot plan should not use a bowl instead of a mug or cup to make coffee. As a result, the bowl cannot be put inside the coffee machine.",
"约束条件输出"
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

---

## [41] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(slice_obj, Potato)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Wrong order - knife is put on countertop before slicing tomato",
"约束条件输出"
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

---

## [42] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Failed to successfully execute (pick_up, TomatoSliced)",
"约束条件输出"
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

---

## [43] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Pan)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "None",
"约束条件输出"
         (无约束检查输出)

---

## [44] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Apple)",
        "(pick_up, Apple)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Apple, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Apple)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(pick_up, AppleSliced)",
        "(put_in, AppleSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Wrong plan - apple instead of tomato",
"约束条件输出"
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

---

## [45] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Wrong perception: pan and bowl",
"约束条件输出"
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

---

## [46] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(pick_up, Potato)",
        "(put_in, Potato, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Missing step - slice potato",
"约束条件输出"
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

---

## [47] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Failed to successfully execute (pick_up, Bowl)",
"约束条件输出"
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

---

## [48] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Missing (pick_up, LettuceSliced), (put_in, LettuceSliced, Bowl)",
"约束条件输出"
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

---

## [49] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Failed to successfully execute (put_on, Tomato, CounterTop)",
"约束条件输出"
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

---

## [50] make a salad with tomato, potato and lettuce and store in the fridge

"任务名称": "make a salad with tomato, potato and lettuce and store in the fridge",
"原计划动作": [
        "(navigate_to_obj, Lettuce)",
        "(pick_up, Lettuce)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Lettuce, CounterTop)",
        "(navigate_to_obj, Tomato)",
        "(pick_up, Tomato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Tomato, CounterTop)",
        "(navigate_to_obj, Potato)",
        "(pick_up, Potato)",
        "(navigate_to_obj, Bowl)",
        "(put_on, Potato, CounterTop)",
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bowl)",
        "(slice_obj, Lettuce)",
        "(slice_obj, Potato)",
        "(slice_obj, Tomato)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, LettuceSliced)",
        "(put_in, LettuceSliced, Bowl)",
        "(pick_up, PotatoSliced)",
        "(put_in, PotatoSliced, Bowl)",
        "(pick_up, TomatoSliced)",
        "(put_in, TomatoSliced, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Dropped Bowl",
"约束条件输出"
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

---

## [51] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Pan)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Pan, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Wrong plan - robot puts pan instead of bowl in the fridge",
"约束条件输出"
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

---

## [52] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Missing (pick_up, Egg)",
"约束条件输出"
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

---

## [53] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "lettuce is blocking the egg",
"约束条件输出"
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

---

## [54] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Pan)",
        "(put_in, Egg, Pan)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Pan)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Pan, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "None",
"约束条件输出"
         (无约束检查输出)

---

## [55] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Container)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Ambiguous plan - says some container (maps to pan)",
"约束条件输出"
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

---

## [56] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Wrong execution - policy puts egg in pan instead of bowl",
"约束条件输出"
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

---

## [57] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Wrong perception - potato detected as egg",
"约束条件输出"
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

---

## [58] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Missing step of open fridge before pick up the egg",
"约束条件输出"
         检查动作 1/10: (pick_up, Egg)
         检查动作 3/10: (put_in, Egg, Bowl)
         ❌ Postcondition 违反 (窗口 3-10 内未满足): Egg must be inside Bowl
         检查动作 7/10: (pick_up, Bowl)
         检查动作 9/10: (put_in, Bowl, Fridge)
         ❌ Precondition 违反: Fridge must be empty
         ❌ Postcondition 违反 (窗口 9-16 内未满足): Bowl must be inside Fridge
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [59] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Dropped Bowl",
"约束条件输出"
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

---

## [60] Store an egg in a bowl in the fridge

"任务名称": "Store an egg in a bowl in the fridge",
"原计划动作": [
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(pick_up, Egg)",
        "(navigate_to_obj, CounterTop)",
        "(put_on, Egg, CounterTop)",
        "(navigate_to_obj, Fridge)",
        "(close_obj, Fridge)",
        "(navigate_to_obj, Egg)",
        "(pick_up, Egg)",
        "(navigate_to_obj, Bowl)",
        "(put_in, Egg, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(open_obj, Fridge)",
        "(navigate_to_obj, Bowl)",
        "(pick_up, Bowl)",
        "(navigate_to_obj, Fridge)",
        "(put_in, Bowl, Fridge)",
        "(close_obj, Fridge)"
      ],
"真实原因": "Failed to successfully execute (open_obj, Fridge)",
"约束条件输出"
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

---

## [61] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(toggle_on, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_off, Television)",
        "(put_on, RemoteControl, TVStand)"
      ],
"真实原因": "Wrong Plan: TV switched off and laptop turned on",
"约束条件输出"
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

---

## [62] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(navigate_to_obj, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)"
      ],
"真实原因": "Missing (pick_up, RemoteControl)",
"约束条件输出"
         检查动作 3/8: (pick_up, Laptop)
         检查动作 5/8: (put_on, Laptop, TVStand)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 8/8: (toggle_on, Television)
         ❌ Postcondition 违反 (窗口 8-12 内未满足): Television must be toggled on
         ✅ 检测完成: 2 个违反, 2 个真实错误, 5 个跳过约束

---

## [63] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(put_on, RemoteControl, TVStand)"
      ],
"真实原因": "Wrong Order: of pick up remote control and toggle on television",
"约束条件输出"
         检查动作 3/10: (pick_up, Laptop)
         检查动作 5/10: (put_on, Laptop, TVStand)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/10: (toggle_on, Television)
         ❌ Postcondition 违反 (窗口 7-11 内未满足): Television must be toggled on
         检查动作 9/10: (pick_up, RemoteControl)
         检查动作 10/10: (put_on, RemoteControl, TVStand)
         ❌ Postcondition 违反 (窗口 10-17 内未满足): RemoteControl must be on top of TVStand
         ✅ 检测完成: 3 个违反, 3 个真实错误, 9 个跳过约束

---

## [64] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(open_obj, Laptop)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)",
        "(put_on, RemoteControl, TVStand)"
      ],
"真实原因": "Wrong Plan: Opens the laptop again",
"约束条件输出"
         检查动作 3/11: (pick_up, Laptop)
         检查动作 5/11: (put_on, Laptop, TVStand)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 8/11: (pick_up, RemoteControl)
         检查动作 10/11: (toggle_on, Television)
         ❌ Postcondition 违反 (窗口 10-14 内未满足): Television must be toggled on
         检查动作 11/11: (put_on, RemoteControl, TVStand)
         ❌ Postcondition 违反 (窗口 11-18 内未满足): RemoteControl must be on top of TVStand
         ✅ 检测完成: 3 个违反, 3 个真实错误, 9 个跳过约束

---

## [65] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, FloorLamp)",
        "(toggle_on, FloorLamp)"
      ],
"真实原因": "Wrong Plan: Floorlamp turned on instead of television",
"约束条件输出"
         检查动作 3/9: (pick_up, Laptop)
         检查动作 5/9: (put_on, Laptop, TVStand)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/9: (pick_up, RemoteControl)
         检查动作 9/9: (toggle_on, FloorLamp)
         ✅ Postcondition 满足 (在 帧 9 (窗口 9-13) 满足): FloorLamp must be toggled on
         ✅ 检测完成: 1 个违反, 1 个真实错误, 8 个跳过约束

---

## [66] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, GarbageCan)",
        "(put_in, Laptop, GarbageCan)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)",
        "(put_on, RemoteControl, TVStand)"
      ],
"真实原因": "Wrong Plan: Laptop is put in garbage can instead of TV stand",
"约束条件输出"
         检查动作 3/10: (pick_up, Laptop)
         检查动作 5/10: (put_in, Laptop, GarbageCan)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be inside GarbageCan
         检查动作 7/10: (pick_up, RemoteControl)
         检查动作 9/10: (toggle_on, Television)
         ❌ Postcondition 违反 (窗口 9-13 内未满足): Television must be toggled on
         检查动作 10/10: (put_on, RemoteControl, TVStand)
         ❌ Postcondition 违反 (窗口 10-17 内未满足): RemoteControl must be on top of TVStand
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [67] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_in, Laptop, TVStand)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)",
        "(put_on, RemoteControl, TVStand)"
      ],
"真实原因": "Book is blocking remote control",
"约束条件输出"
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

---

## [68] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)",
        "(put_on, RemoteControl, TVStand)"
      ],
"真实原因": "Miss step of close laptop",
"约束条件输出"
         检查动作 2/9: (pick_up, Laptop)
         检查动作 4/9: (put_on, Laptop, TVStand)
         ❌ Postcondition 违反 (窗口 4-11 内未满足): Laptop must be on top of TVStand
         检查动作 6/9: (pick_up, RemoteControl)
         检查动作 8/9: (toggle_on, Television)
         ❌ Postcondition 违反 (窗口 8-12 内未满足): Television must be toggled on
         检查动作 9/9: (put_on, RemoteControl, TVStand)
         ❌ Postcondition 违反 (窗口 9-16 内未满足): RemoteControl must be on top of TVStand
         ✅ 检测完成: 3 个违反, 3 个真实错误, 9 个跳过约束

---

## [69] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)"
      ],
"真实原因": "Dropped RemoteControl",
"约束条件输出"
         检查动作 3/9: (pick_up, Laptop)
         检查动作 5/9: (put_on, Laptop, TVStand)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/9: (pick_up, RemoteControl)
         检查动作 9/9: (toggle_on, Television)
         ❌ Postcondition 违反 (窗口 9-13 内未满足): Television must be toggled on
         ✅ 检测完成: 2 个违反, 2 个真实错误, 8 个跳过约束

---

## [70] Close the laptop and put it on the TV stand and switch on television

"任务名称": "Close the laptop and put it on the TV stand and switch on television",
"原计划动作": [
        "(navigate_to_obj, Laptop)",
        "(close_obj, Laptop)",
        "(pick_up, Laptop)",
        "(navigate_to_obj, TVStand)",
        "(put_on, Laptop, TVStand)",
        "(navigate_to_obj, RemoteControl)",
        "(pick_up, RemoteControl)",
        "(navigate_to_obj, Television)",
        "(toggle_on, Television)"
      ],
"真实原因": "Failed to successfully execute (put_on, Laptop, TVStand)",
"约束条件输出"
         检查动作 3/9: (pick_up, Laptop)
         检查动作 5/9: (put_on, Laptop, TVStand)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Laptop must be on top of TVStand
         检查动作 7/9: (pick_up, RemoteControl)
         检查动作 9/9: (toggle_on, Television)
         ❌ Postcondition 违反 (窗口 9-13 内未满足): Television must be toggled on
         ✅ 检测完成: 2 个违反, 2 个真实错误, 8 个跳过约束

---

## [71] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "Dropped BreadSliced",
"约束条件输出"
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [72] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(put_on, Knife, CounterTop)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot forgot to pick up the bread slice from the countertop before moving to the toaster.",
"约束条件输出"
         检查动作 2/8: (pick_up, Knife)
         检查动作 5/8: (put_on, Knife, CounterTop)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 7/8: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 7-14 内未满足): BreadSliced must be inside Toaster
         检查动作 8/8: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 8-12 内未满足): Toaster must be toggled on
         ✅ 检测完成: 3 个违反, 3 个真实错误, 7 个跳过约束

---

## [73] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot cannot pick up knife due to the pot occluding the knife.",
"约束条件输出"
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [74] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot cannot put bread slice inside the toaster due to a cellphone on top of the toaster, blocking the slots.",
"约束条件输出"
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

---

## [75] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot failed to execute the slice bread action.",
"约束条件输出"
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [76] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot failed to put bread slice inside toaster.",
"约束条件输出"
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 8-15 内未满足): BreadSliced must be inside Toaster
         检查动作 9/9: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 9-13 内未满足): Toaster must be toggled on
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [77] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot never picked up a knife and thus cannot slice the bread.",
"约束条件输出"
         检查动作 4/7: (pick_up, BreadSliced)
         检查动作 6/7: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 6-13 内未满足): BreadSliced must be inside Toaster
         检查动作 7/7: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 7-11 内未满足): Toaster must be toggled on
         ✅ 检测完成: 2 个违反, 2 个真实错误, 6 个跳过约束

---

## [78] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Bread)",
        "(pick_up, Bread)",
        "(navigate_to_obj, Toaster)",
        "(put_in, Bread, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot never sliced the bread, and the unsliced bread cannot be put inside the toaster.",
"约束条件输出"
         检查动作 2/5: (pick_up, Bread)
         检查动作 4/5: (put_in, Bread, Toaster)
         ❌ Postcondition 违反 (窗口 4-11 内未满足): Bread must be inside Toaster
         检查动作 5/5: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 5-9 内未满足): Toaster must be toggled on
         ✅ 检测完成: 2 个违反, 2 个真实错误, 6 个跳过约束

---

## [79] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(put_in, BreadSliced, Toaster)",
        "(toggle_on, Toaster)"
      ],
"真实原因": "The robot did not put the knife down after slicing the bread. As a result, the gripper was occupied by the knife and it could not pick up the bread slice.",
"约束条件输出"
         检查动作 2/8: (pick_up, Knife)
         检查动作 5/8: (pick_up, BreadSliced)
         检查动作 7/8: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 7-14 内未满足): BreadSliced must be inside Toaster
         检查动作 8/8: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 8-12 内未满足): Toaster must be toggled on
         ✅ 检测完成: 2 个违反, 2 个真实错误, 9 个跳过约束

---

## [80] toast bread

"任务名称": "toast bread",
"原计划动作": [
        "(navigate_to_obj, Knife)",
        "(pick_up, Knife)",
        "(navigate_to_obj, Bread)",
        "(slice_obj, Bread)",
        "(put_on, Knife, CounterTop)",
        "(pick_up, BreadSliced)",
        "(navigate_to_obj, Toaster)",
        "(toggle_on, Toaster)",
        "(put_in, BreadSliced, Toaster)"
      ],
"真实原因": "The robot toggled on the toaster before trying to put the bread slice inside the toaster. As a result, the bread slice ended up on top of the toaster instead of inside it.",
"约束条件输出"
         检查动作 2/9: (pick_up, Knife)
         检查动作 5/9: (put_on, Knife, CounterTop)
         ❌ Postcondition 违反 (窗口 5-12 内未满足): Knife must be on top of CounterTop
         检查动作 6/9: (pick_up, BreadSliced)
         检查动作 8/9: (toggle_on, Toaster)
         ❌ Postcondition 违反 (窗口 8-12 内未满足): Toaster must be toggled on
         检查动作 9/9: (put_in, BreadSliced, Toaster)
         ❌ Postcondition 违反 (窗口 9-16 内未满足): BreadSliced must be inside Toaster
         ✅ 检测完成: 3 个违反, 3 个真实错误, 10 个跳过约束

---

## [81] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(toggle_on, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Mug, DiningTable)"
      ],
"真实原因": "Wrong order of toggle on microwave and close microwave",
"约束条件输出"
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

---

## [82] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Mug, DiningTable)"
      ],
"真实原因": "Failed to successfully execute (toggle_on, Faucet)",
"约束条件输出"
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

---

## [83] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Microwave)",
        "(open_obj, Microwave)",
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Mug, DiningTable)"
      ],
"真实原因": "Wrong order of filling water and microwaving",
"约束条件输出"
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

---

## [84] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)"
      ],
"真实原因": "Incomplete Plan: missed steps for serving the mug",
"约束条件输出"
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

---

## [85] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(pick_up, Cup)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Cup, DiningTable)"
      ],
"真实原因": "Wrong Plan: serves glass instead of the mug with water",
"约束条件输出"
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

---

## [86] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Cup)",
        "(pick_up, Cup)",
        "(navigate_to_obj, Sink)",
        "(put_in, Cup, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Cup)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Cup, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Cup)",
        "(put_in, Cup, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Cup)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Cup, DiningTable)"
      ],
"真实原因": "Microwave already occupied",
"约束条件输出"
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

---

## [87] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Mug, DiningTable)"
      ],
"真实原因": "Wrong perception: glass and mug",
"约束条件输出"
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

---

## [88] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Cup)",
        "(put_in, Cup, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Mug, DiningTable)"
      ],
"真实原因": "Wrong Plan: puts cup inside microwave instead of mug filled with water",
"约束条件输出"
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

---

## [89] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Mug, DiningTable)"
      ],
"真实原因": "Missing step to pour wine out of the mug",
"约束条件输出"
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

---

## [90] Serve a glass of warm water at the dining table

"任务名称": "Serve a glass of warm water at the dining table",
"原计划动作": [
        "(navigate_to_obj, Mug)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(put_in, Mug, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Microwave)",
        "(put_on, Mug, CounterTop)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(put_in, Mug, Microwave)",
        "(close_obj, Microwave)",
        "(toggle_on, Microwave)",
        "(toggle_off, Microwave)",
        "(open_obj, Microwave)",
        "(pick_up, Mug)",
        "(navigate_to_obj, Sink)",
        "(pour, Mug, Sink)",
        "(navigate_to_obj, DiningTable)",
        "(put_on, Mug, DiningTable)"
      ],
"真实原因": "Wrong Plan: Pours water from mug after microwaving and before serving",
"约束条件输出"
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

---

## [91] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Incomplete Plan - Pot is filled with wine already",
"约束条件输出"
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

---

## [92] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Failed to successfully execute (pick_up, Pot)",
"约束条件输出"
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

---

## [93] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_on, Pot, CounterTop)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Wrong Plan - pot is put on countertop instead of inside sink",
"约束条件输出"
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

---

## [94] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(pour, Pot, Sink)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Wrong Plan - wrong step: (pour, Pot, Sink)",
"约束条件输出"
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

---

## [95] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(put_in, Pot, Sink)",
        "(pick_up, Pot)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Wrong order - faucet is turned on and off before pot is put in the sink",
"约束条件输出"
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

---

## [96] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Kettle)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Kettle, HousePlant)"
      ],
"真实原因": "Wrong plan - kettle is picked up instead of pot",
"约束条件输出"
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

---

## [97] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Container)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Bowl, HousePlant)"
      ],
"真实原因": "Ambiguous plan - some container",
"约束条件输出"
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

---

## [98] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Pan is blocking the pot",
"约束条件输出"
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

---

## [99] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Failed to successfully execute (toggle_on, Faucet)",
"约束条件输出"
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

---

## [100] water the plant

"任务名称": "water the plant",
"原计划动作": [
        "(navigate_to_obj, Pot)",
        "(pick_up, Pot)",
        "(navigate_to_obj, Sink)",
        "(put_in, Pot, Sink)",
        "(toggle_on, Faucet)",
        "(toggle_off, Faucet)",
        "(pick_up, Pot)",
        "(navigate_to_obj, HousePlant)",
        "(pour, Pot, HousePlant)"
      ],
"真实原因": "Dropped Pot",
"约束条件输出"
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

---
