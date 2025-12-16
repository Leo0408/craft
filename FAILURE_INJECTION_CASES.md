# 失败注入案例总结

本文档总结了从 REFLECT 论文中提取的真实场景失败检测问题，以及 CRAFT 框架如何改善这些问题。

## 一、REFLECT 真实场景失败案例

### 案例 1：视觉遮挡导致的误失败（Occlusion-induced False Failure）

**来源**：REFLECT 真实机器人实验（Kitchen / Household manipulation）

**现象**：
- 在抓取任务中（抓苹果/抓工具/拿杯子等），当机器人手臂靠近目标时，视觉检测会因为机械臂遮挡而短暂失效：
  - 探测框（BBox）消失
  - Depth 或 segmentation 缺失
  - 场景图不再包含该物体

**REFLECT 的错误判断**：
```
L1: "the apple is no longer visible"
↓
L2 Task Summary: "Robot failed to pick up the apple."
```
即使实际抓取成功，也会被判定失败。

**原因**：REFLECT 的判断直接依赖场景图的单帧感知

**CRAFT 改善方式**：
- **Environment Memory**：通过 Kalman/Bayesian filter 判断遮挡 ≠ true disappearance
- **轨迹预测**：物体位置合理但暂时不可见 → 不是失败
- **Constraint Code**：`check_pick_up` 在 PostCondition 中验证的是 gripper force / relative geometry，而不是"视觉中是否仍看到物体"

---

### 案例 2：检测抖动导致的成功漏检（Jitter-induced False Failure）

**来源**：REFLECT Real Robot 部分，特别是 kitchen tasks（锅、碗、炉灶、工具）

**现象**：
真实场景中存在：
- object bounding box 不稳定
- mask 抖动
- depth 不连续
- 同一物体在不同帧中位置变化异常（视觉抖动）
- contain/on/near 关系频繁错误

**REFLECT 的错误判断**：
- "物体移动过远 → 操作失败"
- "物体被放到错误位置"
- "contain 关系不存在 → put-in 失败"

但实际上任务成功，只是视觉结果跳动。

**CRAFT 改善方式**：
- **Environment Memory**：平滑 bbox/pose 曲线，判断跳变是否物理可能（瞬移 > 30cm → 感知错误）
- **Constraint Execution Layer**：contain/on/in 关系不是由单帧判断，使用体积、几何、接触关系，多帧稳定验证

---

### 案例 3：容器内部反射/遮挡导致的 contain 误判（False Containment）

**来源**：REFLECT 机器人演示中常见，特别是在放入抽屉/放入微波炉/放入锅里任务中

**现象**：
当机器人执行"put object into container"：
- 容器边沿反光导致 segmentation 出错
- 容器内部深度信息缺失
- 物体被判定 "not in container"
- LLM 层 → 解释为失败

但真实情况：已经放进去了，只是视觉不稳定。

**CRAFT 改善方式**：
- **SCG + 体积/碰撞约束**：使用容器 bounding volume，判断是否"physically inside"，不依赖视觉模型的 contain 分类
- **Memory**：multi-frame check：连续数帧 inside → 才确认，避免单帧错误导致误判

---

## 二、AI2THOR 可复现实验案例（6个场景）

### 案例 1：视觉遮挡 → REFLECT 误判失败（假失败）

**任务**：`PickUp(Apple)`

**真实执行**：
- 机械臂抓起苹果
- 机械臂遮挡视觉 → Apple 短暂不可见

**REFLECT 误判**：
```
"I cannot see apple" → "object was dropped" → 判定失败
```

**CRAFT 正确判定**：
- 利用 Memory：
  - `last_seen(apple) < 0.3 sec`
  - `apple.position_predicted` 在手爪内部
  - `occlusion predicted = True`
- → 成功（SATISFIED）

**预期输出**：
```
[Memory] apple temporarily occluded (0.45s)
[Constraint] post: holding(apple)==True → SATISFIED
```

---

### 案例 2：容器冲突（Inside Space Conflict）→ REFLECT 假成功

**任务**：`PutObject(Cup, Drawer)`

**初始状态**：
- Drawer closed
- 或 Drawer 内已经有 Plate 占位

**真实操作**：
- Cup 被放在 Drawer 前面，距离很近
- 视觉 summary 经常输出：`cup near drawer → maybe inside`

**REFLECT 误判**：
```
"cup is near drawer" → "put inside successful"
```

**CRAFT 正确判定**：
- Pre: `drawer.door == open`
- Post: `inside(cup, drawer) == True`
- Geometry Check: `cup volume ≠ inside drawer volume`
- → Failure: PostconditionViolation (not actually inside)

**预期输出**：
```
[Post] inside(cup, drawer): FALSE
[Geometry] cup does not intersect drawer volume
→ VIOLATED: cup not actually inside drawer
```

---

### 案例 3：动作因果链错误 → REFLECT 无法检测

**任务**：`Heat(Kettle)`

**正确流程**：
1. Put kettle under faucet
2. Fill(kettle)
3. PutOn(stove)
4. TurnOn(stove)

**真实错误**：
- 机器人跳过「Fill」步骤
- `kettle.hasWater = False`

**REFLECT 误判**：
```
"kettle is on stove." → "heating successful"
```

**CRAFT 判定**：
- Pre(heat): `kettle.has_water == True`
- → Failure: PreconditionViolation

**预期输出**：
```
Violation: cannot heat kettle with no water
```

---

### 案例 4：场景跳变（瞬移/teleport）→ REFLECT 容易 hallucinate

**任务**：`Move Mug from Table to Countertop`

**错误情况**：
- AI2THOR 偶尔会因 mesh 问题让 Mug 瞬间出现在远处
- 或因 segmentation 错误导致突然 bounding box 大跳

**REFLECT summary**：
```
"Mug is on countertop"（判断成功）
```

**但实际**：
- `mug.pos_t - mug.pos_(t-1) > 1.5m`
- impossible motion velocity
- 应视为感知错误

**CRAFT 检测**：
- Invariant: `object cannot teleport`
- Violation: `velocity spike detected`

**预期输出**：
```
[InvariantViolation] Mug teleported (>1.2m). Perception error.
```

---

### 案例 5：靠近 ≠ 放入（common THOR failure）

**任务**：`Place Apple into Microwave`

**错误情况**：
- 苹果被放到 Microwave 前面
- 但视觉检测经常把 bounding box overlap 判成 "inside"

**REFLECT**：
```
"apple is close to microwave → inside"
```

**CRAFT**：
- `geometry.is_inside(apple, microwave) == False`
- → PostconditionViolation

**预期输出**：
```
[Geometry] apple not inside microwave volume
→ FAILED
```

---

### 案例 6：环境状态模糊 → REFLECT 无法判断

**任务**：`Open Fridge`

**错误情况**：
- 视觉检测 jitter
- `fridgeState` 在连续帧 oscillate：open/closed/open

**REFLECT**：
```
"I see the door open" → 成功
```

**CRAFT**：
- Memory state smoothing：
  - `if open_state stable < 3 frames → uncertain`

**预期输出**：
```
[Memory] fridge state unstable (open/closed flip)
[Constraint] door must be open → UNCERTAIN (not SATISFIED)
```

---

## 三、失败类型对比表

| 失败类型 | REFLECT 会误判 | CRAFT++ 的优势 |
|---------|--------------|---------------|
| 遮挡造成假失败 | ✔ | Memory 识别遮挡 |
| 靠近造成假成功 | ✔ | 几何约束、体积检测 |
| 跳过关键步骤 | ✔ 因果链缺失 | Pre/Post 验证 |
| 物理不可能事件 | ✔ 不判断 | Invariant 检测 |
| 状态跳变/噪声 | ✔ hallucination | 状态平滑 |
| 容器冲突 | ✔ | 占用冲突检测 |

---

## 四、CRAFT 改进能力总结

CRAFT 通过"可执行场景约束代码 + 环境记忆"的组合，可显著改善 REFLECT 的失效情况：

1. **环境记忆（Environment Memory）**
   - 平滑感知噪声、检测跳变、避免短时遮挡导致物体"消失"

2. **可执行约束代码（Runtime Constraint Checking）**
   - 使用几何、物理和因果约束代替视觉模型不稳定的关系分类
   - 使"contain / on / in"等关键关系具有物理可验证性

3. **时序一致性（Temporal Consistency）**
   - 动作结果不再由单帧决定，而是基于多帧稳定判定

因此，上述 REFLECT 的三类真实失效案例，均能在 CRAFT 框架下获得显著改善。

