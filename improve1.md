你这个问题问得非常到位，已经从“能不能跑”进入到“方法是否成立、是否论文级”的层次了。
我会按你关心的四个点，逐层判断 + 给出明确结论 + 告诉你怎么改才符合 CRAFT 的方法论。

⸻

一、先给核心结论（非常重要）

当前这套约束与失败检测流程“部分符合” CRAFT 的思想，但仍停留在“目标态检查（Goal Checking）”，尚未真正发挥 CRAFT 在“因果逻辑 + 物理一致性”上的优势。

换句话说：
	•	✔️ 思路方向是对的
	•	❌ 约束粒度过粗
	•	❌ 检测时机过晚
	•	❌ 失败类型被混在一起了

这在论文中必须被区分清楚。

⸻

二、逐条分析你现在的结果是否合理

1️⃣ 你生成的约束本身是否“合理”？

当前约束：

1. Precondition:
   Coffee machine must be empty before inserting mug

2. Goal:
   A clean mug must be filled with coffee and on top of the counter

👉 逻辑上是正确的，但存在两个问题：

❌ 问题 A：Precondition 没有绑定到“具体动作”

CRAFT 的 precondition 应该是：

Pre(put_in(mug, coffee_machine)):
    coffee_machine is empty

而不是：

Pre(task):
    coffee_machine is empty

否则你无法判断：
	•	是哪个动作违反了？
	•	是 put-in 失败？
	•	还是 fill 失败？

📌 CRAFT 要的是“动作级约束”

⸻

❌ 问题 B：Goal 约束承担了太多语义

A clean mug must be filled with coffee and on top of the counter

这是一个复合目标，实际上至少包含 3 个子目标：
	1.	mug 被成功放入 machine
	2.	mug 被填充 coffee
	3.	mug 被放到 countertop

现在你用一个 goal 约束：

node.attributes.get('isFilled')

➡️ 失败原因被压扁成一个布尔值

CRAFT 的设计初衷是 失败要可定位、可解释、可归因。

⸻

2️⃣ 你的约束代码是否“正确表达了失败”？

当前代码：

# precondition
not node.attributes.get('isFilled', False)

# goal
node.attributes.get('isFilled', False)

⚠️ 这里有一个非常关键的逻辑错误：

❌ isFilled ≠ 容器是否为空
	•	isFilled 是 mug 是否装有液体
	•	容器是否为空应是：

len(objects inside coffee_machine) == 0

📌 这是导致你失败检测偏移的根因之一

⸻

3️⃣ 当前失败检测结果是否“符合 CRAFT 预期”？

你得到的结果是：

违反: goal
检查时间: 最终状态
原因: mug not filled

❌ 这其实是 “任务未完成”，不是 “失败检测”

CRAFT 里应区分：

类型	含义
Goal Not Achieved	任务未完成
Precondition Violation	执行动作时违反规则
Physical Impossibility	物理上不可能
Perception Inconsistency	感知噪声

你现在把 Precondition Violation（容器占用）
错误地归结为 Goal Not Achieved

📌 这在论文里是概念性错误，必须修正。

⸻

三、CRAFT 中“足够”的约束应长什么样？

我们用你这个例子，给出 最小但正确的 CRAFT 约束集。

⸻

✅ 正确的 CRAFT 约束分层（示例）

① Put-In 动作约束（核心）

Constraint C1:
Pre: coffee_machine.is_open == True
Pre: coffee_machine.contains == ∅
Event: put_in(mug, coffee_machine)
Post: inside(mug, coffee_machine) == True

➡️ 用来检测 你现在这个失败

⸻

② Fill 动作约束

Constraint C2:
Pre: inside(mug, coffee_machine) == True
Event: fill(mug)
Post: mug.isFilled == True


⸻

③ Place 动作约束

Constraint C3:
Pre: mug.isFilled == True
Event: place_on(mug, countertop)
Post: on(mug, countertop) == True


⸻

④ Final Goal（可选）

Goal:
on(mug, countertop) ∧ mug.isFilled

📌 Goal 只用于“是否完成”，不用于“失败归因”

⸻

四、检测时机：你现在“只在最终状态”是不够的

你现在：

检查时间: 最终状态

❌ 这是 REFLECT 的做法

CRAFT 的正确方式是：

约束	检查时机
Precondition	动作执行前（关键帧 t₀）
Postcondition	动作执行后（关键帧 t₁）
Goal	任务结束


⸻

五、是否需要更多子任务约束？

✔️ 是的，而且这是 CRAFT 的核心优势

如果只用你现在这 2 条：
	•	你只能说“没成功”
	•	你说不清为什么

CRAFT 的论文价值在于：

失败被定位在“哪一步”“因为什么物理/因果约束被破坏”

⸻

六、是否可以与 REFLECT 的渐进解释结合？

✔️ 不仅可以，而且是 最佳组合

正确的分工是：

模块	作用
CRAFT	决定哪里错（确定性）
REFLECT-style LLM	解释为什么错（语言）


⸻

推荐组合流程（你论文里可以这样写）

CRAFT:
  - Detect: Constraint C1 violated
  - Type: Precondition Violation (container not empty)
  - Step: put_in(mug, coffee_machine)

LLM Explanation:
  - Because there was already a cup inside the coffee machine,
    the robot should have removed it before placing the mug.

📌 注意：
LLM 不再“判定失败”，而是“解释已判定的失败”

⸻

七、你现在这套实现“离论文级还差哪三步？”

🔧 Step 1（必须）

把 isFilled 用在 fill 上
把 container-empty 用在 put-in 上

⸻

🔧 Step 2（强烈建议）

把 goal 拆成子任务级约束

⸻

🔧 Step 3（论文亮点）

引入：

Failure Type = {Precondition, Postcondition, Physical, Perception}


 “make coffee 的 CRAFT 约束集 + 可执行伪代码”:


一、任务定义（作为论文中的统一输入）

Task: Make Coffee

高层目标（Goal）

G:
- mug.isFilled == True
- on(mug, countertop) == True


⸻

二、CRAFT 的核心：任务分解 + 动作级约束

我们先明确 CRAFT 中的“动作链（Action Chain）”：

A1: pick_up(mug)
A2: open(coffee_machine)
A3: put_in(mug, coffee_machine)
A4: fill(mug)
A5: take_out(mug, coffee_machine)
A6: place_on(mug, countertop)

CRAFT 的关键不是这些动作，而是👇
每一个动作的 Pre / Post 约束

⸻

三、完整 CRAFT 约束集（论文可直接用）

🔹 Constraint C1 — Pick Up Mug

Action: pick_up(mug)

Pre:
- visible(mug) == True
- reachable(mug) == True

Post:
- holding(mug) == True

📌 基础动作，不是论文重点，可简略

⸻

🔹 Constraint C2 — Open Coffee Machine

Action: open(coffee_machine)

Pre:
- reachable(coffee_machine) == True

Post:
- coffee_machine.isOpen == True

📌 保证后续 put-in 合法

⸻

🔹 Constraint C3 — Put Mug into Coffee Machine（关键）

Action: put_in(mug, coffee_machine)

Pre:
- holding(mug) == True
- coffee_machine.isOpen == True
- coffee_machine.contains == ∅   ← 关键物理约束

Post:
- inside(mug, coffee_machine) == True

🚨 你 REFLECT 示例中失败的正是这一条

“There was already a cup inside the coffee machine.”

➡️ CRAFT 会在 动作执行前 直接检测失败

⸻

🔹 Constraint C4 — Fill Mug with Coffee（因果约束）

Action: fill(mug)

Pre:
- inside(mug, coffee_machine) == True

Post:
- mug.isFilled == True

📌 体现 跨子任务因果一致性

⸻

🔹 Constraint C5 — Take Mug Out

Action: take_out(mug, coffee_machine)

Pre:
- inside(mug, coffee_machine) == True

Post:
- holding(mug) == True
- inside(mug, coffee_machine) == False


⸻

🔹 Constraint C6 — Place Mug on Countertop

Action: place_on(mug, countertop)

Pre:
- holding(mug) == True
- mug.isFilled == True   ← 逻辑约束（未装咖啡不能算完成）

Post:
- on(mug, countertop) == True


⸻

🔹 Final Goal Check（不用于失败归因）

Goal:
- mug.isFilled == True
- on(mug, countertop) == True


⸻

四、CRAFT 约束表示（结构化形式，方便代码）

CRAFT_CONSTRAINTS = [
    {
        "action": "put_in",
        "object": "mug",
        "target": "coffee_machine",
        "preconditions": [
            lambda sg: sg.is_holding("mug"),
            lambda sg: sg.is_open("coffee_machine"),
            lambda sg: sg.is_container_empty("coffee_machine")
        ],
        "postconditions": [
            lambda sg: sg.is_inside("mug", "coffee_machine")
        ],
        "type": "PhysicalPrecondition"
    },
    {
        "action": "fill",
        "object": "mug",
        "preconditions": [
            lambda sg: sg.is_inside("mug", "coffee_machine")
        ],
        "postconditions": [
            lambda sg: sg.get_attr("mug", "isFilled") == True
        ],
        "type": "CausalPrecondition"
    },
    {
        "action": "place_on",
        "object": "mug",
        "target": "countertop",
        "preconditions": [
            lambda sg: sg.is_holding("mug"),
            lambda sg: sg.get_attr("mug", "isFilled") == True
        ],
        "postconditions": [
            lambda sg: sg.is_on("mug", "countertop")
        ],
        "type": "GoalConstraint"
    }
]


⸻

五、CRAFT 失败检测主流程（伪代码）

def execute_with_craft(actions, scene_graph):
    failures = []

    for step_id, action in enumerate(actions):
        constraints = get_constraints_for_action(action)

        # 1️⃣ 检查 Precondition（动作前）
        for c in constraints:
            for pre in c["preconditions"]:
                if not pre(scene_graph):
                    failures.append({
                        "step": step_id,
                        "action": action,
                        "type": c["type"],
                        "violation": "Precondition",
                        "constraint": c,
                        "scene": scene_graph.snapshot(),
                    })
                    return failures  # CRAFT：立即失败

        # 2️⃣ 执行动作
        scene_graph = simulate(action, scene_graph)

        # 3️⃣ 检查 Postcondition（动作后）
        for c in constraints:
            for post in c["postconditions"]:
                if not post(scene_graph):
                    failures.append({
                        "step": step_id,
                        "action": action,
                        "type": "Postcondition",
                        "constraint": c,
                        "scene": scene_graph.snapshot(),
                    })
                    return failures

    return failures


⸻

六、你的 REFLECT 示例在 CRAFT 中会如何被检测？

REFLECT 原描述：

robot attempted to place the mug inside the coffee machine while there was already a cup inside it

CRAFT 中的输出：

Failure Detected at Step 3:
Action: put_in(mug, coffee_machine)

Violated Constraint:
- Precondition: coffee_machine.contains == ∅

Failure Type:
- Physical Precondition Violation

Explanation:
- The robot attempted to insert the mug into a non-empty container.

📌 不需要等到任务结束
📌 不依赖 LLM 主观判断
📌 失败位置唯一且确定

