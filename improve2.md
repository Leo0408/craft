这是一个非常好的问题，而且你现在已经在做一件很多工作都会“差一口气”的关键事情：
👉 把“理念正确”变成“判定语义严格一致”。

我先给结论，然后逐条解释你现在的结果哪里是对的、哪里还差一点、如何最小修改就完全对齐 CRAFT 的设计目标。

⸻

一句话结论（先给你定心丸）

现在的结果「在理念上是对的」，但「在结构上还不完全满足 CRAFT 的最优形式」。

更具体地说：
	•	✅ 你已经正确引入了“物理前置条件 + 目标约束”的二层判定
	•	⚠️ 但目前这两个约束还是“任务级的、脱离动作绑定的”
	•	❌ 还缺少 CRAFT 的一个关键点：约束 → 动作 → 时序 的绑定

下面我按 CRAFT 设计目标 来逐项对照你现在的实现。

⸻

一、你现在做对了什么（非常重要）

✅ 1. Precondition 的形式是“正确的 CRAFT 形式”

你现在的 precondition：

len([
  e for e in scene_graph.edges.values()
  if e.end.name == node.name and e.edge_type == 'inside'
]) == 0

这是 非常标准的 CRAFT-style 物理约束：

维度	是否满足
非语言	✅ 是
基于场景图	✅ 是
确定性	✅ 是
可复现	✅ 是
不依赖 LLM	✅ 是

这已经完全区别于 REFLECT 的“LLM 觉得里面空不空”。

✔️ 这一点在论文中可以非常明确地强调。

⸻

✅ 2. 你正确地区分了「Precondition Violation」和「Goal Not Achieved」

你的失败检测输出：

1. [PRECONDITION] Coffee machine must be empty before inserting mug
2. [GOAL] A clean mug is filled with coffee and on top of the countertop

这已经体现出 CRAFT 的失败类型层级：
	•	Precondition Violation → 动作在物理上不合法
	•	Goal Not Achieved → 任务未完成（但不是原因）

这是 REFLECT 明确做不到的。

⸻

✅ 3. Precondition 在“动作前”检查（这是核心）

检查时间: 初始状态（动作前）

这一点非常重要：

CRAFT 的 Precondition 必须在动作执行前检查

你现在这一点是 完全正确的。

⸻

二、现在还“差哪一口气”（这是关键）

问题集中在这几行：

绑定动作: Step ? - None

❌ 问题 1：约束没有绑定到“具体动作”

在 CRAFT 的严格定义里：

一个 Precondition 必须属于一个 Action

但你现在是：
	•	有 precondition
	•	有 goal
	•	但它们是“悬空的约束”

这会带来 3 个后果：
	1.	❌ 无法定位「是哪个动作导致的失败」
	2.	❌ 无法做精确的因果解释
	3.	❌ 与 REFLECT 的“任务级失败检测”在形式上仍然太接近

⸻

❌ 问题 2：Goal 约束“掩盖”了真正失败的原因

你现在的输出是：

发现 2 个违反的约束

但在 严格的 CRAFT 逻辑中：

如果某个 Precondition 在动作前失败
👉 后续动作 & Goal 不应该再作为“失败原因”出现

否则会出现：

“你既告诉我任务一开始就不合法
又告诉我最后没完成任务”

这在逻辑上是 冗余甚至误导的。

⸻

三、如何最小修改，就 100% 对齐 CRAFT

✅ 修改 1：给每个约束绑定 Action

你不需要大改，只需要结构化一点：

constraint = {
    "type": "PRECONDITION",
    "action": "put_in",
    "object": "mug",
    "target": "coffee_machine",
    "check": lambda sg: sg.is_container_empty("coffee_machine")
}

然后你的失败输出就会变成：

[PRECONDITION VIOLATION]
Action: put_in(mug, coffee_machine)
Step: 3
Reason: coffee_machine is not empty

📌 这一步是 CRAFT 和你当前版本的分水岭

⸻

✅ 修改 2：Precondition 失败 → Goal 不再参与失败统计

逻辑应为：

if precondition_failed:
    report(precondition_failure)
    skip_goal_evaluation()

最终失败报告应当是：

发现 1 个违反的约束:

1. [PRECONDITION]
   Action: put_in(mug, coffee_machine)
   Failure Type: Physical Precondition Violation
   Time: Before Action Execution

Goal 只用于 成功判定，而不是失败溯因。

⸻

四、对你现在结果的“正式评估”

是否符合你前面优化后的 CRAFT 概念？

维度	结论
引入可执行物理约束	✅
区分 precondition / goal	✅
动作前检查	✅
失败类型可解释	⚠️ 部分
动作级绑定	❌
因果链定位	❌

总体评价：

你已经完成了 CRAFT 的 70–80% 正确实现
剩下的是“结构性严谨度”，不是方法方向问题

⸻

五、论文中你可以这样写（直接可用）

In our current implementation, CRAFT distinguishes physical precondition violations from goal failures. However, we observe that preconditions must be explicitly bound to action-level execution in order to enable precise failure localization and causal tracing. We therefore refine the framework to associate each constraint with a specific action and execution step, ensuring that precondition violations are detected prior to action execution and prevent downstream goal evaluation.

