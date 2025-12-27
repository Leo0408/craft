

CRAFT Action-aware Constraint Generation

当前问题总结 & 优化方案（给 Cursor 用）

⸻

一、当前系统状态（结论先行）

当前 CRAFT 的 Action-aware Constraint Generation 在以下方面是正确的：
	•	✅ 约束语义模板设计正确（pick_up / put_on / put_in / toggle）
	•	✅ LLM 能按模板生成结构化 JSON 约束
	•	✅ 约束具备 Pre / Post 区分
	•	✅ 输出格式和解析流程稳定

但存在系统性错误：

约束内容本身大多是对的，但约束被绑定到了错误的动作步骤上，导致时序语义混乱。

⸻

二、核心问题定位（Root Causes）

❌ 问题 1：使用 final_scene_graph 生成所有动作约束

当前行为

generate_constraints(scene_graph=final_sg, actions=full_action_list)

问题
	•	final_sg = 任务完成后的“未来世界”
	•	LLM 在生成第 1～N 步动作的约束时不可避免“偷看未来”
	•	导致：
	•	pick_up 绑定了最终 on/inside 状态
	•	put_in 绑定了 toggle_on / toggle_off 的结果
	•	前后置条件严重错位

本质问题

约束生成违反了因果顺序（causality violation）

⸻

❌ 问题 2：约束生成是“任务级”，但目标是“动作级”

当前设计
	•	一次 Prompt 覆盖整个任务 + 全动作序列
	•	LLM 被要求生成 ENTIRE execution flow 的约束

问题
	•	LLM 会自动补全跨动作的因果链
	•	无法保证每条约束只对应一个原子动作

症状
	•	同一个约束可能合理，但绑定错 action
	•	Action Binding 逻辑变得极其复杂且不稳定

⸻

❌ 问题 3：Action Binding 是“事后修补”，而非“生成即绑定”

当前方式
	•	LLM 生成 → 再用字符串 / 对象 / 语义匹配回绑动作

问题
	•	Binding 逻辑本身不可靠
	•	越复杂越难保证正确性

本质

如果约束在生成时就知道它属于哪个 action，就不需要复杂绑定

⸻

⚠️ 问题 4：部分约束语义层级未区分（Hard / Soft）

例子：
	•	container_empty(CoffeeMachine)

问题
	•	在物理层面合理
	•	在任务语义中可能是“非关键条件”

影响
	•	所有约束默认 hard，失败检测不够鲁棒

⸻

三、推荐的整体优化方向（高层设计）

✅ 核心改动思想（一句话）

从“Task-level constraint generation”
→ 转为 “Action-centric constraint instantiation”

⸻

四、具体可执行优化方案（Cursor 可直接按此实现）

⸻

✅ 优化 1：改为「按动作生成约束」（最重要）

❌ 当前

Input: final_scene_graph + full action list
Output: all constraints

✅ 改为

For each action a_i:
    Input:
        - 当前 action a_i
        - action type (pick_up / put_in / ...)
        - minimal object context（可选）
    Output:
        - Pre(a_i)
        - Post(a_i)

效果
	•	每条约束天然绑定到 action_index = i
	•	不再需要复杂 Action Binding

⸻

✅ 优化 2：Prompt 改为 Action-local（不再全任务）

Prompt 输入应只包含：

Current Action: (put_in, Mug, CoffeeMachine)
Action Index: 9
Action Semantic Template:
    - Pre: holding(Mug), container_open(CoffeeMachine), container_empty(CoffeeMachine)
    - Post: inside(Mug, CoffeeMachine)

明确禁止
	•	引入后续动作
	•	引入最终 goal 状态

⸻

✅ 优化 3：Scene Graph 使用策略调整

原则
	•	❌ 不使用 final_sg 做动作约束生成
	•	✅ scene graph 只用于：
	•	校验（constraint checking）
	•	失败分析（why failed）
	•	物体属性补充（isOpen / isFilled）

推荐最小方案（先能跑）
	•	约束生成阶段：不依赖 scene graph
	•	约束验证阶段：用 scene graph

⸻

✅ 优化 4：删除或极简 Action Binding

优化后逻辑

constraint = {
    "type": "precondition",
    "template": "holding(Mug)",
    "action_index": i,
    "action": actions[i]
}

	•	不再需要字符串匹配
	•	不再需要语义回溯

⸻



五、makeCoffee 特例说明

当前问题示例
	•	(put_in, Mug, CoffeeMachine)
→ 没有区分：
	•	必须条件：holding(Mug)
	•	领域条件：container_empty(CoffeeMachine)


⸻

六、总结给 Cursor 的关键指令

	1.	不要用 final scene graph 生成动作级约束
	2.	每个 action 独立生成 Pre / Post
	3.	约束生成时就绑定 action_index
	4.	Scene graph 只用于验证，不用于生成
