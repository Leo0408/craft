好——我已经基于你上传的 demo1_complete_workflow_with_keyframes.pdf 完整读过并开始分析。下面给出论文/工程级别的逐步分析：每一步应该做什么、demo 的实际输出是什么、与 CRAFT++ 预期是否一致、证据（来自 demo）以及明确的改进建议与可立刻落地的修复措施。我用小节清晰呈现，便于直接复制到论文/README/Issue 列表中。

我会引用 demo 中最关键的输出作为证据。 ￼ ￼ ￼ ￼ ￼

⸻

一、总体结论（1 段话）

Demo 实现了你设计的 CRAFT++ 大体流程（AI2THOR 数据 → Scene Graph → 约束生成 → 约束编译/执行 → 关键帧优化 → 输出），并展示了 Smart Keyframe 的收益（场景图数量下降 ~22%，验证结果一致）。但当前 demo 的若干实现细节尚未完全体现 CRAFT++ 的核心价值（可执行 AST、置信度聚合/不确定性处理、几何精检、Environment Memory 在真实噪声下的作用），需要做若干明确改进以保证论文结论成立（详见下文）。

⸻

二、逐步流程与 demo 输出（对照表）

下面把流程分成 8 步（与 notebook 步骤一致），对每步给出：预期、demo 实际输出、是否符合、证据。

Step 0 — Setup / Config

预期：导入模块、配置 LLM、路径等。
Demo 输出：环境路径设置与 imports 成功，见 notebook header 输出。符合预期。 ￼

⸻

Step 1 — Data generation (AI2THOR run)

预期：从 AI2THOR 得到事件序列 events_craft 与 action_results，并保存 ego frames。
Demo 输出：生成了 9 个事件、31 张帧并保存到 thor_tasks/.../ego_img，视频帧成功生成（fps=2），见导出日志。符合预期。

⸻

Step 2 — Video / frame extraction & annotation

预期：生成可视化视频（用于复现与审查）。
Demo 输出：成功生成 craft_ai2thor_workflow_simple.mp4（31 帧），并尝试生成包含 scene-graph 的完整视频（若 scene graphs 与 frames 数量匹配）。此处有一处 mismatch log（scene_graphs (9) vs frames (31)）说明 debug 路径需要注意。
问题点：frame/scene_graph 对齐有 mismatch（需要保证 1:1 mapping 或明确 keyframe 列表）。见改进建议。

⸻

Step 3 — Scene Graph Generation (Original)

预期：为每个关键事件生成结构化 scene_graph（nodes/edges/state/confidence）。
Demo 输出：生成 9 个 scene graphs；示例 scene graph 列出 nodes/edges（CoffeeMachine, Sink, CounterTop, Faucet, Mug），但节点 state 多为 N/A（未包含 object 状态信息），边信息以文字关系呈现（on_top_of, inside 等）。
是否符合：部分符合（关系与拓扑信息有），未体现两个关键字段：a) 对象状态（open/closed/filled/holding）缺失或 N/A；b) 原子置信度/confidence 值没有被保留到最终 scene_graph（或未打印）。这会影响后续置信度聚合与不确定性处理。

⸻

Step 3B — Smart Keyframe Selection（动态关键帧）

预期：根据 state_changed_significantly 或动作边界决定是否保留 scene graph（减少冗余评估）。
Demo 输出：Smart 方法把 scene graphs 从 9 减少到 7（22.2% 降低），并打印每个 keyframe 的 reason（Initial / State changed / Key action），符合预期。对比结果显示两种方法检测到的 violations 相同（0）并且 satisfied 都为3。
评价：Smart keyframe 已实现并能减少计算，但需要保证 keyframe 判定阈值可配置并与 Memory 结合（否则在有遮挡或短暂跳变时可能错过重要帧）。

⸻

Step 4 — Constraint Generation (LLM)

预期：LLM 生成结构化约束 JSON/AST（Pre/Post/Invariants）并附带 type 字段与最小置信度阈值。
Demo 输出：notebook 包含 “Constraint Generation” 步骤并展示最终有 3 个约束被验证（总数 3）。但在文档输出中看不到 LLM 输出的“结构化 AST / JSON”示例或约束的 type 标注（至少打印/记录未见）。 ￼
是否符合：实现了“生成约束”的流程，但缺少证据显示约束以可执行 AST/DSL形式存在（关键）。如果约束仅留文本并在后续通过文本匹配做检查，则不满足 CRAFT++ 的可执行性要求。

⸻

Step 5 — Constraint Code Generation（AST/DSL）

预期：将 LLM 输出编译为受控 AST 或小 DSL（并做静态校验 / 单元测试）。
Demo 输出：文档提到 Step 5：Constraint Code Generation (AST/DSL) 并宣称“Compile constraints to executable AST/DSL expressions”，但没有在输出中展示生成的 AST、执行器日志或静态检查信息。 ￼
是否符合：流程存在，但缺少可验证的 artifact；需要在 notebook 中打印或保存 AST、并运行少量 unit tests 来证明正确性。

⸻

Step 6 — Code-based Failure Detection (Constraint Execution)

预期：以 world_state（由 Memory 提供的平滑状态）作为输入，用执行器校验约束并返回 SATISFIED/VIOLATED/UNCERTAIN（含 confidence & reason）。
Demo 输出：最终 Smart Method validated 3 constraints, Violated: 0, Satisfied: 3（both methods一致）。但是执行器的决策依据（例如 atom-level confidences、geometry checks、memory usage）没有细粒度日志或返回字段展示；scene_graph 中 object states 多为 N/A（见上）。
是否符合：执行并输出了判断，但缺少“可执行性证据链”（即无法看到执行器是如何判断 inside/machine 的）。必须补充 atom-level trace 才能证明“deterministic code-based validation”。

⸻

Step 7 — Progressive Explanation

预期：若 violation，生成可追溯解释（which pre/post failed, which atom false）并给出 remedy。
Demo 输出：Notebook 有 Step 7: Progressive Explanation 标题，但在最终 demo run 没有展示任何 progressive explanation 示例（由于没有 violation）。仍需展示 explanation 格式与样例。

⸻

Step 8 — Final Comparison & Metrics

预期：统计 key metrics（violations, satisfied, frames, SG counts, reduction）并输出对比图表。
Demo 输出：已打印对比信息：Original: 9 SG, Smart: 7 SG, reduction 22.2%， constraints both methods same results (0 violations) 等总结。这是 demo 的亮点之一。

⸻

三、与 CRAFT++ 预期对齐度（高层结论）
	•	对齐且已实现：事件提取 → Scene Graph 生成 → Keyframe 策略 → 约束生成与校验流程的基本骨架；Smart keyframe 能有效减少场景图数。证据见关键打印与对比表。
	•	未完全对齐 / 需加强（影响论文结论可靠性的关键点）：
	1.	约束没有明确展示为可执行 AST/DSL 与其静态校验结果（需要打印 AST & unit-test），否则无法证明“可执行逻辑取代 LLM 探索式推理”的主张。
	2.	SceneGraph 未保留或未显示 object state/confidence & geometry 信息（nodes show state=N/A），这削弱了 geometry checks 与置信度聚合的能力。
	3.	Environment Memory 在此 demo 中被“简化”并未展示其处理遮挡与跳变的能力（文档明确说明 AI2THOR deterministic → memory 简化），导致无法以本 demo 证明 Memory 的贡献（需要在含噪声/遮挡的实验里演示）。
	4.	验证器没有打印 atom-level trace（原子谓词的 truth & confidence），这使得“deterministic且可解释的失败检测”论断无法经受审查。
	5.	frame/scene_graph 对齐问题（scene graphs 9 vs frames 31 的 mismatch），会影响视频可视化与结果重现性。

⸻

四、具体、可执行的优化建议（按优先级）

下面每条都是可立刻在 notebook/代码库中实现的改动，且对论文实验可信度提升明显。

优先级 A（必须立刻做，用于保证实验结论可信）
	1.	把 LLM 输出的约束保存为结构化 JSON + condition_expr（DSL）并把 AST 打印到日志，示例字段：{id,type,condition_expr,metadata.min_confidence,severity}。在 notebook 中展示 1–2 个例子。——理由：证明“可执行约束”存在。
	2.	实现并打印 eval_predicate_ast 的 atom-level trace：每个原子谓词（如 inside(mug, coffeeMachine)、machine.door == open）返回 {value, confidence, source}，并把汇总决定（aggregate_confidence）记录。示例输出格式：

[C1] inside(mug, machine): False (conf=0.97) -> VIOLATED (hard precondition)
atom traces: [is_inside_bbox: False, geometry_iou: 0.02, edge_relation_conf: 0.05]

——理由：可解释、可复现、便于审稿人与读者理解判定过程。

	3.	SceneGraph 增强：把 object.state（open/closed/filled/holding）、bbox/pose、confidence（float）写入 node 属性，并在生成/打印时展示。当前 Node state=N/A 必须修复。

优先级 B（强烈建议，增强论文说服力）
	4.	将约束编译器产出（AST）做静态校验与 unit-test：对每个生成的 AST 用小 mock world_state 运行 3 个 test（单原子真、单原子假、low-confidence case）确保执行器行为与预期一致。——理由：避免 LLM 生成 “反命题”或语法错误。
	5.	将 EnvironmentMemory 从“简化”提升为可选模块并在 demo 中演示：重复运行同一场景但人为注入两个噪声版本：A) camera occlusion during pick up；B) sudden teleport of mug（模拟 sensor glitch）。展示 Memory 如何从 UNCERTAIN 转为 SATISFIED。——理由：直接证明 Memory 的价值（当前 demo 没展示）。
	6.	改进 keyframe 对齐逻辑：保证 len(scene_graphs) == len(frames_for_scene_graphs) 或记录 mapping 列表 keyframe_indices 并用于视频 overlay（避免 mismatch 警告）。

优先级 C（可选，提升工程质量与论文实验覆盖）
	7.	增加 geometry module：实现 geometry.is_inside(obj, container)（使用 bbox/pose/volume）与 iou(bbox1,bbox2)；在 CheckContainerEmpty、CheckLocationConstraint 中用几何结果作为主判据并把边缘情况标为 UNCERTAIN。
	8.	添加置信度聚合策略与阈度调优实验：在论文中给出 ablation：min / mean / product 聚合对 false positive/negative 的影响。
	9.	改进 Progressive Explanation：当 VIOLATED 时输出 causal chain（哪个 pre/post fail，从哪个 atom 导致），并建议 remedy（reobserve/try reopen door/clear container）。

⸻

五、示例修复伪代码（可直接粘贴到 notebook）

A. eval_predicate_ast（简化实现）

def eval_predicate_ast(ast, world_state, memory):
    atom_results = []
    # ast is tree of predicates AND/OR/NOT with atoms like ("inside", obj, container)
    for atom in ast.atoms():
        if atom.op == "inside":
            val, conf = geometry_is_inside(atom.obj, atom.container, world_state)
        elif atom.op == "state_eq":
            val = (world_state.get_state(atom.obj, atom.key) == atom.value)
            conf = world_state.get_state_confidence(atom.obj, atom.key)
        elif atom.op == "holding_recently":
            val = memory.was_holding_recently(atom.obj, window=0.5)
            conf = 0.9 if val else 0.2
        atom_results.append({"atom":atom, "value":val, "conf":conf})
    # aggregate (conservative)
    overall_value = compute_boolean_from_atom_results(ast, atom_results)
    agg_conf = min([a['conf'] for a in atom_results])
    return overall_value, atom_results, agg_conf

B. ValidateConstraint 返回更丰富结构

def validate_constraint(constraint, world_state, memory, eval_time):
    if constraint.type == 'pre' and eval_time != 'pre': return "SKIP"
    ok, atom_trace, conf = eval_predicate_ast(constraint.condition_ast, world_state, memory)
    if conf < constraint.min_confidence: status="UNCERTAIN"
    else: status = "SATISFIED" if ok else "VIOLATED"
    return {"id":constraint.id, "status":status, "confidence":conf, "atom_trace":atom_trace}


⸻

六、需要补实验以支撑论文论点（实验建议）
	1.	噪声/遮挡实验（必须）：对比有/无 Memory 的检测误判率（false negative on pick-up occlusion）。
	2.	几何对比实验（必须）：对比仅文本 summary（REFLECT-style）与 CRAFT++ 在「靠近但不在容器内」场景的 FP/FN。
	3.	关键帧 Ablation（可选）：测试不同 keyframe 策略对检测 latency 与 accuracy 的影响（固定 per-action vs dynamic state-change）。Demo 已有基础（22.2% reduction）可扩展为曲线实验。

⸻

七、最终建议（短清单，便于直接执行）
	•	立即：把约束输出保存为 JSON（含 condition_expr），并在 notebook 打印一个 AST 示例。
	•	立即：在 constraint validation 中打印 atom-level trace（value/confidence/source）。这一步能马上显著提升 demo 的可信度。
	•	近期：在 AI2THOR 中构造 2 个噪声试验（遮挡、teleport）并展示 Memory 开/关对结果的差异（用于论文 Figure）。
	•	中期：实现 geometry.is_inside 并替换掉仅靠 edge 字符串匹配的逻辑。
	•	论文写作：在 Methods 中把 demo 的 Smart Keyframe 优势（22.2%）作为“效率”实验，但同时补充“可解释性 / determinism”实验（上面建议）。

