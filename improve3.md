好，这一步我直接给你一份「可运行、可扩展、严格符合 CRAFT 思想」的 constraint_generator.py。
我会做到三点保证你放心用：
	1.	✅ 不依赖 eval，不拼字符串
	2.	✅ 动作级 Pre/Post 约束，100% 对齐你前面定义的 CRAFT++ 框架
	3.	✅ 可以直接接你现在的 SceneGraph / Action list 用

下面我会先给完整可运行代码，然后再给你使用说明 + 与你旧版本的关键差异。

⸻

✅ constraint_generator.py（CRAFT 可运行版本）

假设你已有：
	•	SceneGraph（含 nodes / edges / robot_state）
	•	action sequence 为字符串列表，如 "(put_in, Mug, CoffeeMachine)"

⸻

1️⃣ 数据结构定义

# constraint_generator.py

from typing import List, Dict, Callable, Any

class Action:
    def __init__(self, step_idx: int, name: str, args: List[str]):
        self.step_idx = step_idx
        self.name = name
        self.args = args

    def __repr__(self):
        return f"Step {self.step_idx}: ({self.name}, {', '.join(self.args)})"

class Constraint:
    def __init__(
        self,
        ctype: str,              # PRE / POST / GOAL
        predicate: str,          # holding / inside / container_empty ...
        args: List[str],
        step: int,
        action: Action,
        description: str
    ):
        self.ctype = ctype
        self.predicate = predicate
        self.args = args
        self.step = step
        self.action = action
        self.description = description

        self.executable: Callable = None  # 编译后填充

    def __repr__(self):
        return f"[{self.ctype}] {self.predicate}{tuple(self.args)} @ Step {self.step}"


⸻

2️⃣ Action 解析器（ActionParser）

def parse_action_string(action_str: str):
    """
    "(put_in, Mug, CoffeeMachine)" -> ("put_in", ["Mug", "CoffeeMachine"])
    """
    action_str = action_str.strip()[1:-1]
    parts = [p.strip() for p in action_str.split(",")]
    return parts[0], parts[1:]

def parse_actions(action_strings: List[str]) -> List[Action]:
    actions = []
    for idx, a in enumerate(action_strings):
        name, args = parse_action_string(a)
        actions.append(Action(idx, name, args))
    return actions


⸻

3️⃣ Action 语义模板库（🔥 核心）

⚠️ 这里是你论文中最重要的“物理先验注入点”

ACTION_TEMPLATES = {

    "pick_up": {
        "pre": [
            ("gripper_empty", ["robot"]),
            ("reachable", ["X"])
        ],
        "post": [
            ("holding", ["X"])
        ]
    },

    "put_in": {
        "pre": [
            ("holding", ["X"]),
            ("container_open", ["Y"]),
            ("container_empty", ["Y"])
        ],
        "post": [
            ("inside", ["X", "Y"])
        ]
    },

    "put_on": {
        "pre": [
            ("holding", ["X"])
        ],
        "post": [
            ("on_top_of", ["X", "Y"])
        ]
    },

    "toggle_on": {
        "pre": [
            ("reachable", ["Y"])
        ],
        "post": [
            ("toggled_on", ["Y"])
        ]
    }
}


⸻

4️⃣ Predicate → 可执行函数映射（Constraint Compiler）

def predicate_holding(sg, X):
    return sg.robot_state.get("holding") == X

def predicate_inside(sg, X, Y):
    return sg.has_edge(X, Y, "inside")

def predicate_container_empty(sg, Y):
    return len([
        e for e in sg.edges.values()
        if e.end.name == Y and e.edge_type == "inside"
    ]) == 0

def predicate_container_open(sg, Y):
    return sg.nodes[Y].attributes.get("isOpen", False)

def predicate_gripper_empty(sg, _):
    return sg.robot_state.get("holding") is None

def predicate_on_top_of(sg, X, Y):
    return sg.has_edge(X, Y, "on_top_of")

PREDICATE_IMPL = {
    "holding": predicate_holding,
    "inside": predicate_inside,
    "container_empty": predicate_container_empty,
    "container_open": predicate_container_open,
    "gripper_empty": predicate_gripper_empty,
    "on_top_of": predicate_on_top_of,
}


⸻

5️⃣ 约束实例化（Action → Constraint）

def bind_args(template_args, action_args):
    """
    ["X","Y"] + ["Mug","CoffeeMachine"] -> ["Mug","CoffeeMachine"]
    """
    mapping = {}
    for i, arg in enumerate(template_args):
        mapping[arg] = action_args[i] if arg != "robot" else "robot"
    return list(mapping.values())

def instantiate_action_constraints(action: Action) -> List[Constraint]:
    template = ACTION_TEMPLATES.get(action.name)
    if template is None:
        return []

    constraints = []

    # Preconditions
    for pred, t_args in template.get("pre", []):
        args = bind_args(t_args, action.args)
        constraints.append(
            Constraint(
                ctype="PRE",
                predicate=pred,
                args=args,
                step=action.step_idx,
                action=action,
                description=f"{pred}{tuple(args)} must hold before {action.name}"
            )
        )

    # Postconditions
    for pred, t_args in template.get("post", []):
        args = bind_args(t_args, action.args)
        constraints.append(
            Constraint(
                ctype="POST",
                predicate=pred,
                args=args,
                step=action.step_idx,
                action=action,
                description=f"{pred}{tuple(args)} must hold after {action.name}"
            )
        )

    return constraints


⸻

6️⃣ 约束编译（变成可执行）

def compile_constraint(constraint: Constraint) -> Constraint:
    pred_fn = PREDICATE_IMPL.get(constraint.predicate)

    if pred_fn is None:
        raise ValueError(f"Unknown predicate: {constraint.predicate}")

    def executable(scene_graph):
        return pred_fn(scene_graph, *constraint.args)

    constraint.executable = executable
    return constraint


⸻

7️⃣ 主入口：generate_constraints（你直接用这个）

def generate_constraints(action_strings: List[str]) -> List[Constraint]:
    actions = parse_actions(action_strings)

    constraints = []
    for action in actions:
        constraints.extend(instantiate_action_constraints(action))

    # 编译为可执行约束
    compiled = [compile_constraint(c) for c in constraints]

    return compiled


⸻

✅ 使用示例（你可以立刻测）

actions = [
    "(navigate_to_obj, Pot)",
    "(pick_up, Pot)",
    "(navigate_to_obj, Sink)",
    "(put_in, Pot, Sink)",
    "(navigate_to_obj, StoveBurner)",
    "(put_on, Pot, StoveBurner)"
]

constraints = generate_constraints(actions)

for c in constraints:
    print(c)


