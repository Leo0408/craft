"""
Simple example demonstrating CRAFT framework usage
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scene_graph import SceneGraph, Node, Edge
from core.task_executor import TaskExecutor, ActionStatus
from reasoning.llm_prompter import LLMPrompter
from reasoning.failure_analyzer import FailureAnalyzer
from correction.correction_planner import CorrectionPlanner


def example_scene_graph():
    """Example: Creating and using a scene graph"""
    print("=== Scene Graph Example ===")
    
    # Create scene graph
    scene_graph = SceneGraph()
    
    # Add nodes (objects)
    coffee_machine = Node("coffee machine", "CoffeeMachine", state="closed")
    cup = Node("purple cup", "Mug", state="empty")
    table = Node("table", "Table")
    
    scene_graph.add_node(coffee_machine)
    scene_graph.add_node(cup)
    scene_graph.add_node(table)
    
    # Add edges (relationships)
    edge1 = Edge(cup, table, "on_top_of")
    edge2 = Edge(coffee_machine, table, "on_top_of")
    
    scene_graph.add_edge(edge1)
    scene_graph.add_edge(edge2)
    
    # Convert to text
    print("Scene description:")
    print(scene_graph.to_text())
    print()


def example_task_executor():
    """Example: Task execution tracking"""
    print("=== Task Executor Example ===")
    
    actions = [
        {"type": "navigate_to", "target": "Mug"},
        {"type": "pick_up", "target": "Mug"},
        {"type": "navigate_to", "target": "CoffeeMachine"},
        {"type": "put_in", "source": "Mug", "target": "CoffeeMachine"}
    ]
    
    executor = TaskExecutor("make coffee", actions)
    
    print(f"Task: {executor.task_name}")
    print(f"Total actions: {len(executor.actions)}")
    
    # Simulate execution
    executor.mark_action_success(0)
    executor.mark_action_success(1)
    executor.mark_action_success(2)
    executor.mark_action_failed(3, "Coffee machine already contains a cup")
    
    failed = executor.get_failed_actions()
    print(f"\nFailed actions: {len(failed)}")
    for action in failed:
        print(f"  - {action.action_type} {action.target}: {action.failure_reason}")
    print()


def example_failure_analysis():
    """Example: Failure analysis with LLM"""
    print("=== Failure Analysis Example ===")
    
    # Initialize LLM prompter (will use mock if no API key)
    llm_prompter = LLMPrompter(gpt_version="gpt-3.5-turbo")
    
    # Verify a subgoal
    is_success, explanation = llm_prompter.verify_subgoal(
        task="make coffee",
        subgoal="put cup in coffee machine",
        observation="a coffee machine (closed), a purple cup, a table. a blue cup is inside the coffee machine."
    )
    
    print(f"Subgoal success: {is_success}")
    print(f"Explanation: {explanation}")
    
    # Explain execution failure
    failure_explanation = llm_prompter.explain_execution_failure(
        task="make coffee",
        action="put_in Mug CoffeeMachine",
        observation="a coffee machine (closed), a purple cup, a table. a blue cup is inside the coffee machine."
    )
    
    print(f"\nFailure explanation: {failure_explanation}")
    print()


def example_correction_planning():
    """Example: Generating correction plans"""
    print("=== Correction Planning Example ===")
    
    llm_prompter = LLMPrompter(gpt_version="gpt-3.5-turbo")
    planner = CorrectionPlanner(llm_prompter)
    
    # Create a simple scene graph for final state
    from core.scene_graph import SceneGraph, Node, Edge
    final_state = SceneGraph()
    final_state.add_node(Node("coffee machine", "CoffeeMachine", state="closed"))
    final_state.add_node(Node("blue cup", "Mug", state="empty"))
    final_state.add_node(Node("purple cup", "Mug", state="empty"))
    
    # Original plan
    from core.task_executor import Action, ActionStatus
    original_plan = [
        Action("navigate_to", target="Mug"),
        Action("pick_up", target="Mug"),
        Action("put_in", source="Mug", target="CoffeeMachine")
    ]
    
    task_info = {
        "name": "make coffee",
        "success_condition": "a clean mug is filled with coffee and on top of the countertop"
    }
    
    correction_plan = planner.generate_correction_plan(
        task_info=task_info,
        original_plan=original_plan,
        failure_explanation="The coffee machine already contains a blue cup, blocking the purple cup",
        final_state=final_state,
        expected_goal=task_info["success_condition"]
    )
    
    print("Correction plan:")
    for i, action in enumerate(correction_plan, 1):
        print(f"  {i}. {action.get('type', '')} {action.get('target', '')}")
    print()


if __name__ == "__main__":
    print("CRAFT Framework Examples\n")
    print("=" * 50)
    
    example_scene_graph()
    example_task_executor()
    example_failure_analysis()
    example_correction_planning()
    
    print("=" * 50)
    print("Examples completed!")




