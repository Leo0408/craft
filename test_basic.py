"""
Basic test script to verify CRAFT installation
"""

import sys
import os

# Add parent directory to path so we can import craft
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from craft.core import SceneGraph, Node, Edge, TaskExecutor
        print("✓ Core modules imported")
        
        from craft.perception import ObjectDetector, SceneAnalyzer
        print("✓ Perception modules imported")
        
        from craft.reasoning import FailureAnalyzer, LLMPrompter
        print("✓ Reasoning modules imported")
        
        from craft.correction import CorrectionPlanner
        print("✓ Correction modules imported")
        
        from craft.utils import load_config, DataLoader
        print("✓ Utility modules imported")
        
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scene_graph():
    """Test scene graph creation"""
    print("\nTesting scene graph...")
    
    try:
        from craft.core import SceneGraph, Node, Edge
        
        sg = SceneGraph()
        node1 = Node("coffee machine", "CoffeeMachine", state="closed")
        node2 = Node("cup", "Mug", state="empty")
        
        sg.add_node(node1)
        sg.add_node(node2)
        
        edge = Edge(node2, node1, "near")
        sg.add_edge(edge)
        
        text = sg.to_text()
        assert len(text) > 0
        print(f"✓ Scene graph created: {text[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Scene graph error: {e}")
        return False


def test_task_executor():
    """Test task executor"""
    print("\nTesting task executor...")
    
    try:
        from craft.core import TaskExecutor
        
        actions = [
            {"type": "pick_up", "target": "Mug"},
            {"type": "put_in", "source": "Mug", "target": "CoffeeMachine"}
        ]
        
        executor = TaskExecutor("make coffee", actions)
        assert len(executor.actions) == 2
        assert executor.task_name == "make coffee"
        
        executor.mark_action_success(action_idx=0)
        executor.mark_action_failed("Test failure", action_idx=1)
        
        failed = executor.get_failed_actions()
        assert len(failed) == 1
        
        print("✓ Task executor works")
        return True
    except Exception as e:
        print(f"✗ Task executor error: {e}")
        return False


def test_llm_prompter():
    """Test LLM prompter (mock mode)"""
    print("\nTesting LLM prompter...")
    
    try:
        from craft.reasoning import LLMPrompter
        
        prompter = LLMPrompter(gpt_version="gpt-3.5-turbo")
        # This will use mock mode if no API key
        response, _ = prompter.query("Test system", "Test user prompt")
        assert response is not None
        
        print("✓ LLM prompter works (mock mode)")
        return True
    except Exception as e:
        print(f"✗ LLM prompter error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 50)
    print("CRAFT Basic Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_scene_graph,
        test_task_executor,
        test_llm_prompter
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return 0
    else:
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

