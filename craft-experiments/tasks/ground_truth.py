"""
Ground Truth Functions for Task Success Conditions

GT 是代码，不是文字 - 这是 CRAFT 的核心优势之一
"""

from typing import Dict, Any


def make_coffee_success(state: Dict[str, Any]) -> bool:
    """Check if make_coffee task succeeded"""
    try:
        # Check if coffee machine is on
        assert state.get("coffee_machine", {}).get("is_on", False), "Coffee machine is not on"
        # Check if mug contains coffee
        mug_state = state.get("mug", {})
        assert "coffee" in mug_state.get("contains", []), "Mug does not contain coffee"
        return True
    except AssertionError:
        return False


def make_tea_success(state: Dict[str, Any]) -> bool:
    """Check if make_tea task succeeded"""
    try:
        # Check if mug contains water
        mug_state = state.get("mug", {})
        assert "water" in mug_state.get("contains", []), "Mug does not contain water"
        # Check if mug is on countertop
        assert mug_state.get("on_top_of") == "countertop", "Mug is not on countertop"
        return True
    except AssertionError:
        return False


def clean_mug_success(state: Dict[str, Any]) -> bool:
    """Check if clean_mug task succeeded"""
    try:
        # Check if mug is clean
        mug_state = state.get("mug", {})
        assert mug_state.get("is_clean", False), "Mug is not clean"
        # Check if mug is on countertop
        assert mug_state.get("on_top_of") == "countertop", "Mug is not on countertop"
        return True
    except AssertionError:
        return False


# Task to GT function mapping
GT_FUNCTIONS = {
    "make_coffee": make_coffee_success,
    "make_tea": make_tea_success,
    "clean_mug": clean_mug_success,
}


def check_task_success(task_name: str, state: Dict[str, Any]) -> bool:
    """Check if a task succeeded using its GT function"""
    if task_name not in GT_FUNCTIONS:
        raise ValueError(f"Unknown task: {task_name}")
    return GT_FUNCTIONS[task_name](state)
