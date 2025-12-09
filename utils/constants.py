"""
Constants and mappings for AI2THOR simulation
Adapted from REFLECT framework
"""

# Object name mappings (from AI2THOR objectType to natural language)
NAME_MAP = {
    "TomatoSliced": "tomato slice",
    "PotatoSliced": "potato slice",
    "LettuceSliced": "lettuce slice",
    "BreadSliced": "bread slice",
    "EggCracked": "cracked egg",
    "AppleSliced": "apple slice",
    "HousePlant": "house plant",
    "CounterTop-1": "first countertop",
    "CounterTop-2": "second countertop",
    "CounterTop-3": "third countertop",
    "CounterTop-4": "fourth countertop",
    "StoveBurner-1": "first stove burner",
    "StoveBurner-2": "second stove burner",
    "StoveBurner-3": "third stove burner",
    "StoveBurner-4": "fourth stove burner",
    "Cabinet-1": "first cabinet",
    "Cabinet-2": "second cabinet",
    "Faucet-1": "first faucet",
    "Faucet-2": "second faucet",
    "Sink-1": "first sink",
    "Sink-2": "second sink",
    "StoveBurner": "stove burner",
    "CoffeeMachine": "coffee machine",
    "SinkBasin": "sink",
    "Mug": "mug",
    "CounterTop": "countertop",
}

# Task dictionary mapping
TASK_DICT = {
    0: "makeCoffee",
    1: "makeBreakfast",
    2: "makeLunch",
    3: "makeDinner",
    4: "cleanRoom",
    5: "makeCoffee",  # Default for make coffee task
}

# Failure types
FAILURE_TYPES = ['drop', 'failed_action', 'missing_step', 'blocking', 'occupied']

# Object type mappings for sliced/unsliced objects
OBJ_UNSLICED_MAP = {
    "TomatoSliced": "Tomato",
    "PotatoSliced": "Potato",
    "LettuceSliced": "Lettuce",
    "BreadSliced": "Bread",
    "AppleSliced": "Apple",
}

OBJ_SLICED_MAP = {
    "Tomato": "TomatoSliced",
    "Potato": "PotatoSliced",
    "Lettuce": "LettuceSliced",
    "Bread": "BreadSliced",
    "Apple": "AppleSliced",
}

