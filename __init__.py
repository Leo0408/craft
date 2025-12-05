"""
CRAFT: Core Robot Analysis Framework for Tasks
A framework for robot failure analysis and correction based on REFLECT
"""

__version__ = "0.1.0"

from .core import SceneGraph, Node, Edge, TaskExecutor
from .perception import ObjectDetector, SceneAnalyzer
from .reasoning import FailureAnalyzer, LLMPrompter
from .correction import CorrectionPlanner

# Optional imports (may fail if dependencies not installed)
try:
    from .perception import ReflectSceneGraphBuilder, SimulatedSceneGraphBuilder
except ImportError:
    ReflectSceneGraphBuilder = None
    SimulatedSceneGraphBuilder = None

try:
    from .utils import ReflectDataLoader, SimulatedDataGenerator
except ImportError:
    ReflectDataLoader = None
    SimulatedDataGenerator = None

__all__ = [
    'SceneGraph', 'Node', 'Edge', 'TaskExecutor',
    'ObjectDetector', 'SceneAnalyzer',
    'FailureAnalyzer', 'LLMPrompter',
    'CorrectionPlanner'
]

# Add optional exports
if ReflectSceneGraphBuilder is not None:
    __all__.append('ReflectSceneGraphBuilder')
if SimulatedSceneGraphBuilder is not None:
    __all__.append('SimulatedSceneGraphBuilder')
if ReflectDataLoader is not None:
    __all__.append('ReflectDataLoader')
if SimulatedDataGenerator is not None:
    __all__.append('SimulatedDataGenerator')


