"""
Perception Module
Handles multi-modal perception: vision, depth, audio
"""

from .object_detector import ObjectDetector
from .scene_analyzer import SceneAnalyzer

# Lazy imports for optional dependencies
try:
    from .reflect_scene_graph_builder import ReflectSceneGraphBuilder
except ImportError:
    ReflectSceneGraphBuilder = None

try:
    from .simulated_scene_graph_builder import SimulatedSceneGraphBuilder
except ImportError:
    SimulatedSceneGraphBuilder = None

__all__ = ['ObjectDetector', 'SceneAnalyzer']
if ReflectSceneGraphBuilder is not None:
    __all__.append('ReflectSceneGraphBuilder')
if SimulatedSceneGraphBuilder is not None:
    __all__.append('SimulatedSceneGraphBuilder')

