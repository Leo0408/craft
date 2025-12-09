"""
Utility functions
"""

from .config_loader import load_config
from .data_loader import DataLoader

# Optional imports (may fail if dependencies not installed)
try:
    from .reflect_data_loader import ReflectDataLoader
except ImportError:
    ReflectDataLoader = None

try:
    from .simulated_data_generator import SimulatedDataGenerator
except ImportError:
    SimulatedDataGenerator = None

try:
    from .video_generator import VideoGenerator
except ImportError:
    VideoGenerator = None

# AI2THOR data generation modules (REFLECT-style)
try:
    from .gen_data import run_data_gen
    from .task_utils import TaskUtil
    from . import action_primitives
    from .constants import NAME_MAP, TASK_DICT, FAILURE_TYPES
except ImportError:
    run_data_gen = None
    TaskUtil = None
    action_primitives = None
    NAME_MAP = None
    TASK_DICT = None
    FAILURE_TYPES = None

__all__ = ['load_config', 'DataLoader']

if ReflectDataLoader is not None:
    __all__.append('ReflectDataLoader')
if SimulatedDataGenerator is not None:
    __all__.append('SimulatedDataGenerator')
if VideoGenerator is not None:
    __all__.append('VideoGenerator')

if run_data_gen is not None:
    __all__.extend(['run_data_gen', 'TaskUtil', 'action_primitives', 'NAME_MAP', 'TASK_DICT', 'FAILURE_TYPES'])

