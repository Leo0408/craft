"""
Failure Types Enum
"""
from enum import Enum


class FailureType(Enum):
    """失败类型枚举"""
    MISSING_PRECONDITION = "MISSING_PRECONDITION"
    PHYSICAL_IMPOSSIBLE = "PHYSICAL_IMPOSSIBLE"
    CAUSAL_BREAK = "CAUSAL_BREAK"
    PERCEPTION_NOISE = "PERCEPTION_NOISE"
