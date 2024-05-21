from enum import Flag, auto


class ExecutionMode(Flag):
    """Different execution modes"""

    ACQUIRE = auto()
    """Peform acquisition only."""
    FIT = auto()
    """Perform fitting only"""


AUTOCALIBRATION = ExecutionMode.ACQUIRE | ExecutionMode.FIT
"""Perform acquisition and fitting."""
