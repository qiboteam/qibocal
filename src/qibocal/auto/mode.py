from enum import Flag, auto


class ExecutionMode(Flag):
    """Different execution modes"""

    ACQUIRE = auto()
    """Peform acquisition only."""
    FIT = auto()
    """Perform fitting only"""
    UPDATE = auto()
    """Perform update of platform."""


AUTOCALIBRATION = ExecutionMode.ACQUIRE | ExecutionMode.FIT | ExecutionMode.UPDATE
"""Perform acquisition and fitting."""
