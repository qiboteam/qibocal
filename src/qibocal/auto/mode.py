from enum import Flag, auto


class ExecutionMode(Flag):
    """Different execution modes."""

    ACQUIRE = auto()
    """Peform acquisition only."""
    FIT = auto()
    """Perform fitting only."""

    def __str__(self):
        if self is AUTOCALIBRATION:
            return "AUTOCALIBRATION"

        return self.name


AUTOCALIBRATION = ExecutionMode.ACQUIRE | ExecutionMode.FIT
"""Perform acquisition and fitting."""
