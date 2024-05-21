from enum import IntFlag, auto


class ExecutionMode(IntFlag):
    """Different execution modes"""

    ACQUIRE = auto()
    FIT = auto()
    REPORT = auto()


AUTOCALIBRATION = ExecutionMode.ACQUIRE | ExecutionMode.FIT | ExecutionMode.REPORT
