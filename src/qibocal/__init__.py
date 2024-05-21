"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

import importlib.metadata as im

from qibocal.auto.execute import Executor

__version__ = im.version(__package__)
__all__ = ["Executor"]
