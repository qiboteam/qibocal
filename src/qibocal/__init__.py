"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

import importlib.metadata as im

from . import protocols
from .auto.execute import Executor
from .cli import command

__version__ = im.version(__package__)
__all__ = ["Executor", "protocols", "command"]
