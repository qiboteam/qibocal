"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

from . import protocols
from .auto.execute import Executor
from .cli import command
from .version import __version__

__all__ = ["Executor", "protocols", "command", "__version__"]

Executor.create("qibocal.routines")
