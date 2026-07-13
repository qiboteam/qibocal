"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

from . import auto, protocols
from .auto import *
from .auto.execute import Executor
from .calibration import create_calibration_platform
from .cli import command
from .version import __version__

__all__ = [
    "Executor",
    "protocols",
    "command",
    "__version__",
    "create_calibration_platform",
]
__all__ += auto.__all__
