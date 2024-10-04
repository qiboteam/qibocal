"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

from . import protocols
from .auto.execute import Executor
from .cli import command
from .version import __version__

__all__ = ["Executor", "protocols", "command", "__version__"]

DEFAULT_EXECUTOR = Executor.create(".routines", platform="dummy")
"""Default executor, registered as a qibocal submodule.

It is defined for streamlined usage of qibocal protocols in simple
contexts, where no additional options has to be defined for the
executor.

This is not meant to be used directly, thus is not meant to be publicly
exposed.
"""
