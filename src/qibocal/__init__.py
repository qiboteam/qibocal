"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

import importlib.metadata as im

from . import protocols
from .auto.execute import Executor
from .cli import command

__version__ = im.version(__package__)
__all__ = ["Executor", "protocols", "command"]

DEFAULT_EXECUTOR = Executor.create("qibocal.routines")
"""Default executor, registered as a qibocal submodule.

It is defined for streamlined usage of qibocal protocols in simple
contexts, where no additional options has to be defined for the
executor.

This is not meant to be used directly, thus is not meant to be publicly
exposed.
"""
