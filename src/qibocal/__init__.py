"""qibocal: Quantum Calibration Verification and Validation using Qibo."""

import importlib.metadata as im

from .cli import command, compare, live_plot, monitor, upload

__version__ = im.version(__package__)
