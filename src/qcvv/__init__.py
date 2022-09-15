# -*- coding: utf-8 -*-
from .cli import command, high, live_plot, upload

"""qcvv: Quantum Calibration Verification and Validation using Qibo."""
import importlib.metadata as im

__version__ = im.version(__package__)
