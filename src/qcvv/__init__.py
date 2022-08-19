# -*- coding: utf-8 -*-
from .cli import command, live_plot, report

"""qcvv: Quantum Calibration Verification and Validation using Qibo."""
import importlib.metadata as im

__version__ = im.version(__package__)
