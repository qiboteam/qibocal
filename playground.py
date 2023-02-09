import os

# import pdb
from shutil import rmtree
from typing import Optional

import numpy as np
import plotly.graph_objects as go
import pytest
from qibo.models import Circuit

from qibocal.calibrations.niGSC import correlatedrb
from qibocal.calibrations.niGSC.basics import fitting, noisemodels, utils
from qibocal.calibrations.niGSC.basics.circuitfactory import Qibo1qGatesFactory
from qibocal.calibrations.niGSC.basics.experiment import *

success = 0
number_runs = 50
success = 0
number_runs = 50
for count in range(number_runs):
    x = np.sort(np.random.choice(np.linspace(0, 15, 50), size=20, replace=False))
    A, f = np.random.uniform(0.1, 0.99, size=2)
    y = A * f**x
    # Distort ``y`` a bit.
    y_dist = y + np.random.randn(len(y)) * 0.005
    popt, perr = fitting.fit_exp1_func(x, y_dist)
    success += np.all(
        np.logical_or(
            np.abs(np.array(popt) - [A, f]) < 2 * np.array(perr),
            np.abs(np.array(popt) - [A, f]) < 0.01,
        )
    )
    print(popt)
    print(A, f)
print(success)
