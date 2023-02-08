import os

# import pdb
from shutil import rmtree
from typing import Optional

import numpy as np
import pytest
from qibo.models import Circuit

from qibocal.calibrations.niGSC.basics.circuitfactory import Qibo1qGatesFactory
from qibocal.calibrations.niGSC.basics.experiment import *

nqubits = 2
depths = [1, 3, 6]
runs = 2
qubits = [0, 1]


def populate_experiment_data(circuit: Circuit, datarow: dict) -> dict:
    datarow["rand_uniform"] = np.random.uniform(0, 1)
    datarow["rand_normal"] = np.random.uniform(0, 1)
    datarow["rand_int"] = np.random.randint(0, 2)
    return datarow


data = [{} for _ in range(10)]
experiment1 = Experiment(None, data=data)
experiment1.perform(populate_experiment_data)
randnormal_array = experiment1.extract("rand_normal")
assert np.mean(randnormal_array) == experiment1.extract("rand_normal", "", "mean")
print(experiment1.extract("rand_uniform", groupby_key="rand_int"))
