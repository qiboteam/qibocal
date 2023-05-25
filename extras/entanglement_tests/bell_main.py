import shutil
import time

import numpy as np
from bell_functions import BellExperiment
from qibo import set_backend
from qibo.config import log
from qibolab import Platform
from qibolab.backends import QibolabBackend
from qibolab.paths import qibolab_folder
from readout_mitigation import ReadoutErrorMitigation

nqubits = 5
qubits = [2, 3]
nshots = 10000
runcard = "../../../qibolab/src/qibolab/runcards/qw5q_gold_qblox.yml"
timestr = time.strftime("%Y%m%d-%H%M")
shutil.copy(runcard, f"{timestr}_runcard.yml")
ntheta = 20

bell_basis = [0, 1, 2, 3]

thetas = np.linspace(0, 2 * np.pi, ntheta)

platform = Platform("qblox", runcard)
# platform = None

platform.connect()
platform.setup()
platform.start()

readout_mitigation = ReadoutErrorMitigation(platform, nqubits, qubits)

calibration_matrix = readout_mitigation.get_calibration_matrix(nshots)

bell = BellExperiment(platform, nqubits)

bell.execute(
    qubits,
    bell_basis,
    thetas,
    nshots,
    pulses=True,
    native=True,
    readout_mitigation=readout_mitigation,
    exact=True,
)

platform.stop()
platform.disconnect()

"""
Simulation version:

set_backend('numpy')

nqubits = 5
qubits = [2, 3]
nshots = 10000
ntheta = 20

rerr = (0.05, 0.25)

bell_basis = [0, 1, 2, 3]

thetas = np.linspace(0, 2*np.pi, ntheta)

readout_mitigation = ReadoutErrorMitigation(None, nqubits, qubits, rerr)

calibration_matrix = readout_mitigation.get_calibration_matrix(nshots)

bell = BellExperiment(None, nqubits, rerr)

bell.execute(qubits,
	bell_basis,
	thetas,
	nshots,
	pulses=False,
	native=True,
	readout_mitigation=readout_mitigation,
	exact=True)

"""
