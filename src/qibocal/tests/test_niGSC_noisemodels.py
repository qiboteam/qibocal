import numpy as np
import qibo

from qibocal.calibrations.niGSC.basics import noisemodels


def test_PauliErrorOnUnitary():
    pauli_onU_error = noisemodels.PauliErrorOnUnitary()
    assert isinstance(pauli_onU_error, qibo.noise.NoiseModel)
    errorkeys = pauli_onU_error.errors.keys()
    assert len(errorkeys) == 1 and list(errorkeys)[0] == qibo.gates.gates.Unitary
    error = pauli_onU_error.errors[qibo.gates.gates.Unitary][0]
    assert isinstance(error, qibo.noise.PauliError)
    assert len(error.options) == 3 and np.sum(error.options) < 1


def test_PauliErrorOnX():
    pauli_onX_error = noisemodels.PauliErrorOnX()
    assert isinstance(pauli_onX_error, qibo.noise.NoiseModel)
    errorkeys = pauli_onX_error.errors.keys()
    assert len(errorkeys) == 1 and list(errorkeys)[0] == qibo.gates.gates.X
    error = pauli_onX_error.errors[qibo.gates.gates.X][0]
    assert isinstance(error, qibo.noise.PauliError)
    assert len(error.options) == 3 and np.sum(error.options) < 1
