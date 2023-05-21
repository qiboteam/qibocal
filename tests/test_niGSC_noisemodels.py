import numpy as np
import pytest
import qibo

from qibocal.calibrations.niGSC.basics import noisemodels


def test_PauliErrorOnAll():
    def test_model(pauli_error):
        assert isinstance(pauli_error, qibo.noise.NoiseModel)
        errorkeys = pauli_error.errors.keys()
        assert len(errorkeys) == 1 and list(errorkeys)[0] is None
        error = pauli_error.errors[None][0][1]
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(error.options) < 1

    noise_model1 = noisemodels.PauliErrorOnAll()
    test_model(noise_model1)
    noise_model2 = noisemodels.PauliErrorOnAll(0.1, 0.1, 0.1)
    test_model(noise_model2)
    noise_model3 = noisemodels.PauliErrorOnAll(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.PauliErrorOnAll(0.1, 0.2)


def test_PauliErrorOnX():
    def test_model(pauli_onX_error):
        assert isinstance(pauli_onX_error, qibo.noise.NoiseModel)
        errorkeys = pauli_onX_error.errors.keys()
        assert len(errorkeys) == 1 and list(errorkeys)[0] == qibo.gates.gates.X
        error = pauli_onX_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(error.options) < 1

    noise_model1 = noisemodels.PauliErrorOnX()
    test_model(noise_model1)
    noise_model2 = noisemodels.PauliErrorOnX(0.1, 0.1, 0.1)
    test_model(noise_model2)
    noise_model3 = noisemodels.PauliErrorOnX(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.PauliErrorOnX(0.1, 0.2)
