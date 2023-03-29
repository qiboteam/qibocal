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


def test_PauliErrorOnXAndRX():
    def test_model(pauli_onXRX_error):
        assert isinstance(pauli_onXRX_error, qibo.noise.NoiseModel)
        errorkeys = pauli_onXRX_error.errors.keys()
        assert (
            len(errorkeys) == 2
            and qibo.gates.gates.X in list(errorkeys)
            and qibo.gates.gates.RX in list(errorkeys)
        )
        error = pauli_onXRX_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(error.options) < 1
        error = pauli_onXRX_error.errors[qibo.gates.gates.RX][0][1]
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(error.options) < 1

    noise_model1 = noisemodels.PauliErrorOnXAndRX()
    test_model(noise_model1)
    noise_model2 = noisemodels.PauliErrorOnXAndRX(0.1, 0.1, 0.1)
    test_model(noise_model2)
    noise_model3 = noisemodels.PauliErrorOnXAndRX(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.PauliErrorOnXAndRX(0.1, 0.2)


def test_PauliErrorOnNonDiagonal():
    def test_model(pauli_non_diag_error):
        assert isinstance(pauli_non_diag_error, qibo.noise.NoiseModel)
        errorkeys = pauli_non_diag_error.errors.keys()
        assert (
            len(errorkeys) == 3
            and qibo.gates.gates.X in list(errorkeys)
            and qibo.gates.gates.Y in list(errorkeys)
            and qibo.gates.gates.Unitary in list(errorkeys)
        )
        error = pauli_non_diag_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(error.options) < 1
        error = pauli_non_diag_error.errors[qibo.gates.gates.Y][0][1]
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(error.options) < 1
        condition, error = pauli_non_diag_error.errors[qibo.gates.gates.Unitary][0][:2]
        assert not condition(qibo.gates.gates.Unitary(np.eye(2), 0))
        assert condition(qibo.gates.gates.Unitary(np.array([[0, 1], [1, 0]]), 0))
        assert isinstance(error, qibo.noise.PauliError)
        assert len(error.options) == 3 and np.sum(error.options) < 1

    noise_model1 = noisemodels.PauliErrorOnNonDiagonal()
    test_model(noise_model1)
    noise_model2 = noisemodels.PauliErrorOnNonDiagonal(0.1, 0.1, 0.1)
    test_model(noise_model2)
    noise_model3 = noisemodels.PauliErrorOnNonDiagonal(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.PauliErrorOnNonDiagonal(0.1, 0.2)


def test_UnitaryErrorOnAll():
    def test_model(unitary_error):
        assert isinstance(unitary_error, qibo.noise.NoiseModel)
        errorkeys = unitary_error.errors.keys()
        assert len(errorkeys) == 1 and list(errorkeys)[0] is None
        error = unitary_error.errors[None][0][1]
        assert isinstance(error, qibo.noise.UnitaryError)
        assert len(error.probabilities) == len(error.unitaries)

    u1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    u2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probabilities = [0.3, 0.7]

    noise_model1 = noisemodels.UnitaryErrorOnAll()
    test_model(noise_model1)
    noise_model2 = noisemodels.UnitaryErrorOnAll(probabilities, [u1, u2])
    test_model(noise_model2)
    noise_model3 = noisemodels.UnitaryErrorOnAll(None)
    test_model(noise_model3)
    with pytest.raises(TypeError):
        noisemodels.UnitaryErrorOnX(t="0.1")
    with pytest.raises(ValueError):
        noisemodels.UnitaryErrorOnAll(probabilities, [u1, np.array([[1, 0], [0, 1]])])


def test_UnitaryErrorOnX():
    def test_model(unitary_error):
        assert isinstance(unitary_error, qibo.noise.NoiseModel)
        errorkeys = unitary_error.errors.keys()
        assert len(errorkeys) == 1 and list(errorkeys)[0] == qibo.gates.gates.X
        error = unitary_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.UnitaryError)
        assert len(error.probabilities) == len(error.unitaries)

    u1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    u2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probabilities = [0.3, 0.7]

    noise_model1 = noisemodels.UnitaryErrorOnX()
    test_model(noise_model1)
    noise_model2 = noisemodels.UnitaryErrorOnX(probabilities, [u1, u2])
    test_model(noise_model2)
    noise_model3 = noisemodels.UnitaryErrorOnX(None)
    test_model(noise_model3)
    with pytest.raises(TypeError):
        noisemodels.UnitaryErrorOnX(t="0.1")
    with pytest.raises(ValueError):
        noisemodels.UnitaryErrorOnX(probabilities, [u1, np.array([[1, 0], [0, 1]])])


def test_UnitaryErrorOnXAndRX():
    def test_model(unitary_error):
        assert isinstance(unitary_error, qibo.noise.NoiseModel)
        errorkeys = unitary_error.errors.keys()
        assert (
            len(errorkeys) == 2
            and qibo.gates.gates.X in list(errorkeys)
            and qibo.gates.gates.RX in list(errorkeys)
        )
        error = unitary_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.UnitaryError)
        assert len(error.probabilities) == len(error.unitaries)
        error = unitary_error.errors[qibo.gates.gates.RX][0][1]
        assert isinstance(error, qibo.noise.UnitaryError)
        assert len(error.probabilities) == len(error.unitaries)

    u1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    u2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probabilities = [0.3, 0.7]

    noise_model1 = noisemodels.UnitaryErrorOnXAndRX()
    test_model(noise_model1)
    noise_model2 = noisemodels.UnitaryErrorOnXAndRX(probabilities, [u1, u2])
    test_model(noise_model2)
    noise_model3 = noisemodels.UnitaryErrorOnXAndRX(None)
    test_model(noise_model3)
    with pytest.raises(TypeError):
        noisemodels.UnitaryErrorOnXAndRX(t="0.1")
    with pytest.raises(ValueError):
        noisemodels.UnitaryErrorOnXAndRX(
            probabilities, [u1, np.array([[1, 0], [0, 1]])]
        )


def test_UnitaryErrorOnNonDiagonal():
    def test_model(unitary_error):
        assert isinstance(unitary_error, qibo.noise.NoiseModel)
        errorkeys = unitary_error.errors.keys()
        assert (
            len(errorkeys) == 3
            and qibo.gates.gates.X in list(errorkeys)
            and qibo.gates.gates.Y in list(errorkeys)
            and qibo.gates.gates.Unitary in list(errorkeys)
        )
        error = unitary_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.UnitaryError)
        assert len(error.probabilities) == len(error.unitaries)

        error = unitary_error.errors[qibo.gates.gates.Y][0][1]
        assert isinstance(error, qibo.noise.UnitaryError)
        assert len(error.probabilities) == len(error.unitaries)

        condition, error = unitary_error.errors[qibo.gates.gates.Unitary][0][:2]
        assert not condition(qibo.gates.gates.Unitary(np.eye(2), 0))
        assert condition(qibo.gates.gates.Unitary(np.array([[0, 1], [1, 0]]), 0))
        assert isinstance(error, qibo.noise.UnitaryError)
        assert len(error.probabilities) == len(error.unitaries)

    u1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    u2 = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    probabilities = [0.3, 0.7]

    noise_model1 = noisemodels.UnitaryErrorOnNonDiagonal()
    test_model(noise_model1)
    noise_model2 = noisemodels.UnitaryErrorOnNonDiagonal(probabilities, [u1, u2])
    test_model(noise_model2)
    noise_model3 = noisemodels.UnitaryErrorOnNonDiagonal(None)
    test_model(noise_model3)
    with pytest.raises(TypeError):
        noisemodels.UnitaryErrorOnNonDiagonal(t="0.1")
    with pytest.raises(ValueError):
        noisemodels.UnitaryErrorOnNonDiagonal(
            probabilities, [u1, np.array([[1, 0], [0, 1]])]
        )


def test_ThermalRelaxationErrorOnAll():
    def test_model(thermal_error):
        assert isinstance(thermal_error, qibo.noise.NoiseModel)
        errorkeys = thermal_error.errors.keys()
        assert len(errorkeys) == 1 and list(errorkeys)[0] is None
        error = thermal_error.errors[None][0][1]
        assert isinstance(error, qibo.noise.ThermalRelaxationError)
        assert len(error.options) == 4

    noise_model1 = noisemodels.ThermalRelaxationErrorOnAll()
    test_model(noise_model1)
    noise_model2 = noisemodels.ThermalRelaxationErrorOnAll(10, 10, 1)
    test_model(noise_model2)
    noise_model3 = noisemodels.ThermalRelaxationErrorOnAll(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.ThermalRelaxationErrorOnAll(0.1, 0.2)


def test_ThermalRelaxationErrorOnX():
    def test_model(thermal_error):
        assert isinstance(thermal_error, qibo.noise.NoiseModel)
        errorkeys = thermal_error.errors.keys()
        assert len(errorkeys) == 1 and list(errorkeys)[0] == qibo.gates.gates.X
        error = thermal_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.ThermalRelaxationError)
        assert len(error.options) == 4

    noise_model1 = noisemodels.ThermalRelaxationErrorOnX()
    test_model(noise_model1)
    noise_model2 = noisemodels.ThermalRelaxationErrorOnX(10, 10, 1)
    test_model(noise_model2)
    noise_model3 = noisemodels.ThermalRelaxationErrorOnX(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.ThermalRelaxationErrorOnX(0.1, 0.2)


def test_ThermalRelaxationErrorOnXAndRX():
    def test_model(thermal_error):
        assert isinstance(thermal_error, qibo.noise.NoiseModel)
        errorkeys = thermal_error.errors.keys()
        assert (
            len(errorkeys) == 2
            and qibo.gates.gates.X in list(errorkeys)
            and qibo.gates.gates.RX in list(errorkeys)
        )
        error = thermal_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.ThermalRelaxationError)
        assert len(error.options) == 4
        error = thermal_error.errors[qibo.gates.gates.RX][0][1]
        assert isinstance(error, qibo.noise.ThermalRelaxationError)
        assert len(error.options) == 4

    noise_model1 = noisemodels.ThermalRelaxationErrorOnXAndRX()
    test_model(noise_model1)
    noise_model2 = noisemodels.ThermalRelaxationErrorOnXAndRX(10, 10, 1)
    test_model(noise_model2)
    noise_model3 = noisemodels.ThermalRelaxationErrorOnXAndRX(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.ThermalRelaxationErrorOnXAndRX(0.1, 0.2)


def test_ThermalRelaxationErrorOnNonDiagonal():
    def test_model(thermal_error):
        assert isinstance(thermal_error, qibo.noise.NoiseModel)
        errorkeys = thermal_error.errors.keys()
        assert (
            len(errorkeys) == 3
            and qibo.gates.gates.X in list(errorkeys)
            and qibo.gates.gates.Y in list(errorkeys)
            and qibo.gates.gates.Unitary in list(errorkeys)
        )
        error = thermal_error.errors[qibo.gates.gates.X][0][1]
        assert isinstance(error, qibo.noise.ThermalRelaxationError)
        assert len(error.options) == 4
        error = thermal_error.errors[qibo.gates.gates.Y][0][1]
        assert isinstance(error, qibo.noise.ThermalRelaxationError)
        assert len(error.options) == 4
        condition, error = thermal_error.errors[qibo.gates.gates.Unitary][0][:2]
        assert not condition(qibo.gates.gates.Unitary(np.eye(2), 0))
        assert condition(qibo.gates.gates.Unitary(np.array([[0, 1], [1, 0]]), 0))
        assert isinstance(error, qibo.noise.ThermalRelaxationError)
        assert len(error.options) == 4

    noise_model1 = noisemodels.ThermalRelaxationErrorOnNonDiagonal()
    test_model(noise_model1)
    noise_model2 = noisemodels.ThermalRelaxationErrorOnNonDiagonal(10, 10, 1)
    test_model(noise_model2)
    noise_model3 = noisemodels.ThermalRelaxationErrorOnNonDiagonal(None)
    test_model(noise_model3)
    with pytest.raises(ValueError):
        noisemodels.ThermalRelaxationErrorOnNonDiagonal(0.1, 0.2)
