import numpy as np
from scipy.sparse import lil_matrix

from qibocal.calibration.calibration import (
    CALIBRATION,
    QubitCalibration,
    TwoQubitCalibration,
)
from qibocal.calibration.platform import Calibration


def assert_close_lil_sparse(a, b, atol=1e-8, rtol=1e-5):
    a_csr, b_csr = a.tocsr(), b.tocsr()
    return (
        a_csr.shape == b_csr.shape
        and np.allclose(a_csr.data, b_csr.data, atol=atol, rtol=rtol)
        and np.array_equal(a_csr.indices, b_csr.indices)
        and np.array_equal(a_csr.indptr, a_csr.indptr)
    )


def test_serialization_single_qubits(tmp_path):
    """Testing serialization forn single qubits."""

    single_qubits = {}
    for i in range(2):
        single_qubits[i] = QubitCalibration()

        single_qubits[i].resonator.bare_frequency = 7e9
        single_qubits[i].resonator.dressed_frequency = 7.001e9
        assert single_qubits[i].resonator.dispersive_shift == -0.001e9
        single_qubits[i].qubit.frequency_01 = 5e9
        assert single_qubits[i].qubit.anharmonicity == 0.0
        single_qubits[i].qubit.frequency_12 = 4.8e9
        single_qubits[i].qubit.maximum_frequency = single_qubits[0].qubit.frequency_01
        single_qubits[i].t1 = (10e6, 1e6)

        assert single_qubits[i].qubit.anharmonicity == -0.2e9
        assert single_qubits[i].qubit.charging_energy == 0.2e9
        assert (
            single_qubits[i].qubit.josephson_energy
            == (
                single_qubits[i].qubit.maximum_frequency
                + single_qubits[i].qubit.charging_energy
            )
            ** 2
            / 8
            / single_qubits[i].qubit.charging_energy
        )

        single_qubits[i].readout.fidelity = 0.9
        assert single_qubits[i].readout.assignment_fidelity == 0.95

    cal = Calibration(single_qubits=single_qubits)
    assert cal.nqubits == 2
    assert cal.qubits == list(range(2))

    cal.dump(tmp_path)
    new_cal = cal.model_validate_json((tmp_path / CALIBRATION).read_text())

    assert new_cal == cal


def test_serialization_qubit_pairs(tmp_path):
    """Testing serialization for qubit pairs."""

    cal = Calibration(
        two_qubits={
            (0, 1): TwoQubitCalibration(rb_fidelity=[0.99, 0.1], coupling=[0.5, 0.01])
        }
    )
    cal.dump(tmp_path)
    new_cal = cal.model_validate_json((tmp_path / CALIBRATION).read_text())
    assert new_cal == cal


def test_serialization_crosstalk_matrices(tmp_path):
    """Testing serialization from flux crosstalk matrix."""

    single_qubits = {}
    for i in range(5):
        single_qubits[f"A{i}"] = QubitCalibration()

    cal = Calibration(single_qubits=single_qubits)

    assert cal.flux_crosstalk_matrix["A0", "A1"] == 0
    assert cal.microwave_crosstalk_matrix["A0", "A1"] == np.inf

    cal.flux_crosstalk_matrix["A0", "A1"] = 1
    assert cal.flux_crosstalk_matrix["A0", "A1"] == 1
    cal.microwave_crosstalk_matrix["A0", "A1"] = 1
    assert cal.microwave_crosstalk_matrix["A0", "A1"] == 1

    cal.flux_crosstalk_matrix["A3", "A4"] = 99

    cal.dump(tmp_path)
    new_cal = cal.model_validate_json((tmp_path / CALIBRATION).read_text())

    cal.microwave_crosstalk_matrix["A3", "A4"] = 99

    assert cal.flux_crosstalk_matrix["A3", "A4"] == 99
    assert cal.microwave_crosstalk_matrix["A3", "A4"] == 99

    assert cal.flux_crosstalk_matrix == new_cal.flux_crosstalk_matrix
    assert cal.microwave_crosstalk_matrix != new_cal.microwave_crosstalk_matrix


def test_serialization_readout(tmp_path):
    """Test serialization for readout mitigation matrix."""

    single_qubits = {}
    for i in range(5):
        single_qubits[f"A{i}"] = QubitCalibration()

    cal = Calibration(single_qubits=single_qubits)

    cal.readout_mitigation_matrix = lil_matrix((2**cal.nqubits, 2**cal.nqubits))

    cal.readout_mitigation_matrix[0, 5] = 42
    cal.dump(tmp_path)
    new_cal = cal.model_validate_json((tmp_path / CALIBRATION).read_text())
    assert_close_lil_sparse(
        new_cal.readout_mitigation_matrix, new_cal.readout_mitigation_matrix
    )
