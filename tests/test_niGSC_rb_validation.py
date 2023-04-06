import importlib

import numpy as np
import pandas as pd
import pytest
from qibo.config import PRECISION_TOL

from qibocal.calibrations.niGSC.basics.noisemodels import *
from qibocal.calibrations.niGSC.basics.rb_validation import *

test_module_names = [
    "Idrb",
    "paulisfilteredrb",
    "simulfilteredrb",
    "XIdrb",
    "Z3rb",
    "Z4rb",
]

test_noisemodels_list = [
    None,
    PauliErrorOnAll(),
    UnitaryErrorOnAll(),
    ThermalRelaxationErrorOnAll(),
]


@pytest.mark.parametrize("module_name", test_module_names)
@pytest.mark.parametrize("noise", test_noisemodels_list)
def test_fourier_transform(module_name, noise):
    nqubits = 1
    module = importlib.import_module(f"qibocal.calibrations.niGSC.{module_name}")
    irrep_info = module.irrep_info(nqubits)
    gate_group = module.gate_group(nqubits)

    f_transform = fourier_transform(gate_group, irrep_info, nqubits, noise)
    assert f_transform.shape == (4 * irrep_info[2], 4 * irrep_info[2])


@pytest.mark.parametrize("module_name", test_module_names)
def test_channel_twirl(module_name):
    nqubits = 1
    module = importlib.import_module(f"qibocal.calibrations.niGSC.{module_name}")
    irrep_info = module.irrep_info(nqubits)
    gate_group = module.gate_group(nqubits)

    dephasing_channel = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=complex
    )

    ch_twirl = channel_twirl(gate_group, nqubits, dephasing_channel, irrep_info[0])
    assert ch_twirl.shape == (4, 4)


@pytest.mark.parametrize("module_name", test_module_names)
@pytest.mark.parametrize("noise", test_noisemodels_list)
@pytest.mark.parametrize("N", [None, 10])
@pytest.mark.parametrize("with_coefficients", [True, False])
def test_filtered_decay_parameters(module_name, noise, with_coefficients, N):
    nqubits = 1

    result_parameters = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients, N
    )
    coefficients = result_parameters[0]
    decays = result_parameters[1]

    if with_coefficients:
        assert len(coefficients) == len(decays)
    else:
        assert len(coefficients) == 0 and len(decays) > 0


def test_pauli_validation():
    module_name = "paulisfilteredrb"
    nqubits = 1
    # Pauli noise: F(m) = 0.5 (1 - px - py)^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnNonDiagonal(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(coefficients[0], 0.5, rtol=1e-2)
    assert np.allclose(decays[0], 1 - px - py, rtol=1e-3)

    # Unitary noise: F(m) = 0.5 * r^m
    noise = UnitaryErrorOnNonDiagonal()
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(coefficients[0], 0.5, rtol=1e-2)

    # Thermal Relaxation: F(m) = 0.5 * (0.5[1 + exp(-t/T1)])^m
    t1 = np.random.uniform(0.0, 10.0)
    t2 = np.random.uniform(0.0, 2 * t1)
    coeff = np.random.uniform(1.0, 10.0)
    time = t1 / coeff
    a0 = np.random.uniform(0.0, 1.0)
    noise = ThermalRelaxationErrorOnNonDiagonal(t1, t2, time, a0)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(decays[0], (1 + np.exp(-time / t1)) / 2, rtol=1e-3)


def test_id_validation():
    module_name = "Idrb"
    nqubits = 1

    # Pauli noise: F(m) = 0.5 * 1^m + 0.5 (1 - 2px - 2py)^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnAll(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) <= 4 and len(decays) <= 4
    assert np.allclose(np.sum(coefficients), 1.0, rtol=1e-2)
    assert np.allclose(coefficients[:2], 0.5, rtol=1e-2)
    assert np.allclose(coefficients[2:], 0.0, rtol=1e-2)
    assert np.allclose(np.max(decays[:2]), 1.0, rtol=1e-3)
    assert np.allclose(np.min(decays[:2]), 1 - 2 * px - 2 * py, rtol=1e-3)

    # Unitary noise: F(m) = 0.5 * 1^m + ...
    noise = UnitaryErrorOnAll(t=0.1)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) <= 4 and len(decays) <= 4
    assert np.allclose(np.sum(coefficients), 1.0, rtol=1e-2)
    assert np.max(coefficients) > 0.5 - PRECISION_TOL
    assert np.allclose(np.max(decays), 1.0, rtol=0.1)

    # Thermal Relaxation: F(m) = 0.5 * 1^m + 0.5 * 1^m
    noise = ThermalRelaxationErrorOnAll()
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert np.allclose(np.sum(coefficients), 1.0, rtol=1e-2)
    assert np.allclose(np.max(decays), 1.0, rtol=1e-3)


def test_xid_validation():
    module_name = "XIdrb"
    nqubits = 1

    # Pauli noise: F(m) = 0.5 (1 - px - py)^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnX(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert np.allclose(np.sum(coefficients), 0.5, rtol=1e-2)
    assert np.allclose(coefficients[0], 0.5, rtol=1e-2)
    assert np.allclose(decays[0], 1 - px - py, rtol=1e-3)

    # Unitary noise: F(m) = a1 * r1^m + a2 * r2^m
    noise = UnitaryErrorOnX(t=0.1)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert np.allclose(np.sum(coefficients), 0.5, rtol=0.1)
    assert len(coefficients) == 2
    assert len(decays) == 2

    # Thermal Relaxation: F(m) = 0.5 * (0.5[1 + exp(-t/T1)])^m
    t1 = np.random.uniform(0.0, 10.0)
    t2 = np.random.uniform(0.0, 2 * t1)
    coeff = np.random.uniform(1.0, 10.0)
    time = t1 / coeff
    a0 = np.random.uniform(0.0, 1.0)
    noise = ThermalRelaxationErrorOnX(t1, t2, time, a0)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert np.allclose(np.sum(coefficients), 0.5, rtol=1e-2)
    assert np.allclose(coefficients[0], 0.5, rtol=1e-2)
    assert np.allclose(decays[0], (1 + np.exp(-time / t1)) / 2, rtol=1e-2)


def test_z4_validation():
    module_name = "Z4rb"
    nqubits = 1

    # Gate-independent Pauli noise: F(m) = a * r^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnAll(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert coefficients[0] < 0.5 + PRECISION_TOL
    expected_decay = 1 - 2 * px - py - pz
    assert len(coefficients) == 1 and len(decays) == 1
    assert coefficients[0] < 0.5 + PRECISION_TOL
    assert np.allclose(decays[0], expected_decay, rtol=1e-3)

    # Gate-dependent Pauli noise: F(m) = a * r^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnXAndRX(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert coefficients[0] < 0.5 + PRECISION_TOL
    expected_decay = (
        0.25
        + px**2
        + 0.0625 * py**2
        + 0.0625 * pz**2
        + 0.875 * py * pz
        + px * py
        + px * pz
        - px
        - 0.5 * py
        - 0.5 * pz
    )
    expected_decay = (
        1j * np.sqrt(np.abs(expected_decay))
        if expected_decay < -PRECISION_TOL
        else np.sqrt(np.abs(expected_decay))
    )
    expected_decay += 0.5 - (0.5 * px) - (0.25 * py) - (0.25 * pz)
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(decays[0], expected_decay, rtol=1e-3)

    # Unitary noise: F(m) = a1 * r1^m
    noise = UnitaryErrorOnXAndRX(t=0.1)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1
    assert np.iscomplex(decays[0])

    # Thermal Relaxation: F(m) = 0.5 * (0.5[1 + exp(-t/T1)])^m
    noise = ThermalRelaxationErrorOnXAndRX()
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1


def test_z3_validation():
    module_name = "Z3rb"
    nqubits = 1

    # Pauli noise: F(m) = a * r^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnXAndRX(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert coefficients[0] < 0.5 + PRECISION_TOL
    expected_decay = (
        0.25
        + px**2
        + 0.03 * py**2
        + 0.03 * pz**2
        + 0.95 * py * pz
        + px * py
        + px * pz
        - px
        - 0.5 * py
        - 0.5 * pz
    )
    expected_decay = (
        1j * np.sqrt(np.abs(expected_decay))
        if expected_decay < -PRECISION_TOL
        else np.sqrt(np.abs(expected_decay))
    )
    expected_decay += 0.5 - (1 / 3) * px - (1 / 6) * py - (1 / 6) * pz
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(decays[0], expected_decay, rtol=0.1)

    # Unitary noise: F(m) = a1 * r1^m
    noise = UnitaryErrorOnXAndRX(t=0.1)
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1
    assert np.iscomplex(decays[0])

    # Thermal Relaxation: F(m) = 0.5 * (0.5[1 + exp(-t/T1)])^m
    noise = ThermalRelaxationErrorOnXAndRX()
    coefficients, decays = filtered_decay_parameters(
        module_name, nqubits, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1
