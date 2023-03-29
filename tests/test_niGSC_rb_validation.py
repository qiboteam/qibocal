import numpy as np
import pandas as pd
import pytest
from plotly.graph_objects import Figure
from qibo import gates
from qibo.config import PRECISION_TOL
from qibo.noise import NoiseModel

from qibocal.calibrations.niGSC import Idrb, XIdrb, Z3rb, Z4rb, paulisfilteredrb
from qibocal.calibrations.niGSC.basics.circuitfactory import (
    SingleCliffordsFactory,
    ZkFilteredCircuitFactory,
)
from qibocal.calibrations.niGSC.basics.noisemodels import *
from qibocal.calibrations.niGSC.basics.rb_validation import *

test_factories_list = [
    ZkFilteredCircuitFactory(1, [1]),
    SingleCliffordsFactory(1, [1]),
    XIdrb.ModuleFactory(1, [1]),
    Idrb.ModuleFactory(1, [1]),
    paulisfilteredrb.ModuleFactory(1, [1]),
    Z3rb.ModuleFactory(1, [1]),
    Z4rb.ModuleFactory(1, [1]),
]

test_noisemodels_list = [
    PauliErrorOnAll(),
    UnitaryErrorOnAll(),
    ThermalRelaxationErrorOnAll(),
]


@pytest.mark.parametrize("factory", test_factories_list)
def test_irrep_info(factory):
    basis, index, size, multiplicity = irrep_info(factory)
    assert isinstance(basis, np.ndarray)
    assert isinstance(index, int)
    assert isinstance(size, int)
    assert isinstance(multiplicity, int)
    assert basis.shape == (4, 4)
    with pytest.raises(TypeError):
        irrep_info(list(factory))


@pytest.mark.parametrize("factory", test_factories_list)
@pytest.mark.parametrize("noise", test_noisemodels_list)
@pytest.mark.parametrize("N", [None, 10])
@pytest.mark.parametrize("ideal", [True, False])
def test_fourier_transform(factory, noise, N, ideal):
    noise = noise if ideal else None

    size = irrep_info(factory)[2]
    f_transform = fourier_transform(factory, noise, N, ideal)
    assert f_transform.shape == (4 * size, 4 * size)

    with pytest.raises(TypeError):
        fourier_transform(list(factory), noise, N, ideal)


@pytest.mark.parametrize("factory", test_factories_list)
@pytest.mark.parametrize("N", [None, 10])
def test_channel_twirl(factory, N):
    dephasing_channel = np.array(
        [[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], dtype=complex
    )

    ch_twirl = channel_twirl(factory, dephasing_channel, N)
    assert ch_twirl.shape == (4, 4)


@pytest.mark.parametrize("factory", test_factories_list)
@pytest.mark.parametrize("noise", test_noisemodels_list)
@pytest.mark.parametrize("N", [None, 10])
@pytest.mark.parametrize("with_coefficients", [True, False])
def test_filtered_decay_parameters(factory, noise, with_coefficients, N):
    result_parameters = filtered_decay_parameters(factory, noise, with_coefficients, N)
    coefficients = result_parameters[0] if with_coefficients else []
    decays = result_parameters[1] if with_coefficients else result_parameters

    if with_coefficients:
        assert len(coefficients) == len(decays)

    with pytest.raises(TypeError):
        filtered_decay_parameters(list(factory), noise, with_coefficients, N)


def test_pauli_validation():
    factory = paulisfilteredrb.ModuleFactory(1, [1])

    # Pauli noise: F(m) = 0.5 (1 - px - py)^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnNonDiagonal(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(coefficients[0], 0.5, rtol=1e-2)
    assert np.allclose(decays[0], 1 - px - py, rtol=1e-3)

    # Unitary noise: F(m) = 0.5 * r^m
    noise = UnitaryErrorOnNonDiagonal()
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
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
        factory, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(decays[0], (1 + np.exp(-time / t1)) / 2, rtol=1e-3)


def test_id_validation():
    factory = Idrb.ModuleFactory(1, [1])

    # Pauli noise: F(m) = 0.5 * 1^m + 0.5 (1 - 2px - 2py)^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnAll(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
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
        factory, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) <= 4 and len(decays) <= 4
    assert np.allclose(np.sum(coefficients), 1.0, rtol=1e-2)
    assert np.max(coefficients) > 0.5 - PRECISION_TOL
    assert np.allclose(np.max(decays), 1.0, rtol=0.1)

    # Thermal Relaxation: F(m) = 0.5 * 1^m + 0.5 * 1^m
    noise = ThermalRelaxationErrorOnAll()
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
    )
    assert np.allclose(np.sum(coefficients), 1.0, rtol=1e-2)
    assert np.allclose(np.max(decays), 1.0, rtol=1e-3)


def test_xid_validation():
    factory = XIdrb.ModuleFactory(1, [1])

    # Pauli noise: F(m) = 0.5 (1 - px - py)^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnX(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
    )
    assert np.allclose(np.sum(coefficients), 0.5, rtol=1e-2)
    assert np.allclose(coefficients[0], 0.5, rtol=1e-2)
    assert np.allclose(decays[0], 1 - px - py, rtol=1e-3)

    # Unitary noise: F(m) = a1 * r1^m + a2 * r2^m
    noise = UnitaryErrorOnX(t=0.1)
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
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
        factory, noise, with_coefficients=True, N=None
    )
    assert np.allclose(np.sum(coefficients), 0.5, rtol=1e-2)
    assert np.allclose(coefficients[0], 0.5, rtol=1e-2)
    assert np.allclose(decays[0], (1 + np.exp(-time / t1)) / 2, rtol=1e-2)


def test_z4_validation():
    factory = Z4rb.ModuleFactory(1, [1])

    # Pauli noise: F(m) = a * r^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnXAndRX(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
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
    expected_decay += 0.5 - 0.5 * px - 0.25 * py - 0.25 * pz
    assert len(coefficients) == 1 and len(decays) == 1
    assert np.allclose(decays[0], expected_decay, rtol=1e-3)

    # Unitary noise: F(m) = a1 * r1^m
    noise = UnitaryErrorOnXAndRX(t=0.1)
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1
    assert np.iscomplex(decays[0])

    # Thermal Relaxation: F(m) = 0.5 * (0.5[1 + exp(-t/T1)])^m
    noise = ThermalRelaxationErrorOnXAndRX()
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1


def test_z3_validation():
    factory = Z3rb.ModuleFactory(1, [1])

    # Pauli noise: F(m) = a * r^m
    px, py, pz = np.random.uniform(0, 0.15, size=3)
    noise = PauliErrorOnXAndRX(px, py, pz)
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
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
        factory, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1
    assert np.iscomplex(decays[0])

    # Thermal Relaxation: F(m) = 0.5 * (0.5[1 + exp(-t/T1)])^m
    noise = ThermalRelaxationErrorOnXAndRX()
    coefficients, decays = filtered_decay_parameters(
        factory, noise, with_coefficients=True, N=None
    )
    assert len(coefficients) == 1
    assert len(decays) == 1
