from pathlib import Path

import numpy as np
import pytest
from conftest import approx_for_regression

from qibocal.protocols.two_qubit_interaction.chevron.chevron import (
    ChevronData,
    ChevronResults,
    _fit,
)
from qibocal.protocols.two_qubit_interaction.chevron.utils import fit_gate_parameters

PATH_TESTING_DATA = Path(__file__).resolve().parent / "tests_data"


def test_chevron_fit():
    folder = PATH_TESTING_DATA / "chevron-0"
    data, expected = ChevronData.load(folder), ChevronResults.load(folder)
    fitted = _fit(data)
    pair = (2, 4)
    assert fitted.amplitude[pair] == approx_for_regression(expected.amplitude[pair])
    assert fitted.duration[pair] == approx_for_regression(expected.duration[pair])


def chevron(duration, amplitude, amplitude0, slope, g):
    """Excited-state population of a chevron, shape ``(len(amplitude), len(duration))``.

    Eq. 14.80 of Manenti and Motta, Quantum Information Science (Oxford University
    Press, 2023), with the detuning ``2 pi slope (amplitude - amplitude0)`` linear
    around the resonance. The population returns at ``pi / g``.
    """
    delta = 2 * np.pi * slope * (amplitude[:, None] - amplitude0)
    omega = np.sqrt(delta**2 + 4 * g**2)
    return delta**2 / omega**2 + 4 * g**2 / omega**2 * np.cos(omega * duration / 2) ** 2


@pytest.mark.parametrize("native,gate_time", [("CZ", 40.0), ("iSWAP", 20.0)])
def test_chevron_fit_synthetic(native, gate_time):
    duration = np.arange(4, 100, 1.0)
    amplitude = np.arange(0.24, 0.31, 0.001)
    pe = chevron(duration, amplitude, amplitude0=0.275, slope=4.0, g=np.pi / 40)
    low = pe if native == "CZ" else 1 - pe
    data = np.stack([low, pe]).transpose(0, 2, 1)
    data = data + np.random.default_rng(0).normal(0, 0.03, data.shape)
    amp, dur = fit_gate_parameters(data, amplitude, duration, native)
    assert amp == pytest.approx(0.275, abs=0.001)
    assert dur == pytest.approx(gate_time, rel=0.03)
