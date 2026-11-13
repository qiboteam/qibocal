import numpy as np
import pytest
from qibolab import Pulse

from qibocal.protocols.two_qubit_interaction.chevron.chevron import ChevronParameters
from qibocal.protocols.two_qubit_interaction.chevron.utils import chevron_sequence
from qibocal.protocols.two_qubit_interaction.utils import (
    determine_cz_hybridization_state,
)

PAIR = (0, 1)
"""Pair of the mock platform, ordered as (low, high) frequency qubit."""


def _set_11_20_branch(platform):
    """Lower the high qubit frequency below the |11>-|02> resonance.

    With f01_high + eta_high < f01_low, the |11>-|02> resonance is not reachable,
    while the |11>-|20> one still is.
    """
    qubit = platform.calibration.single_qubits[PAIR[1]].qubit
    eta = qubit.anharmonicity
    qubit.frequency_01 = 4.05e9
    qubit.frequency_12 = qubit.frequency_01 + eta


def _rx_qubits(sequence, platform):
    """Qubits receiving a drive pulse, in order of appearance."""
    drive = {platform.qubits[q].drive: q for q in PAIR}
    return [
        drive[ch] for ch, pulse in sequence if ch in drive and isinstance(pulse, Pulse)
    ]


def test_hybridization_11_02(platform):
    mapping, roots = determine_cz_hybridization_state(platform, PAIR)
    assert mapping == {0: PAIR[0], 2: PAIR[1]}
    # f_high(x) = f01_low - eta_high, with the mock flux coefficients
    np.testing.assert_allclose(roots, np.sort(np.roots([-2.0, 0.1, 0.101])), rtol=1e-10)


def test_hybridization_reversed_pair(platform):
    """Pairs are found in the calibration regardless of their order."""
    assert determine_cz_hybridization_state(platform, PAIR[::-1]) == (
        determine_cz_hybridization_state(platform, PAIR)
    )


def test_hybridization_11_20(platform):
    _set_11_20_branch(platform)
    mapping, roots = determine_cz_hybridization_state(platform, PAIR)
    assert mapping == {0: PAIR[1], 2: PAIR[0]}
    assert len(roots) == 2


@pytest.mark.parametrize(
    "qubit,field",
    [(1, "flux_coefficients"), (0, "frequency_12"), (1, "frequency_01")],
)
def test_hybridization_missing_calibration(platform, qubit, field):
    setattr(platform.calibration.single_qubits[qubit].qubit, field, None)
    with pytest.raises(ValueError):
        determine_cz_hybridization_state(platform, PAIR)


def test_hybridization_missing_pair(platform):
    platform.calibration.two_qubits.clear()
    with pytest.raises(ValueError):
        determine_cz_hybridization_state(platform, PAIR)


def test_chevron_sequence_default(platform):
    """By default the final RX targets the high qubit, without calibration data."""
    _set_11_20_branch(platform)
    platform.calibration.single_qubits[PAIR[1]].qubit.flux_coefficients = None
    sequence, *_ = chevron_sequence(platform, PAIR)
    assert _rx_qubits(sequence, platform) == [PAIR[0], PAIR[1], PAIR[1]]


@pytest.mark.parametrize("branch_11_20", [False, True])
def test_chevron_sequence_infer_hybridization(platform, branch_11_20):
    if branch_11_20:
        _set_11_20_branch(platform)
    sequence, *_ = chevron_sequence(platform, PAIR, infer_hybridization=True)
    state2_qubit = PAIR[0] if branch_11_20 else PAIR[1]
    assert _rx_qubits(sequence, platform) == [PAIR[0], PAIR[1], state2_qubit]


def test_chevron_sequence_iswap_ignores_inference(platform):
    """iSWAP does not require CZ calibration data, nor applies a final RX."""
    platform.calibration.single_qubits[PAIR[1]].qubit.flux_coefficients = None
    sequence, *_ = chevron_sequence(
        platform, PAIR, native="iSWAP", infer_hybridization=True
    )
    assert _rx_qubits(sequence, platform) == [PAIR[1]]


def test_chevron_parameters_ranges():
    params = ChevronParameters(
        amplitude=(0.1, 0.2, 0.05),
        duration=(10, 30, 10),
        amplitude_min=0.0,
        amplitude_max=1.0,
        amplitude_step=0.5,
    )
    # the new fields take precedence over the legacy ones
    assert params.amplitude_range == (0.1, 0.2, 0.05)
    np.testing.assert_allclose(params.duration_values, [10, 20])

    legacy = ChevronParameters(
        amplitude_min=0.1,
        amplitude_max=0.2,
        amplitude_step=0.05,
        duration_min=10,
        duration_max=30,
        duration_step=10,
    )
    assert legacy.amplitude_range == (0.1, 0.2, 0.05)
    assert legacy.duration_range == (10, 30, 10)

    with pytest.raises(ValueError):
        _ = ChevronParameters(amplitude_min=0.1, amplitude_max=0.2).amplitude_range
    with pytest.raises(ValueError):
        _ = ChevronParameters().duration_range
