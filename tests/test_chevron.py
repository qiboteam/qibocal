import numpy as np
import pytest
from qibolab import Pulse

from qibocal.protocols.two_qubit_interaction import utils
from qibocal.protocols.two_qubit_interaction.chevron.chevron import (
    ChevronData,
    ChevronParameters,
    _fit,
    _resonance_amplitude,
    _set_interaction_branch,
    invert_level2,
)
from qibocal.protocols.two_qubit_interaction.chevron.utils import (
    chevron_fit,
    chevron_sequence,
    cz_level2_qubit,
)
from qibocal.protocols.two_qubit_interaction.utils import (
    determine_cz_hybridization_state,
)

PAIR = (0, 1)
"""Pair of the mock platform, ordered as (low, high) frequency qubit."""
ROOTS_11_02 = np.sort(np.roots([-2.0, 0.1, 0.101]))
"""|11>-|02> resonances of the mock platform, approximately -0.201 and 0.251."""
ROOTS_11_20 = np.sort(np.roots([-2.0, 0.1, 0.301]))
"""|11>-|20> resonances of the mock platform, approximately -0.364 and 0.414."""


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
    np.testing.assert_allclose(roots, ROOTS_11_02, rtol=1e-10)


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
    "amplitude_range,level2_qubit,roots",
    [
        ((0.1, 0.6), PAIR[1], ROOTS_11_02[1:]),
        ((-0.6, -0.1), PAIR[1], ROOTS_11_02[:1]),
        # only the |11>-|20> resonance is scanned, for both amplitude signs
        ((0.3, 0.6), PAIR[0], ROOTS_11_20[1:]),
        ((-0.6, -0.3), PAIR[0], ROOTS_11_20[:1]),
        # the order of the range limits does not matter
        ((0.6, 0.3), PAIR[0], ROOTS_11_20[1:]),
    ],
)
def test_hybridization_amplitude_range(platform, amplitude_range, level2_qubit, roots):
    """Only resonances within the amplitude range are reachable."""
    mapping, found = determine_cz_hybridization_state(platform, PAIR, amplitude_range)
    assert mapping[2] == level2_qubit
    np.testing.assert_allclose(found, roots, rtol=1e-10)


@pytest.mark.parametrize("amplitude_range", [(0.05, 0.15), None])
def test_hybridization_unreachable(platform, amplitude_range):
    if amplitude_range is None:
        # with f_high always above both resonances, none of them is ever reached
        platform.calibration.single_qubits[PAIR[1]].qubit.flux_coefficients = [
            2.0,
            0.0,
            0.0,
        ]
    with pytest.raises(ValueError, match="No |11>-|02> or |11>-|20> resonance"):
        determine_cz_hybridization_state(platform, PAIR, amplitude_range)


def test_hybridization_tie(platform, monkeypatch):
    """Resonances equally distant from zero do not select any branch."""
    roots = iter([[0.3], [-0.3]])
    monkeypatch.setattr(utils, "solve_quadratic", lambda *args: next(roots))
    with pytest.raises(ValueError, match="equally distant"):
        determine_cz_hybridization_state(platform, PAIR)


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


@pytest.mark.parametrize(
    "kwargs,level2_qubit",
    [
        ({}, PAIR[1]),
        ({"cz_branch": "11-20"}, PAIR[0]),
        ({"infer_hybridization": True}, PAIR[1]),
        ({"infer_hybridization": True, "amplitude_range": (0.3, 0.6)}, PAIR[0]),
        # the explicit branch is ignored when inferring it
        ({"infer_hybridization": True, "cz_branch": "11-20"}, PAIR[1]),
        ({"native": "iSWAP", "infer_hybridization": True}, None),
    ],
)
def test_cz_level2_qubit(platform, kwargs, level2_qubit):
    assert cz_level2_qubit(platform, PAIR, **kwargs) == level2_qubit


def test_cz_level2_qubit_no_calibration(platform):
    """The explicit branch selection does not require calibration data."""
    platform.calibration.single_qubits[PAIR[1]].qubit.flux_coefficients = None
    assert cz_level2_qubit(platform, PAIR, cz_branch="11-20") == PAIR[0]
    with pytest.raises(ValueError):
        cz_level2_qubit(platform, PAIR, infer_hybridization=True)


@pytest.mark.parametrize("level2_qubit", [None, *PAIR])
def test_chevron_sequence_final_rx(platform, level2_qubit):
    """The final RX targets the level 2 qubit, the high one by default."""
    sequence, *_ = chevron_sequence(platform, PAIR, level2_qubit=level2_qubit)
    final = PAIR[1] if level2_qubit is None else level2_qubit
    assert _rx_qubits(sequence, platform) == [PAIR[0], PAIR[1], final]


def test_chevron_sequence_iswap(platform):
    """iSWAP only prepares the high qubit, without any final RX."""
    sequence, *_ = chevron_sequence(platform, PAIR, native="iSWAP")
    assert _rx_qubits(sequence, platform) == [PAIR[1]]


@pytest.mark.parametrize(
    "params,amplitude,level2_qubit,anharmonicity",
    [
        ({}, (0.1, 0.6, 0.1), PAIR[1], -0.1),
        ({"cz_branch": "11-20"}, (0.1, 0.6, 0.1), PAIR[0], 0.1),
        ({"infer_hybridization": True}, (0.3, 0.6, 0.1), PAIR[0], 0.1),
        ({"infer_hybridization": True}, (-0.3, -0.1, 0.1), PAIR[1], -0.1),
        ({"native": "iSWAP"}, (0.1, 0.6, 0.1), None, 0.0),
    ],
)
def test_set_interaction_branch(
    platform, params, amplitude, level2_qubit, anharmonicity
):
    """The level 2 qubit and the detuning anharmonicity term follow the branch."""
    params = ChevronParameters(amplitude=amplitude, duration=(10, 50, 10), **params)
    data = ChevronData(
        native=params.native,
        amplitude=params.amplitude_values.tolist(),
        duration=params.duration_values.tolist(),
    )
    data.sorted_pairs = [PAIR]
    _set_interaction_branch(params, platform, data)
    assert data.level2_qubit.get(PAIR) == level2_qubit
    assert data.anharmonicity[PAIR] == pytest.approx(anharmonicity)


@pytest.mark.parametrize("level2_qubit", [None, *PAIR])
def test_invert_level2(level2_qubit):
    data = ChevronData(native="CZ", amplitude=[], duration=[])
    if level2_qubit is not None:
        data.level2_qubit[PAIR] = level2_qubit
    signals = [np.full(3, 0.2), np.full(3, 0.3)]
    inverted = invert_level2(data, PAIR, signals)
    expected = [1 - s if q == level2_qubit else s for q, s in zip(PAIR, signals)]
    np.testing.assert_allclose(inverted, expected)


@pytest.mark.parametrize(
    "amplitudes,expected",
    [
        ([0.1, 0.3], 0.2),
        ([-0.3, -0.1], -0.2),
        # within both ranges, or none of them, the positive one is kept
        ([-0.3, 0.3], 0.2),
        ([0.5, 0.6], 0.2),
    ],
)
def test_resonance_amplitude(amplitudes, expected):
    assert _resonance_amplitude(0.2, amplitudes) == expected


@pytest.mark.parametrize("sign", [1, -1])
def test_fit_amplitude_sign(sign):
    """The fitted CZ amplitude lies within the scanned amplitudes."""
    detuning, alpha, g, phase = 0.1, -2.0, 0.05, 0.0
    data = ChevronData(
        native="CZ",
        amplitude=(sign * np.arange(0.1, 0.35, 0.005)).tolist(),
        duration=np.arange(10, 100, 2).tolist(),
    )
    data.sorted_pairs = [PAIR]
    data.detuning[PAIR], data.anharmonicity[PAIR] = 0.2, -0.1
    data.flux_coefficient[PAIR] = alpha
    signal = chevron_fit(data.grid, detuning, alpha, g, phase).reshape(
        len(data.amplitude), len(data.duration)
    )
    data.data[PAIR] = np.stack([signal.T, signal.T])

    results = _fit(data)
    np.testing.assert_allclose(
        results.amplitude[PAIR], 2 * [sign * np.sqrt(-detuning / alpha)], rtol=1e-4
    )


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
