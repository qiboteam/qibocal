from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.resonator_spectroscopy import resonator_spectroscopy
from qibocal.protocols.utils import HZ_TO_GHZ, PowerLevel, table_dict, table_html


@dataclass
class ResonatorTWPAPowerParameters(Parameters):
    """ResonatorTWPAPower runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    twpa_pow_width: int
    """Width for TPWA power sweep (dBm)."""
    twpa_pow_step: int
    """TPWA power step (dBm)."""
    power_level: PowerLevel
    """resonator Power regime (low or high)."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""

    def __post_init__(self):
        self.power_level = PowerLevel(self.power_level)


@dataclass
class ResonatorTWPAPowerResults(Results):
    """ResonatorTWPAPower outputs."""

    twpa_power: dict[QubitId, float] = field(default_factory=dict)
    """TWPA frequency [GHz] for each qubit."""
    frequency: Optional[dict[QubitId, float]] = field(default_factory=dict)
    """Readout frequency [GHz] for each qubit."""
    bare_frequency: Optional[dict[QubitId, float]] = field(default_factory=dict)
    """Bare frequency [GHz] for each qubit."""


ResonatorTWPAPowerType = np.dtype(
    [
        ("freq", np.float64),
        ("twpa_pow", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for Resonator TWPA Power."""


@dataclass
class ResonatorTWPAPowerData(Data):
    """ResonatorTWPAPower data acquisition."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[ResonatorTWPAPowerType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""
    power_level: Optional[PowerLevel] = None
    """Power regime of the resonator."""

    @classmethod
    def load(cls, path):
        obj = super().load(path)
        # Instantiate PowerLevel object
        if obj.power_level is not None:  # pylint: disable=E1101
            obj.power_level = PowerLevel(obj.power_level)  # pylint: disable=E1101
        return obj


def _acquisition(
    params: ResonatorTWPAPowerParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ResonatorTWPAPowerData:
    r"""
    Data acquisition for TWPA power optmization using SNR.
    This protocol perform a classification protocol for twpa powers
    in the range [twpa_power - frequency_width / 2, twpa_power + frequency_width / 2]
    with step frequency_step.

    Args:
        params (:class:`ResonatorTWPAPowerParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`ResonatorTWPAPowerData`)
    """

    data = ResonatorTWPAPowerData(
        power_level=params.power_level,
        resonator_type=platform.resonator_type,
    )

    TWPAPower_range = np.arange(
        -params.twpa_pow_width / 2, params.twpa_pow_width / 2, params.twpa_pow_step
    )

    initial_twpa_pow = {}
    for qubit in targets:
        initial_twpa_pow[qubit] = float(
            platform.qubits[qubit].twpa.local_oscillator.power
        )

    for _pow in TWPAPower_range:
        for qubit in targets:
            platform.qubits[qubit].twpa.local_oscillator.power = (
                initial_twpa_pow[qubit] + _pow
            )

        resonator_spectroscopy_data, _ = resonator_spectroscopy.acquisition(
            resonator_spectroscopy.parameters_type.load(
                {
                    "freq_width": params.freq_width,
                    "freq_step": params.freq_step,
                    "power_level": params.power_level,
                    "relaxation_time": params.relaxation_time,
                    "nshots": params.nshots,
                }
            ),
            platform,
            targets,
        )

        for qubit in targets:
            data.register_qubit(
                ResonatorTWPAPowerType,
                (qubit),
                dict(
                    signal=resonator_spectroscopy_data.data[qubit].signal,
                    phase=resonator_spectroscopy_data.data[qubit].phase,
                    freq=resonator_spectroscopy_data.data[qubit].freq,
                    twpa_pow=_pow + initial_twpa_pow[qubit],
                ),
            )

    return data


def _fit(data: ResonatorTWPAPowerData, fit_type="att") -> ResonatorTWPAPowerResults:
    """Post-processing function for ResonatorTWPASpectroscopy."""
    qubits = data.qubits
    bare_frequency = {}
    frequency = {}
    twpa_power = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        if data.resonator_type == "3D":
            index_best_pow = np.argmax(data_qubit["signal"])
        else:
            index_best_pow = np.argmin(data_qubit["signal"])
        twpa_power[qubit] = data_qubit["twpa_pow"][index_best_pow]

        if data.power_level is PowerLevel.high:
            bare_frequency[qubit] = data_qubit["freq"][index_best_pow]
        else:
            frequency[qubit] = data_qubit["freq"][index_best_pow]

    if data.power_level is PowerLevel.high:
        return ResonatorTWPAPowerResults(
            twpa_power=twpa_power,
            bare_frequency=bare_frequency,
        )
    else:
        return ResonatorTWPAPowerResults(
            twpa_power=twpa_power,
            frequency=frequency,
        )


def _plot(data: ResonatorTWPAPowerData, fit: ResonatorTWPAPowerResults, target):
    """Plotting for ResonatorTWPAPower."""

    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Signal [a.u.]",
            "Phase [rad]",
        ),
    )

    fitting_report = ""
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    powers = qubit_data.twpa_pow

    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=powers,
            z=qubit_data.signal,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Frequency [GHz]", row=1, col=1)
    fig.update_yaxes(title_text="TWPA Power [dBm]", row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=frequencies,
            y=powers,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Frequency [GHz]", row=1, col=2)
    fig.update_yaxes(title_text="TWPA Power [dBm]", row=1, col=2)

    if fit is not None:
        label_1 = "TWPA Power"
        twpa_power = np.round(fit.twpa_power[target])
        if target in fit.bare_frequency:
            label_2 = "High Power Resonator Frequency [Hz]"
            resonator_frequency = np.round(fit.bare_frequency[target])
        else:
            label_2 = "Low Power Resonator Frequency [Hz]"
            resonator_frequency = np.round(fit.frequency[target])

        summary = table_dict(
            target,
            [
                label_2,
                label_1,
            ],
            [
                resonator_frequency,
                twpa_power,
            ],
        )

        fitting_report = table_html(summary)

    fig.update_layout(
        showlegend=False,
    )

    figures.append(fig)

    return figures, fitting_report


twpa_power_snr = Routine(_acquisition, _fit, _plot)
"""Resonator TWPA Power Routine object."""
