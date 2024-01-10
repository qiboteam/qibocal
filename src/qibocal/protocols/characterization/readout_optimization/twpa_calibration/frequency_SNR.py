from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.protocols.characterization import resonator_spectroscopy
from qibocal.protocols.characterization.utils import (
    HZ_TO_GHZ,
    V_TO_UV,
    PowerLevel,
    table_dict,
    table_html,
)


@dataclass
class ResonatorTWPAFrequencyParameters(Parameters):
    """ResonatorTWPAFrequency runcard inputs."""

    freq_width: int
    """Width for frequency sweep relative to the readout frequency (Hz)."""
    freq_step: int
    """Frequency step for sweep (Hz)."""
    min_twpa_freq: int
    """TPWA frequency minimum value (Hz)."""
    max_twpa_freq: int
    """TPWA frequency maximum value (Hz)."""
    step_twpa_freq: int
    """TPWA frequency step (Hz)."""
    power_level: PowerLevel
    """Power regime (low or high). If low the readout frequency will be updated.
    If high both the readout frequency and the bare resonator frequency will be updated."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""

    def __post_init__(self):
        self.power_level = PowerLevel(self.power_level)


@dataclass
class ResonatorTWPAFrequencyResults(Results):
    """ResonatorTWPAFrequency outputs."""

    twpa_frequency: dict[QubitId, float] = field(metadata=dict(update="twpa_frequency"))
    """TWPA frequency [GHz] for each qubit."""

    frequency: Optional[dict[QubitId, float]] = field(
        default_factory=dict, metadata=dict(update="readout_frequency")
    )
    """Readout frequency [GHz] for each qubit."""

    bare_frequency: Optional[dict[QubitId, float]] = field(
        default_factory=dict, metadata=dict(update="bare_resonator_frequency")
    )
    """Bare frequency [GHz] for each qubit."""


ResonatorTWPAFrequencyType = np.dtype(
    [
        ("freq", np.float64),
        ("twpa_freq", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for Resonator TWPA Frequency."""


@dataclass
class ResonatorTWPAFrequencyData(Data):
    """ResonatorTWPAFrequency data acquisition."""

    resonator_type: str
    """Resonator type."""
    data: dict[QubitId, npt.NDArray[ResonatorTWPAFrequencyType]] = field(
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

    def register_qubit(self, qubit, freq, twpa_freq, signal, phase):
        """Store output for single qubit."""
        size = len(freq)
        ar = np.empty(size, dtype=ResonatorTWPAFrequencyType)
        ar["freq"] = freq
        ar["twpa_freq"] = np.array([twpa_freq] * size)
        ar["signal"] = signal
        ar["phase"] = phase
        if qubit in self.data:
            self.data[qubit] = np.rec.array(np.concatenate((self.data[qubit], ar)))
        else:
            self.data[qubit] = np.rec.array(ar)


def _acquisition(
    params: ResonatorTWPAFrequencyParameters,
    platform: Platform,
    qubits: Qubits,
) -> ResonatorTWPAFrequencyData:
    r"""
    Data acquisition for TWPA frequency optmization using SNR.
    This protocol perform a classification protocol for twpa frequencies
    in the range [twpa_frequency - frequency_width / 2, twpa_frequency + frequency_width / 2]
    with step frequency_step.

    Args:
        params (:class:`ResonatorTWPAFrequencyParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`ResonatorTWPAFrequencyData`)
    """

    data = ResonatorTWPAFrequencyData(
        power_level=params.power_level,
        resonator_type=platform.resonator_type,
    )

    TWPAFrequency_range = np.arange(
        params.min_twpa_freq, params.max_twpa_freq, params.step_twpa_freq
    )

    for _freq in TWPAFrequency_range:
        for z in qubits:
            qubits[z].twpa.local_oscillator.frequency = _freq

        resonator_spectroscopy_data = resonator_spectroscopy._acquisition(
            resonator_spectroscopy.ResonatorSpectroscopyParameters.load(
                {
                    "freq_width": params.freq_width,
                    "freq_step": params.freq_step,
                    "power_level": params.power_level,
                    "nshots": params.nshots,
                }
            ),
            platform,
            qubits,
        )

        for qubit in qubits:
            data.register_qubit(
                qubit,
                signal=resonator_spectroscopy_data.data[qubit]["signal"],
                phase=resonator_spectroscopy_data.data[qubit]["phase"],
                freq=resonator_spectroscopy_data.data[qubit]["freq"],
                twpa_freq=_freq,
            )

    return data


def _fit(data: ResonatorTWPAFrequencyData) -> ResonatorTWPAFrequencyResults:
    """Post-processing function for ResonatorTWPASpectroscopy."""
    qubits = data.qubits
    bare_frequency = {}
    frequency = {}
    twpa_frequency = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        if data.resonator_type == "3D":
            print("3D")
            index_best_freq = np.argmax(data_qubit["signal"])
            twpa_frequency[qubit] = data_qubit["twpa_freq"][index_best_freq]
        else:
            print("2D")
            index_best_freq = np.argmin(data_qubit["signal"])
            twpa_frequency[qubit] = data_qubit["twpa_freq"][index_best_freq]

        if data.power_level is PowerLevel.high:
            print("high")
            bare_frequency[qubit] = data_qubit["freq"][index_best_freq]
        else:
            print("low")
            frequency[qubit] = data_qubit["freq"][index_best_freq]

    if data.power_level is PowerLevel.high:
        return ResonatorTWPAFrequencyResults(
            twpa_frequency=twpa_frequency,
            bare_frequency=bare_frequency,
        )
    else:
        return ResonatorTWPAFrequencyResults(
            twpa_frequency=twpa_frequency,
            frequency=frequency,
        )


def _plot(data: ResonatorTWPAFrequencyData, fit: ResonatorTWPAFrequencyResults, qubit):
    """Plotting for ResonatorTWPAFrequency."""

    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "MSR",
            "phase (rad)",
        ),
    )

    qubit_data = data[qubit]
    resonator_frequencies = qubit_data.freq * HZ_TO_GHZ
    twpa_frequencies = qubit_data.twpa_freq

    fig.add_trace(
        go.Heatmap(
            x=resonator_frequencies,
            y=twpa_frequencies,
            z=qubit_data.signal * V_TO_UV,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text=f"{qubit}: Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="TWPA Frequency", row=1, col=1)
    fig.add_trace(
        go.Heatmap(
            x=resonator_frequencies,
            y=twpa_frequencies,
            z=qubit_data.phase,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text=f"{qubit}/: Frequency (Hz)", row=1, col=2)
    fig.update_yaxes(title_text="TWPA Frequency", row=1, col=2)

    if qubit in fit.bare_frequency:
        summary = table_dict(
            qubit,
            [
                "High Power Resonator Frequency [Hz]",
                "TWPA Frequency [Hz]",
            ],
            [
                np.round(fit.bare_frequency[qubit]),
                np.round(fit.twpa_frequency[qubit]),
            ],
        )
    else:
        summary = table_dict(
            qubit,
            [
                "Low Power Resonator Frequency [Hz]",
                "TWPA Frequency [Hz]",
            ],
            [
                np.round(fit.frequency[qubit]),
                np.round(fit.twpa_frequency[qubit]),
            ],
        )

    fitting_report = table_html(summary)

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


twpa_frequency_snr = Routine(_acquisition, _fit, _plot)
"""Resonator TWPA Frequency Routine object."""
