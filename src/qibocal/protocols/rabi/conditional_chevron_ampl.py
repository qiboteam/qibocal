"""Rabi experiment that sweeps amplitude and frequency when toggling a spectator qubit."""

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    PulseSequence,
    Sweeper,
)

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform

from ..utils import HZ_TO_GHZ
from .utils import sequence_amplitude

__all__ = ["conditional_rabi_chevron_ampl"]


@dataclass
class ConditionalRabiChevronAmplParameters(Parameters):
    """ConditionalRabiAmplChevron runcard inputs."""

    amplitude_start: float
    """Initial pi pulse amplitude [a.u.]."""
    amplitude_end: float
    """Final pi amplitude amplitude [a.u.]."""
    amplitude_step: float
    """Step pi pulse amplitude [a.u.]."""
    min_freq: int
    """Minimum frequency as an offset."""
    max_freq: int
    """Maximum frequency as an offset."""
    step_freq: int
    """Frequency to use as step for the scan."""
    activate_spectators: bool = True
    """Flag for setting spectator qbuit in state 1."""
    pulse_length: Optional[float] = None
    """Pi pulse duration. Same for all qubits."""

    @property
    def frequency_range(self):
        return np.arange(
            self.min_freq,
            self.max_freq,
            self.step_freq,
        )

    @property
    def amplitude_range(self):
        return np.arange(
            self.amplitude_start,
            self.amplitude_end,
            self.amplitude_step,
        )


@dataclass
class ConditionalRabiChevronAmplResults(Results):
    """ConditionalRabiAmplChevron outputs."""

    activate_spectators: bool
    """Flag for setting spectator qbuit in state 1."""
    amplitude: dict[QubitPairId, Union[float, list[float]]]
    """Pi pulse duration for each qubit."""
    fitted_parameters: dict[QubitPairId, dict[str, float]]
    """Raw fitting output."""


CondRabiAmplChevronType = np.dtype(
    [
        ("ampl", np.float64),
        ("freq", np.float64),
        ("target", np.float64),
        ("spectator", np.float64),
    ]
)
"""Custom dtype for rabi amplitude with spectator."""


@dataclass
class ConditionalRabiChevronAmplData(Data):
    """ConditionalRabiAmplChevron data acquisition."""

    activate_spectators: bool
    """Flag for setting spectator qbuit in state 1."""
    data: dict[QubitPairId, npt.NDArray[CondRabiAmplChevronType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, pair, freq, ampl, p_targ, p_spect):
        """Store output for single qubit."""
        size = len(freq) * len(ampl)
        amplitude, frequency = np.meshgrid(ampl, freq)
        data = np.empty(size, dtype=CondRabiAmplChevronType)
        data["freq"] = frequency.ravel()
        data["ampl"] = amplitude.ravel()
        data["target"] = p_targ.ravel()
        data["spectator"] = p_spect.ravel()
        self.data[pair] = np.rec.array(data)

    def amplitudes(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Unique qubit amplitudes."""
        return np.unique(self[qubit_pair].ampl)

    def frequencies(self, qubit_pair: QubitPairId) -> npt.NDArray:
        """Unique qubit frequency."""
        return np.unique(self[qubit_pair].freq)


def _acquisition(
    params: ConditionalRabiChevronAmplParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> ConditionalRabiChevronAmplData:
    """Data acquisition for ConditionalRabiChevron."""

    if any(isinstance(x, (str, int)) for x in targets):
        raise ValueError("At least one target is not a QubitPairId type.")

    target_qubits_list, spectator_qubits_list = map(list, zip(*targets))
    if any(s in target_qubits_list for s in spectator_qubits_list):
        raise ValueError("One or multiple qubits are set as both spectator and target.")

    complete_sequence, t_qd_pulses, t_ro_pulses, _ = sequence_amplitude(
        target_qubits_list, params, platform, False
    )

    spectators_drive_dict: dict[QubitId, PulseSequence] = {}
    spectators_ro_dict: dict[QubitId, PulseSequence] = {}
    for s in spectator_qubits_list:
        spectator_natives = platform.natives.single_qubit[s]
        spectators_drive_dict[s] = spectator_natives.RX()[0]
        spectators_ro_dict[s] = spectator_natives.MZ()[0]

    complete_sequence += PulseSequence(
        (spectators_ro_dict[s][0], Delay(duration=t_qd_pulses[t].duration))
        for t, s in targets
    )
    complete_sequence += PulseSequence(list(spectators_ro_dict.values()))

    if params.activate_spectators:
        complete_sequence = (
            PulseSequence(list(spectators_drive_dict.values())) | complete_sequence
        )

    ampl_sweeper = Sweeper(
        parameter=Parameter.amplitude,
        values=params.amplitude_range,
        pulses=[t_qd_pulses[q] for q in target_qubits_list],
    )

    freq_sweepers: dict[QubitId, Sweeper] = {}
    for t in target_qubits_list:
        target_drive_ch = platform.qubits[t].drive
        freq_sweepers[t] = Sweeper(
            parameter=Parameter.frequency,
            values=platform.config(target_drive_ch).frequency + params.frequency_range,
            channels=[target_drive_ch],
        )

    data = ConditionalRabiChevronAmplData(
        activate_spectators=params.activate_spectators
    )

    results = platform.execute(
        [complete_sequence],
        [list(freq_sweepers.values()), [ampl_sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for pair in targets:
        t, s = pair
        target_result = results[t_ro_pulses[t].id]
        spect_result = results[spectators_ro_dict[s][1].id]

        data.register_qubit(
            pair=pair,
            freq=freq_sweepers[t].values,
            ampl=ampl_sweeper.values,
            p_targ=target_result,
            p_spect=spect_result,
        )
    return data


def _fit(data: ConditionalRabiChevronAmplData) -> ConditionalRabiChevronAmplResults:
    """Do not perform any fitting procedure."""

    return ConditionalRabiChevronAmplResults(
        activate_spectators=data.activate_spectators,
        amplitude={},
        fitted_parameters={},
    )


def _plot(
    data: ConditionalRabiChevronAmplData,
    target: QubitPairId,
    fit: Optional[ConditionalRabiChevronAmplResults] = None,
):
    """Plotting function for ConditionalRabiChevron."""
    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            ("PI pulse applied -" if data.activate_spectators else "")
            + f" Prob Target {target[0]}",
            ("PI pulse applied -" if data.activate_spectators else "")
            + f" Prob Spectator {target[1]}",
        ),
    )
    qubit_data = data[target]
    frequencies = qubit_data.freq * HZ_TO_GHZ
    amplitudes = qubit_data.ampl
    targ_prob = data[target].target
    spect_prob = data[target].spectator

    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=targ_prob,
            colorbar_x=0.46,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Heatmap(
            x=amplitudes,
            y=frequencies,
            z=spect_prob,
            colorbar_x=1.01,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=1)
    fig.update_xaxes(title_text="Amplitude [a.u.]", row=1, col=2)
    fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=1)

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
    )
    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ConditionalRabiChevronAmplResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    return


conditional_rabi_chevron_ampl = Routine(_acquisition, _fit, _plot, _update)
"""Rabi length with frequency tuning."""
