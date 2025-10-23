"""virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, Sweeper

from qibocal import update
from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import table_dict, table_html

from .utils import fit_virtualz, order_pair
from .virtual_z_phases import create_sequence

__all__ = ["optimize_two_qubit_gate"]


@dataclass
class OptimizeTwoQubitGateParameters(Parameters):
    """OptimizeTwoQubitGate runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    flux_pulse_amplitude_min: float
    """Minimum amplitude of flux pulse swept."""
    flux_pulse_amplitude_max: float
    """Maximum amplitude of flux pulse swept."""
    flux_pulse_amplitude_step: float
    """Step amplitude of flux pulse swept."""
    duration_min: int
    """Minimum duration of flux pulse swept."""
    duration_max: int
    """Maximum duration of flux pulse swept."""
    duration_step: int
    """Step duration of flux pulse swept."""
    dt: float = 0
    """Time delay between flux pulses and readout."""
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """

    @property
    def theta_range(self) -> np.ndarray:
        return np.arange(self.theta_start, self.theta_end, self.theta_step)

    @property
    def amplitude_range(self) -> np.ndarray:
        return np.arange(
            self.flux_pulse_amplitude_min,
            self.flux_pulse_amplitude_max,
            self.flux_pulse_amplitude_step,
        )

    @property
    def duration_range(self) -> np.ndarray:
        return np.arange(self.duration_min, self.duration_max, self.duration_step)


@dataclass
class OptimizeTwoQubitGateResults(Results):
    """CzVirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId, float], list]
    """Fitted parameters"""
    native: str
    """Native two qubit gate."""
    angles: dict[tuple[QubitPairId, float], float]
    """Two qubit gate angle."""
    virtual_phases: dict[tuple[QubitPairId, float], dict[QubitId, float]]
    """Virtual Z phase correction."""
    leakages: dict[tuple[QubitPairId, float], dict[QubitId, float]]
    """Leakage on control qubit for pair."""
    best_angles: dict[QubitPairId, list[float]]
    best_leakages: dict[QubitPairId, list[float]]
    best_angles_per_duration: dict[QubitPairId, list[float]]
    best_leakages_per_duration: dict[QubitPairId, list[float]]
    best_amp: dict[QubitPairId]
    """Flux pulse amplitude of best configuration."""
    best_dur: dict[QubitPairId]
    """Flux pulse duration of best configuration."""
    best_virtual_phase: dict[QubitPairId]
    """Virtual phase to correct best configuration."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.

        Additional  manipulations required because of the Results class.
        """
        return True
        # pairs = {
        #     (target, control) for target, control, _, _, _ in self.fitted_parameters
        # }
        # return key in pairs


@dataclass
class OptimizeTwoQubitGateData(Data):
    """OptimizeTwoQubitGate data."""

    data: dict[tuple, npt.NDArray] = field(default_factory=dict)
    """Raw data."""
    _sorted_pairs: list[QubitPairId] = field(default_factory=dict)
    thetas: list = field(default_factory=list)
    """Angles swept."""
    native: str = "CZ"
    """Native two qubit gate."""
    amplitudes: list = field(default_factory=list)
    """"Amplitudes swept."""
    durations: list = field(default_factory=list)
    """Durations swept."""

    @property
    def sorted_pairs(self):
        return [
            pair if isinstance(pair, tuple) else tuple(pair)
            for pair in self._sorted_pairs
        ]

    @sorted_pairs.setter
    def sorted_pairs(self, value):
        self._sorted_pairs = value

    def parse(self, i, j):
        return {
            key: value[
                :,
                i,
                j,
            ]
            for key, value in self.data.items()
        }


def _acquisition(
    params: OptimizeTwoQubitGateParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> OptimizeTwoQubitGateData:
    r"""
    Repetition of correct virtual phase experiment for several amplitude and duration values.
    """

    data = OptimizeTwoQubitGateData(
        _sorted_pairs=[order_pair(pair, platform) for pair in targets],
        thetas=params.theta_range.tolist(),
        amplitudes=params.amplitude_range.tolist(),
        durations=params.duration_range.tolist(),
        native=params.native,
    )
    for ordered_pair in data.sorted_pairs:
        for target_q, control_q in (
            (ordered_pair[0], ordered_pair[1]),
            (ordered_pair[1], ordered_pair[0]),
        ):
            for setup in ("I", "X"):
                (
                    sequence,
                    flux_pulse,
                    vz_pulses,
                ) = create_sequence(
                    platform,
                    setup,
                    target_q,
                    control_q,
                    ordered_pair,
                    params.native,
                    params.dt,
                    flux_pulse_max_duration=params.duration_max,
                )

                sweeper_theta = Sweeper(
                    parameter=Parameter.phase,
                    values=-params.theta_range,
                    pulses=vz_pulses,
                )

                sweeper_amplitude = Sweeper(
                    parameter=Parameter.amplitude,
                    values=params.amplitude_range,
                    pulses=[flux_pulse],
                )

                sweeper_duration = Sweeper(
                    parameter=Parameter.duration,
                    values=params.duration_range,
                    pulses=[flux_pulse],
                )

                ro_target = list(
                    sequence.channel(platform.qubits[target_q].acquisition)
                )[-1]
                ro_control = list(
                    sequence.channel(platform.qubits[control_q].acquisition)
                )[-1]
                results = platform.execute(
                    [sequence],
                    [[sweeper_duration], [sweeper_amplitude], [sweeper_theta]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                )
                data.data[target_q, control_q, setup] = np.stack(
                    [results[ro_target.id], results[ro_control.id]]
                )
    return data


def _fit(
    data: OptimizeTwoQubitGateData,
) -> OptimizeTwoQubitGateResults:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    virtual_phases = {}
    angles = {}
    best_angles = {}
    best_angles_per_duration = {}
    best_leakages = {}
    best_leakages_per_duration = {}
    leakages = {}
    best_amp = {}
    best_dur = {}
    best_virtual_phase = {}
    for pair in data.sorted_pairs:
        for target_q, control_q in (
            (pair[0], pair[1]),
            (pair[1], pair[0]),
        ):
            _pair = (target_q, control_q)
            angles[_pair], leakages[_pair], virtual_phases[_pair] = [], [], []
            best_angles[_pair], best_leakages[_pair] = [], []
            best_angles_per_duration[_pair], best_leakages_per_duration[_pair] = [], []
            (
                fitted_parameters[_pair[0], _pair[1], "I"],
                fitted_parameters[_pair[0], _pair[1], "X"],
            ) = [], []
            for i in range(len(data.durations)):
                temp_angles = []
                temp_leakages = []
                for j in range(len(data.amplitudes)):
                    new_fitted_parameter, new_phases, new_angle, new_leak = (
                        fit_virtualz(data.parse(i, j), _pair, data.thetas, 1)
                    )
                    temp_angles.append(new_angle[_pair])
                    temp_leakages.append(new_leak[_pair])
                    angles[_pair].append(new_angle[_pair])
                    leakages[_pair].append(new_leak[_pair])
                    virtual_phases[_pair].append(new_phases[_pair])
                    for setup in ["I", "X"]:
                        fitted_parameters[_pair[0], _pair[1], setup].append(
                            new_fitted_parameter[_pair, setup]
                        )
                best_angles_per_duration[_pair].append(
                    int(np.argmin(np.abs(np.pi - np.array(temp_angles))))
                )
                best_leakages_per_duration[_pair].append(
                    np.where(np.array(temp_leakages) < 0.015)[0].tolist()
                )
            duration_index = max(
                range(len(best_leakages_per_duration[_pair])),
                key=lambda i: len(best_leakages_per_duration[_pair][i]),
            )
            amplitude_index = best_angles_per_duration[_pair][duration_index]

            best_virtual_phase[_pair] = virtual_phases[_pair][
                int(len(data.amplitudes) * duration_index + amplitude_index)
            ]
            best_dur[_pair] = data.durations[duration_index]
            best_amp[_pair] = data.amplitudes[amplitude_index]

    return OptimizeTwoQubitGateResults(
        angles=angles,
        native=data.native,
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        best_angles=best_angles,
        best_angles_per_duration=best_angles_per_duration,
        best_leakages=best_leakages,
        best_leakages_per_duration=best_leakages_per_duration,
        leakages=leakages,
        best_amp=best_amp,
        best_dur=best_dur,
        best_virtual_phase=best_virtual_phase,
    )


def _plot(
    data: OptimizeTwoQubitGateData,
    fit: OptimizeTwoQubitGateResults,
    target: QubitPairId,
):
    """Plot routine for OptimizeTwoQubitGate."""
    if target not in data.sorted_pairs:
        target = (target[1], target[0])
    fitting_report = ""

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Qubit {target[0]} {data.native} angle",
            f"Qubit {target[1]} Leakage",
            f"Qubit {target[1]} {data.native} angle",
            f"Qubit {target[0]} Leakage",
        ),
    )
    if fit is not None:
        for target_q, control_q in (
            target,
            list(target)[::-1],
        ):
            condition = [target_q, control_q] == list(target)
            fig.add_trace(
                go.Heatmap(
                    x=data.durations,
                    y=data.amplitudes,
                    z=np.array(fit.angles[target])
                    .reshape(len(data.durations), len(data.amplitudes))
                    .T,
                    zmin=0,
                    zmax=2 * np.pi,
                    name="{fit.native} angle",
                    colorbar_x=-0.1,
                    colorscale="Twilight",
                    showscale=condition,
                ),
                row=1 if condition else 2,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=data.durations,
                    y=[
                        data.amplitudes[fit.best_angles_per_duration[target][i]]
                        for i in range(len(data.durations))
                    ],
                    line=dict(color="black"),
                    showlegend=condition,
                    name="Best CZ angle",
                    mode="markers",
                    legendgroup="Best CZ angle",
                ),
                row=1 if condition else 2,
                col=1,
            )

            fig.add_trace(
                go.Heatmap(
                    x=data.durations,
                    y=data.amplitudes,
                    z=np.array(fit.leakages[target])
                    .reshape(len(data.durations), len(data.amplitudes))
                    .T,
                    zmin=0,
                    zmax=0.25,
                    name="{fit.native} angle",
                    colorscale="Inferno",
                    showscale=condition,
                ),
                row=1 if condition else 2,
                col=2,
            )

            fig.add_trace(
                go.Scatter(
                    x=np.array(
                        [
                            data.durations[i]
                            for i in range(len(data.durations))
                            for _ in range(
                                len(fit.best_leakages_per_duration[target][i])
                            )
                            if len(fit.best_leakages_per_duration[target][i]) > 0
                        ]
                    ),
                    y=np.array(
                        [
                            data.amplitudes[idx]
                            for i in range(len(data.durations))
                            for idx in fit.best_leakages_per_duration[target][i]
                        ]
                    ),
                    line=dict(color="gray"),
                    showlegend=condition,
                    name="Minimum leakage",
                    mode="markers",
                    legendgroup="Minimum leakage",
                ),
                row=1 if condition else 2,
                col=2,
            )

            for j in [1, 2]:
                fig.add_trace(
                    go.Scatter(
                        x=[fit.best_dur[target_q, control_q]],
                        y=[fit.best_amp[target_q, control_q]],
                        line=dict(color="yellow"),
                        name="Best CZ",
                        mode="markers",
                        showlegend=condition and j == 1,
                        legendgroup="Best CZ",
                    ),
                    row=1 if condition else 2,
                    col=j,
                )

            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h"),
            )

            fitting_report = table_html(
                table_dict(
                    [target_q, target_q],
                    [
                        "Flux pulse amplitude [a.u.]",
                        "Flux pulse duration [ns]",
                    ],
                    [
                        np.round(fit.best_amp[target_q, control_q], 4),
                        np.round(fit.best_dur[target_q, control_q], 4),
                    ],
                )
            )

        fig.update_layout(
            xaxis3_title="Pulse duration [ns]",
            xaxis4_title="Pulse duration [ns]",
            yaxis1_title="Flux Amplitude [a.u.]",
            yaxis3_title="Flux Amplitude [a.u.]",
        )

    return [fig], fitting_report


def _update(
    results: OptimizeTwoQubitGateResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    # FIXME: quick fix for qubit order
    target = tuple(sorted(target))
    update.virtual_phases(results.best_virtual_phase, results.native, platform, target)
    getattr(update, f"{results.native}_duration")(
        (results.best_dur[target] + results.best_dur[target[1], target[0]]) / 2,
        platform,
        target,
    )
    getattr(update, f"{results.native}_amplitude")(
        (results.best_amp[target] + results.best_amp[target[1], target[0]]) / 2,
        platform,
        target,
    )


optimize_two_qubit_gate = Routine(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""Optimize two qubit gate protocol"""
