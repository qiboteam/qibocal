"""virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from .utils import order_pair
from .virtual_z_phases import create_sequence, fit_function


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
    dt: Optional[float] = 20
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """


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
        # TODO: to be improved
        pairs = {
            (target, control) for target, control, _, _, _ in self.fitted_parameters
        }
        return tuple(key) in list(pairs)


OptimizeTwoQubitGateType = np.dtype(
    [
        ("amp", np.float64),
        ("theta", np.float64),
        ("duration", np.float64),
        ("prob_target", np.float64),
        ("prob_control", np.float64),
    ]
)


@dataclass
class OptimizeTwoQubitGateData(Data):
    """OptimizeTwoQubitGate data."""

    data: dict[tuple, npt.NDArray[OptimizeTwoQubitGateType]] = field(
        default_factory=dict
    )
    """Raw data."""
    thetas: list = field(default_factory=list)
    """Angles swept."""
    native: str = "CZ"
    """Native two qubit gate."""
    amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """"Amplitudes swept."""
    durations: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    """Durations swept."""

    def __getitem__(self, pair):
        """Extract data for pair."""
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    def register_qubit(
        self, target, control, setup, theta, amp, duration, prob_control, prob_target
    ):
        """Store output for single pair."""
        size = len(theta) * len(amp) * len(duration)
        duration, amplitude, angle = np.meshgrid(duration, amp, theta, indexing="ij")
        ar = np.empty(size, dtype=OptimizeTwoQubitGateType)
        ar["theta"] = angle.ravel()
        ar["amp"] = amplitude.ravel()
        ar["duration"] = duration.ravel()
        ar["prob_control"] = prob_control.ravel()
        ar["prob_target"] = prob_target.ravel()
        self.data[target, control, setup] = np.rec.array(ar)


def _acquisition(
    params: OptimizeTwoQubitGateParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> OptimizeTwoQubitGateData:
    r"""
    Repetition of correct virtual phase experiment for several amplitude and duration values.
    """

    theta_absolute = np.arange(params.theta_start, params.theta_end, params.theta_step)
    data = OptimizeTwoQubitGateData(
        thetas=theta_absolute.tolist(), native=params.native
    )
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ord_pair = order_pair(pair, platform)

        for target_q, control_q in (
            (ord_pair[0], ord_pair[1]),
            (ord_pair[1], ord_pair[0]),
        ):
            for setup in ("I", "X"):
                (
                    sequence,
                    theta_pulse,
                    amplitude,
                    data.durations[ord_pair],
                ) = create_sequence(
                    platform,
                    setup,
                    target_q,
                    control_q,
                    ord_pair,
                    params.native,
                    params.dt,
                    params.parking,
                    params.flux_pulse_amplitude_min,
                )
                theta = np.arange(
                    params.theta_start,
                    params.theta_end,
                    params.theta_step,
                    dtype=float,
                )

                amplitude_range = np.arange(
                    params.flux_pulse_amplitude_min,
                    params.flux_pulse_amplitude_max,
                    params.flux_pulse_amplitude_step,
                    dtype=float,
                )

                duration_range = np.arange(
                    params.duration_min,
                    params.duration_max,
                    params.duration_step,
                    dtype=float,
                )

                data.amplitudes[ord_pair] = amplitude_range.tolist()
                data.durations[ord_pair] = duration_range.tolist()

                sweeper_theta = Sweeper(
                    Parameter.relative_phase,
                    theta,
                    pulses=[theta_pulse],
                    type=SweeperType.ABSOLUTE,
                )

                sweeper_amplitude = Sweeper(
                    Parameter.amplitude,
                    amplitude_range / amplitude,
                    pulses=[sequence.qf_pulses[0]],
                    type=SweeperType.FACTOR,
                )

                sweeper_duration = Sweeper(
                    Parameter.duration,
                    duration_range,
                    pulses=[sequence.qf_pulses[0]],
                    type=SweeperType.ABSOLUTE,
                )

                results = platform.sweep(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        relaxation_time=params.relaxation_time,
                        acquisition_type=AcquisitionType.DISCRIMINATION,
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                    sweeper_duration,
                    sweeper_amplitude,
                    sweeper_theta,
                )

                result_target = results[target_q].probability(1)
                result_control = results[control_q].probability(1)
                data.register_qubit(
                    target_q,
                    control_q,
                    setup,
                    theta,
                    data.amplitudes[ord_pair],
                    data.durations[ord_pair],
                    result_control,
                    result_target,
                )
    return data


def _fit(
    data: OptimizeTwoQubitGateData,
) -> OptimizeTwoQubitGateResults:
    """Repetition of correct virtual phase fit for all configurations."""
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phases = {}
    angles = {}
    leakages = {}
    best_amp = {}
    best_dur = {}
    best_virtual_phase = {}
    # FIXME: experiment should be for single pair
    for pair in pairs:
        # TODO: improve this
        ord_pair = next(iter(data.amplitudes))[:2]
        for duration in data.durations[ord_pair]:
            for amplitude in data.amplitudes[ord_pair]:
                virtual_phases[ord_pair[0], ord_pair[1], amplitude, duration] = {}
                leakages[ord_pair[0], ord_pair[1], amplitude, duration] = {}
                for target, control, setup in data[pair]:
                    target_data = data[pair][target, control, setup].prob_target[
                        np.where(
                            np.logical_and(
                                data[pair][target, control, setup].amp == amplitude,
                                data[pair][target, control, setup].duration == duration,
                            )
                        )
                    ]
                    pguess = [
                        np.max(target_data) - np.min(target_data),
                        np.mean(target_data),
                        np.pi,
                    ]

                    try:
                        popt, _ = curve_fit(
                            fit_function,
                            np.array(data.thetas),
                            target_data,
                            p0=pguess,
                            bounds=(
                                (0, -np.max(target_data), 0),
                                (np.max(target_data), np.max(target_data), 2 * np.pi),
                            ),
                        )

                        fitted_parameters[
                            target, control, setup, amplitude, duration
                        ] = popt.tolist()

                    except Exception as e:
                        log.warning(
                            f"Fit failed for pair ({target, control}) due to {e}."
                        )

                try:
                    for target_q, control_q in (
                        pair,
                        list(pair)[::-1],
                    ):
                        angles[target_q, control_q, amplitude, duration] = abs(
                            fitted_parameters[
                                target_q, control_q, "X", amplitude, duration
                            ][2]
                            - fitted_parameters[
                                target_q, control_q, "I", amplitude, duration
                            ][2]
                        )
                        virtual_phases[ord_pair[0], ord_pair[1], amplitude, duration][
                            target_q
                        ] = fitted_parameters[
                            target_q, control_q, "I", amplitude, duration
                        ][
                            2
                        ]

                        # leakage estimate: L = m /2
                        # See NZ paper from Di Carlo
                        # approximation which does not need qutrits
                        # https://arxiv.org/pdf/1903.02492.pdf
                        leakages[ord_pair[0], ord_pair[1], amplitude, duration][
                            control_q
                        ] = 0.5 * float(
                            np.mean(
                                data[pair][target_q, control_q, "X"].prob_control[
                                    np.where(
                                        np.logical_and(
                                            data[pair][target_q, control_q, "X"].amp
                                            == amplitude,
                                            data[pair][
                                                target_q, control_q, "X"
                                            ].duration
                                            == duration,
                                        )
                                    )
                                ]
                                - data[pair][target_q, control_q, "I"].prob_control[
                                    np.where(
                                        np.logical_and(
                                            data[pair][target_q, control_q, "I"].amp
                                            == amplitude,
                                            data[pair][
                                                target_q, control_q, "I"
                                            ].duration
                                            == duration,
                                        )
                                    )
                                ]
                            )
                        )
                except KeyError:
                    pass
        index = np.argmin(np.abs(np.array(list(angles.values())) - np.pi))
        _, _, amp, dur = np.array(list(angles))[index]
        best_amp[pair] = float(amp)
        best_dur[pair] = float(dur)
        best_virtual_phase[pair] = virtual_phases[
            ord_pair[0], ord_pair[1], float(amp), float(dur)
        ]

    return OptimizeTwoQubitGateResults(
        angles=angles,
        native=data.native,
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
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
    fitting_report = ""
    qubits = next(iter(data.amplitudes))[:2]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Qubit {qubits[0]} {data.native} angle",
            f"Qubit {qubits[0]} Leakage",
            f"Qubit {qubits[1]} {data.native} angle",
            f"Qubit {qubits[1]} Leakage",
        ),
    )
    if fit is not None:
        for target_q, control_q in (
            target,
            list(target)[::-1],
        ):
            cz = []
            durs = []
            amps = []
            leakage = []
            for i in data.amplitudes[qubits]:
                for j in data.durations[qubits]:
                    durs.append(j)
                    amps.append(i)
                    cz.append(fit.angles[target_q, control_q, i, j])
                    leakage.append(fit.leakages[qubits[0], qubits[1], i, j][control_q])

            condition = [target_q, control_q] == list(target)

            fig.add_trace(
                go.Heatmap(
                    x=durs,
                    y=amps,
                    z=cz,
                    zmin=np.pi / 2,
                    zmax=3 * np.pi / 2,
                    name="{fit.native} angle",
                    colorbar_x=-0.1,
                    colorscale="RdBu",
                    showscale=condition,
                ),
                row=1 if condition else 2,
                col=1,
            )

            fig.add_trace(
                go.Heatmap(
                    x=durs,
                    y=amps,
                    z=leakage,
                    name="Leakage",
                    showscale=condition,
                    colorscale="Reds",
                    zmin=0,
                    zmax=0.05,
                ),
                row=1 if condition else 2,
                col=2,
            )
            fitting_report = table_html(
                table_dict(
                    [qubits[1], qubits[1]],
                    [
                        "Flux pulse amplitude [a.u.]",
                        "Flux pulse duration [ns]",
                    ],
                    [
                        np.round(fit.best_amp[qubits], 4),
                        np.round(fit.best_dur[qubits], 4),
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
    results: OptimizeTwoQubitGateResults, platform: Platform, target: QubitPairId
):
    # FIXME: quick fix for qubit order
    target = tuple(sorted(target))
    update.virtual_phases(
        results.best_virtual_phase[target], results.native, platform, target
    )
    getattr(update, f"{results.native}_duration")(
        results.best_dur[target], platform, target
    )
    getattr(update, f"{results.native}_amplitude")(
        results.best_amp[target], platform, target
    )


optimize_two_qubit_gate = Routine(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""Optimize two qubit gate protocol"""
