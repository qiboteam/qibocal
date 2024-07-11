"""CZ virtual correction experiment for two qubit gates, tune landscape."""

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

from qibocal.auto.operation import Data, Parameters, Results, Routine

from .cz_virtualz import create_sequence, fit_function
from .utils import order_pair


@dataclass
class CZSweepParameters(Parameters):
    """CZSweep runcard inputs."""

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


@dataclass
class CZSweepResults(Results):
    """CzVirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId, float], list]
    """Fitted parameters"""
    cz_angles: dict[tuple[QubitPairId, float], float]
    """CZ angle."""
    virtual_phases: dict[tuple[QubitPairId, float], dict[QubitId, float]]
    """Virtual Z phase correction."""
    leakages: dict[tuple[QubitPairId, float], dict[QubitId, float]]
    """Leakage on control qubit for pair."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.
        While key is a QubitPairId both chsh and chsh_mitigated contain
        an additional key which represents the basis chosen.
        """
        return True
        # return key in [
        #     (target, control) for target, control, _ in self.fitted_parameters
        # ]


CZSweepType = np.dtype(
    [
        ("amp", np.float64),
        ("theta", np.float64),
        ("duration", np.float64),
        ("prob_target", np.float64),
        ("prob_control", np.float64),
    ]
)


@dataclass
class CZSweepData(Data):
    """CZSweep data."""

    data: dict[tuple, npt.NDArray[CZSweepType]] = field(default_factory=dict)
    """Raw data."""
    thetas: list = field(default_factory=list)
    """Angles swept."""
    vphases: dict[QubitPairId, dict[QubitId, float]] = field(default_factory=dict)
    """Virtual phases for each qubit."""
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
        ar = np.empty(size, dtype=CZSweepType)
        ar["theta"] = angle.ravel()
        ar["amp"] = amplitude.ravel()
        ar["duration"] = duration.ravel()
        ar["prob_control"] = prob_control.ravel()
        ar["prob_target"] = prob_target.ravel()
        self.data[target, control, setup] = np.rec.array(ar)


def _acquisition(
    params: CZSweepParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> CZSweepData:
    r"""
    Repetition of CZVirtualZ experiment for several amplitude and duration values.
    """

    theta_absolute = np.arange(params.theta_start, params.theta_end, params.theta_step)
    data = CZSweepData(thetas=theta_absolute.tolist())
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
                    virtual_z_phase,
                    theta_pulse,
                    amplitude,
                    data.durations[ord_pair],
                ) = create_sequence(
                    platform,
                    setup,
                    target_q,
                    control_q,
                    ord_pair,
                    params.dt,
                    params.parking,
                    params.flux_pulse_amplitude_min,
                )
                data.vphases[ord_pair] = dict(virtual_z_phase)
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
                    theta - data.vphases[ord_pair][target_q],
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
                    theta - data.vphases[ord_pair][target_q],
                    data.amplitudes[ord_pair],
                    data.durations[ord_pair],
                    result_control,
                    result_target,
                )
    return data


def _fit(
    data: CZSweepData,
) -> CZSweepResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phases = {}
    cz_angles = {}
    leakages = {}
    for pair in pairs:
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
                            np.array(data.thetas) - data.vphases[ord_pair][target],
                            target_data,
                            p0=pguess,
                            bounds=(
                                (0, -np.max(target_data), 0),
                                (np.max(target_data), np.max(target_data), 2 * np.pi),
                            ),
                        )
                        import matplotlib.pyplot as plt

                        plt.figure()
                        plt.scatter(
                            np.array(data.thetas) - data.vphases[ord_pair][target],
                            target_data,
                        )
                        plt.plot(
                            np.array(data.thetas) - data.vphases[ord_pair][target],
                            fit_function(
                                np.array(data.thetas) - data.vphases[ord_pair][target],
                                *popt,
                            ),
                        )
                        plt.savefig(
                            f"test_cz_{duration}_{amplitude}_t{target}c{control}.png"
                        )
                        plt.close()
                        fitted_parameters[
                            target, control, setup, amplitude, duration
                        ] = popt.tolist()

                    except Exception as e:
                        log.warning(
                            f"CZ fit failed for pair ({target, control}) due to {e}."
                        )

                try:
                    for target_q, control_q in (
                        pair,
                        list(pair)[::-1],
                    ):
                        cz_angles[target_q, control_q, amplitude, duration] = abs(
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
    return CZSweepResults(
        cz_angles=cz_angles,
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
    )


def _plot(data: CZSweepData, fit: CZSweepResults, target: QubitPairId):
    """Plot routine for CZSweep."""
    qubits = next(iter(data.amplitudes))[:2]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Qubit {qubits[0]} CZ angle",
            f"Qubit {qubits[0]} Leakage",
            f"Qubit {qubits[1]} CZ angle",
            f"Qubit {qubits[1]} Leakage",
        ),
    )
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
                cz.append(fit.cz_angles[target_q, control_q, i, j])
                leakage.append(fit.leakages[qubits[0], qubits[1], i, j][control_q])

        condition = (target_q, control_q) == target
        fig.add_trace(
            go.Heatmap(
                x=durs,
                y=amps,
                z=cz,
                zmin=np.pi / 2,
                zmax=3 * np.pi / 2,
                name="CZ angle",
                colorbar_x=-0.4,
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

        fig.update_layout(
            xaxis3_title="Pulse duration [ns]",
            xaxis4_title="Pulse duration [ns]",
            yaxis1_title="Flux Amplitude [a.u.]",
            yaxis3_title="Flux Amplitude [a.u.]",
        )

    return [fig], ""


def _update(results: CZSweepResults, platform: Platform, target: QubitPairId):
    # FIXME: quick fix for qubit order
    # qubit_pair = tuple(sorted(target))
    # target = tuple(sorted(target))
    # update.virtual_phases(results.virtual_phase[target], platform, target)
    pass


cz_sweep = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""CZ virtual Z correction routine."""
