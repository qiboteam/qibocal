"""CZ virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import Pulse, PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.characterization.two_qubit_interaction.chevron import order_pair

# from .utils import order_pair


@dataclass
class CZVirtualZSweepParameters(Parameters):
    """CzVirtualZ runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    flux_pulse_amplitude_min: float
    flux_pulse_amplitude_max: float
    flux_pulse_amplitude_step: float
    duration_min: int
    duration_max: int
    duration_step: int
    """Amplitude of flux pulse implementing CZ."""
    flux_pulse_duration: Optional[float] = None
    """Duration of flux pulse implementing CZ."""
    dt: Optional[float] = 20
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class CZVirtualZSweepResults(Results):
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
        #     (qubit, control) for qubit, control, _ in self.fitted_parameters
        # ]


ChevronType = np.dtype(
    [
        ("amp", np.float64),
        ("theta", np.float64),
        ("duration", np.float64),
        ("prob_target", np.float64),
        ("prob_control", np.float64),
    ]
)


@dataclass
class CZVirtualZSweepData(Data):
    """CZVirtualZ data."""

    data: dict[tuple, npt.NDArray[ChevronType]] = field(default_factory=dict)
    thetas: list = field(default_factory=list)
    vphases: dict[QubitPairId, dict[QubitId, float]] = field(default_factory=dict)
    amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)
    durations: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)

    def __getitem__(self, pair):
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    def register_qubit(
        self, qubit, control, setup, theta, amp, duration, prob_control, prob_target
    ):
        """Store output for single qubit."""
        size = len(theta) * len(amp) * len(duration)
        duration, amplitude, angle = np.meshgrid(duration, amp, theta)
        ar = np.empty(size, dtype=ChevronType)
        ar["theta"] = angle.ravel()
        ar["amp"] = amplitude.ravel()
        ar["duration"] = duration.ravel()
        ar["prob_control"] = prob_control.ravel()
        ar["prob_target"] = prob_target.ravel()
        self.data[qubit, control, setup] = np.rec.array(ar)


def create_sequence(
    platform: Platform,
    setup: str,
    target_qubit: QubitId,
    control_qubit: QubitId,
    ordered_pair: list[QubitId, QubitId],
    parking: bool,
    dt: float,
    amplitude: float = None,
    duration: float = None,
) -> tuple[
    PulseSequence,
    dict[QubitId, Pulse],
    dict[QubitId, Pulse],
    dict[QubitId, Pulse],
    dict[QubitId, Pulse],
]:
    """Create the experiment PulseSequence."""

    sequence = PulseSequence()

    Y90_pulse = platform.create_RX90_pulse(
        target_qubit, start=0, relative_phase=np.pi / 2
    )
    RX_pulse_start = platform.create_RX_pulse(control_qubit, start=0, relative_phase=0)

    cz, virtual_z_phase = platform.create_CZ_pulse_sequence(
        (ordered_pair[1], ordered_pair[0]),
        start=max(Y90_pulse.finish, RX_pulse_start.finish),
    )

    if amplitude is not None:
        cz.get_qubit_pulses(ordered_pair[1])[0].amplitude = amplitude

    if duration is not None:
        cz.get_qubit_pulses(ordered_pair[1])[0].duration = duration

    theta_pulse = platform.create_RX90_pulse(
        target_qubit,
        start=cz.finish + dt,
        relative_phase=virtual_z_phase[target_qubit],
    )
    RX_pulse_end = platform.create_RX_pulse(
        control_qubit,
        start=cz.finish + dt,
        relative_phase=virtual_z_phase[control_qubit],
    )
    measure_target = platform.create_qubit_readout_pulse(
        target_qubit, start=theta_pulse.finish
    )
    measure_control = platform.create_qubit_readout_pulse(
        control_qubit, start=theta_pulse.finish
    )

    sequence.add(
        Y90_pulse,
        cz.get_qubit_pulses(ordered_pair[1]),
        theta_pulse,
        measure_target,
        measure_control,
    )

    if setup == "X":
        sequence.add(
            RX_pulse_start,
            RX_pulse_end,
        )

    if parking:
        for pulse in cz:
            if pulse.qubit not in ordered_pair:
                pulse.duration = theta_pulse.finish
                sequence.add(pulse)

    return (
        sequence,
        virtual_z_phase,
        theta_pulse,
        cz.get_qubit_pulses(ordered_pair[1])[0].amplitude,
        cz.get_qubit_pulses(ordered_pair[1])[0].duration,
    )


def _acquisition(
    params: CZVirtualZSweepParameters,
    platform: Platform,
    qubits: list[QubitPairId],
) -> CZVirtualZSweepData:
    r"""
    Acquisition for CZVirtualZ.

    Check the two-qubit landscape created by a flux pulse of a given duration
    and amplitude.
    The system is initialized with a Y90 pulse on the low frequency qubit and either
    an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to
    the high frequency qubit in order to perform a two-qubit interaction. The Id/X gate
    is undone in the high frequency qubit and a theta90 pulse is applied to the low
    frequency qubit before measurement. That is, a pi-half pulse around the relative phase
    parametereized by the angle theta.
    Measurements on the low frequency qubit yield the 2Q-phase of the gate and the
    remnant single qubit Z phase aquired during the execution to be corrected.
    Population of the high frequency qubit yield the leakage to the non-computational states
    during the execution of the flux pulse.
    """

    theta_absolute = np.arange(params.theta_start, params.theta_end, params.theta_step)
    data = CZVirtualZSweepData(thetas=theta_absolute.tolist())
    for pair in qubits:
        # order the qubits so that the low frequency one is the first
        ord_pair = order_pair(pair, platform.qubits)

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
                    sweeper_theta,
                    sweeper_duration,
                    sweeper_amplitude,
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


def fit_function(x, p0, p1, p2):
    """Sinusoidal fit function."""
    # return p0 + p1 * np.sin(2*np.pi*p2 * x + p3)
    return np.sin(x + p2) * p0 + p1


def _fit(
    data: CZVirtualZSweepData,
) -> CZVirtualZSweepResults:
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
                for qubit, control, setup in data[pair]:
                    target_data = data[pair][qubit, control, setup].prob_target[
                        np.where(
                            np.logical_and(
                                data[pair][qubit, control, setup].amp == amplitude,
                                data[pair][qubit, control, setup].duration == duration,
                            )
                        )
                    ]
                    pguess = [
                        np.max(target_data) - np.min(target_data),
                        np.mean(target_data),
                        np.pi,
                    ]
                    # try:
                    popt, _ = curve_fit(
                        fit_function,
                        np.array(data.thetas) - data.vphases[ord_pair][qubit],
                        target_data,
                        p0=pguess,
                        bounds=(
                            (0, -np.max(target_data), 0),
                            (np.max(target_data), np.max(target_data), 2 * np.pi),
                        ),
                    )
                    fitted_parameters[
                        qubit, control, setup, amplitude, duration
                    ] = popt.tolist()
                # except Exception as e:
                #     log.warning(f"CZ fit failed for pair ({qubit, control}) due to {e}.")

                # try:
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
                                        data[pair][target_q, control_q, "X"].duration
                                        == duration,
                                    )
                                )
                            ]
                            - data[pair][target_q, control_q, "I"].prob_control[
                                np.where(
                                    np.logical_and(
                                        data[pair][target_q, control_q, "I"].amp
                                        == amplitude,
                                        data[pair][target_q, control_q, "I"].duration
                                        == duration,
                                    )
                                )
                            ]
                        )
                    )
            # except KeyError:
            #     pass  # exception covered above
    return CZVirtualZSweepResults(
        cz_angles=cz_angles,
        virtual_phases=virtual_phases,
        fitted_parameters=fitted_parameters,
        leakages=leakages,
    )


# TODO: remove str
def _plot(data: CZVirtualZSweepData, fit: CZVirtualZSweepResults, qubit: QubitPairId):
    """Plot routine for CZVirtualZ."""
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
    # fig2 = make_subplots(
    #     rows=1,
    #     cols=2,
    #     subplot_titles=(
    #         f"Qubit {qubits[0]}",
    #         f"Qubit {qubits[1]}",
    #     ),
    # )
    for target_q, control_q in (
        qubit,
        list(qubit)[::-1],
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

        condition = (target_q, control_q) == qubit
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
                # zmin=0,
                # zmax=0.05,
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


def _update(results: CZVirtualZSweepResults, platform: Platform, qubit: QubitPairId):
    # FIXME: quick fix for qubit order
    # qubit_pair = tuple(sorted(qubit))
    # qubit = tuple(sorted(qubit))
    # update.virtual_phases(results.virtual_phase[qubit], platform, qubit)
    pass


cz_sweep = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""CZ virtual Z correction routine."""
