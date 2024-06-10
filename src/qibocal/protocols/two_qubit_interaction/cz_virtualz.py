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

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from .utils import order_pair


@dataclass
class CZVirtualZParameters(Parameters):
    """CzVirtualZ runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    flux_pulse_amplitude: Optional[float] = None
    """Amplitude of flux pulse implementing CZ."""
    flux_pulse_duration: Optional[float] = None
    """Duration of flux pulse implementing CZ."""
    dt: Optional[float] = 20
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class CZVirtualZResults(Results):
    """CzVirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId],]
    """Fitted parameters"""
    cz_angle: dict[QubitPairId, float]
    """CZ angle."""
    virtual_phase: dict[QubitPairId, dict[QubitId, float]]
    """Virtual Z phase correction."""
    leakage: dict[QubitPairId, dict[QubitId, float]]
    """Leakage on control qubit for pair."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.
        While key is a QubitPairId both chsh and chsh_mitigated contain
        an additional key which represents the basis chosen.
        """

        return key in [
            (target, control) for target, control, _ in self.fitted_parameters
        ]


CZVirtualZType = np.dtype([("target", np.float64), ("control", np.float64)])


@dataclass
class CZVirtualZData(Data):
    """CZVirtualZ data."""

    data: dict[tuple, npt.NDArray[CZVirtualZType]] = field(default_factory=dict)
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
    params: CZVirtualZParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> CZVirtualZData:
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
    data = CZVirtualZData(thetas=theta_absolute.tolist())
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
                    data.amplitudes[ord_pair],
                    data.durations[ord_pair],
                ) = create_sequence(
                    platform,
                    setup,
                    target_q,
                    control_q,
                    ord_pair,
                    params.dt,
                    params.parking,
                    params.flux_pulse_amplitude,
                )
                data.vphases[ord_pair] = dict(virtual_z_phase)
                theta = np.arange(
                    params.theta_start,
                    params.theta_end,
                    params.theta_step,
                    dtype=float,
                )
                sweeper = Sweeper(
                    Parameter.relative_phase,
                    theta - data.vphases[ord_pair][target_q],
                    pulses=[theta_pulse],
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
                    sweeper,
                )

                result_target = results[target_q].probability(1)
                result_control = results[control_q].probability(1)

                data.register_qubit(
                    CZVirtualZType,
                    (target_q, control_q, setup),
                    dict(
                        target=result_target,
                        control=result_control,
                    ),
                )
    return data


def fit_function(x, amplitude, offset, phase):
    """Sinusoidal fit function."""
    # return p0 + p1 * np.sin(2*np.pi*p2 * x + p3)
    return np.sin(x + phase) * amplitude + offset


def _fit(
    data: CZVirtualZData,
) -> CZVirtualZResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phase = {}
    cz_angle = {}
    leakage = {}
    for pair in pairs:
        virtual_phase[pair] = {}
        leakage[pair] = {}
        for target, control, setup in data[pair]:
            target_data = data[pair][target, control, setup].target
            pguess = [
                np.max(target_data) - np.min(target_data),
                np.mean(target_data),
                np.pi,
            ]
            try:
                popt, _ = curve_fit(
                    fit_function,
                    np.array(data.thetas) - data.vphases[pair][target],
                    target_data,
                    p0=pguess,
                    bounds=(
                        (0, -np.max(target_data), 0),
                        (np.max(target_data), np.max(target_data), 2 * np.pi),
                    ),
                )
                fitted_parameters[target, control, setup] = popt.tolist()

            except Exception as e:
                log.warning(f"CZ fit failed for pair ({target, control}) due to {e}.")

        try:
            for target_q, control_q in (
                pair,
                list(pair)[::-1],
            ):
                cz_angle[target_q, control_q] = abs(
                    fitted_parameters[target_q, control_q, "X"][2]
                    - fitted_parameters[target_q, control_q, "I"][2]
                )
                virtual_phase[pair][target_q] = fitted_parameters[
                    target_q, control_q, "I"
                ][2]

                # leakage estimate: L = m /2
                # See NZ paper from Di Carlo
                # approximation which does not need qutrits
                # https://arxiv.org/pdf/1903.02492.pdf
                leakage[pair][control_q] = 0.5 * float(
                    np.mean(
                        data[pair][target_q, control_q, "X"].control
                        - data[pair][target_q, control_q, "I"].control
                    )
                )
        except KeyError:
            pass  # exception covered above

    return CZVirtualZResults(
        cz_angle=cz_angle,
        virtual_phase=virtual_phase,
        fitted_parameters=fitted_parameters,
        leakage=leakage,
    )


# TODO: remove str
def _plot(data: CZVirtualZData, fit: CZVirtualZResults, target: QubitPairId):
    """Plot routine for CZVirtualZ."""
    pair_data = data[target]
    qubits = next(iter(pair_data))[:2]
    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {qubits[0]}",
            f"Qubit {qubits[1]}",
        ),
    )
    fitting_report = set()
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {qubits[0]}",
            f"Qubit {qubits[1]}",
        ),
    )

    thetas = data.thetas
    for target_q, control_q, setup in pair_data:
        target_prob = pair_data[target_q, control_q, setup].target
        control_prob = pair_data[target_q, control_q, setup].control
        fig = fig1 if (target_q, control_q) == qubits else fig2
        fig.add_trace(
            go.Scatter(
                x=np.array(thetas) + data.vphases[qubits][target_q],
                y=target_prob,
                name=f"{setup} sequence",
                legendgroup=setup,
            ),
            row=1,
            col=1 if fig == fig1 else 2,
        )

        fig.add_trace(
            go.Scatter(
                x=np.array(thetas) + data.vphases[qubits][control_q],
                y=control_prob,
                name=f"{setup} sequence",
                legendgroup=setup,
            ),
            row=1,
            col=2 if fig == fig1 else 1,
        )
        if fit is not None:
            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            fitted_parameters = fit.fitted_parameters[target_q, control_q, setup]
            fig.add_trace(
                go.Scatter(
                    x=angle_range + data.vphases[qubits][target_q],
                    y=fit_function(
                        angle_range - data.vphases[qubits][target_q],
                        *fitted_parameters,
                    ),
                    name="Fit",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1 if fig == fig1 else 2,
            )

            fitting_report.add(
                table_html(
                    table_dict(
                        [target_q, target_q, control_q],
                        [
                            "CZ angle [rad]",
                            "Virtual Z phase [rad]",
                            "Leakage [a.u.]",
                        ],
                        [
                            np.round(fit.cz_angle[target_q, control_q], 4),
                            np.round(
                                fit.virtual_phase[tuple(sorted(target))][target_q], 4
                            ),
                            np.round(fit.leakage[tuple(sorted(target))][control_q], 4),
                        ],
                    )
                )
            )
    fitting_report.add(
        table_html(
            table_dict(
                [qubits[1], qubits[1]],
                [
                    "Flux pulse amplitude [a.u.]",
                    "Flux pulse duration [ns]",
                ],
                [
                    np.round(data.amplitudes[qubits], 4),
                    np.round(data.durations[qubits], 4),
                ],
            )
        )
    )

    fig1.update_layout(
        title_text=f"Phase correction Qubit {qubits[0]}",
        showlegend=True,
        xaxis1_title="Virtual phase[rad]",
        xaxis2_title="Virtual phase [rad]",
        yaxis_title="State 0 Probability",
    )

    fig2.update_layout(
        title_text=f"Phase correction Qubit {qubits[1]}",
        showlegend=True,
        xaxis1_title="Virtual phase[rad]",
        xaxis2_title="Virtual phase[rad]",
        yaxis_title="State 0 Probability",
    )

    return [fig1, fig2], "".join(fitting_report)  # target and control qubit


def _update(results: CZVirtualZResults, platform: Platform, target: QubitPairId):
    # FIXME: quick fix for qubit order
    qubit_pair = tuple(sorted(target))
    target = tuple(sorted(target))
    update.virtual_phases(results.virtual_phase[target], platform, target)


cz_virtualz = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""CZ virtual Z correction routine."""
