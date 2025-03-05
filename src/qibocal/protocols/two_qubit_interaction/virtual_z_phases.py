"""CZ virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Delay,
    Parameter,
    Pulse,
    PulseSequence,
    Sweeper,
    VirtualZ,
)
from scipy.optimize import curve_fit

from qibocal.auto.operation import (
    Data,
    Parameters,
    QubitId,
    QubitPairId,
    Results,
    Routine,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html

from ... import update
from ...update import replace
from .utils import order_pair


@dataclass
class VirtualZPhasesParameters(Parameters):
    """VirtualZ runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    native: str = "CZ"
    """Two qubit interaction to be calibrated.

    iSWAP and CZ are the possible options.

    """
    dt: Optional[float] = 0
    """Time delay between flux pulses and readout."""


@dataclass
class VirtualZPhasesResults(Results):
    """VirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId],]
    """Fitted parameters"""
    native: str
    """Native two qubit gate."""
    angle: dict[QubitPairId, float]
    """Native angle."""
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


VirtualZPhasesType = np.dtype([("target", np.float64), ("control", np.float64)])


@dataclass
class VirtualZPhasesData(Data):
    """VirtualZPhases data."""

    data: dict[tuple, npt.NDArray[VirtualZPhasesType]] = field(default_factory=dict)
    native: str = "CZ"
    thetas: list = field(default_factory=list)

    def __getitem__(self, pair):
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }


def create_sequence(
    platform: CalibrationPlatform,
    setup: Literal["I", "X"],
    target_qubit: QubitId,
    control_qubit: QubitId,
    ordered_pair: list[QubitId, QubitId],
    native: Literal["CZ", "iSWAP"],
    dt: float,
    flux_pulse_max_duration: float = None,
) -> tuple[
    PulseSequence,
    Pulse,
    Pulse,
]:
    """Create the experiment PulseSequence."""

    target_natives = platform.natives.single_qubit[target_qubit]
    control_natives = platform.natives.single_qubit[control_qubit]

    sequence = PulseSequence()
    # Y90
    sequence += target_natives.R(theta=np.pi / 2, phi=np.pi / 2)
    # X
    if setup == "X":
        sequence += control_natives.RX()
    else:
        sequence.append(
            (platform.qubits[control_qubit].drive, Delay(duration=sequence.duration))
        )

    drive_duration = sequence.duration

    # CZ
    flux_sequence = getattr(platform.natives.two_qubit[ordered_pair], native)()
    flux_pulses = [
        (ch, pulse) for ch, pulse in flux_sequence if not isinstance(pulse, VirtualZ)
    ]

    if flux_pulse_max_duration is not None:
        flux_pulses[0] = (
            flux_pulses[0][0],
            replace(flux_pulses[0][1], duration=flux_pulse_max_duration),
        )

    _, flux_pulse = flux_pulses[0]
    flux_sequence = PulseSequence(flux_pulses)
    sequence |= flux_sequence

    flux_duration = flux_sequence.duration
    dt_delay = Delay(duration=dt)

    theta_sequence = PulseSequence(
        [
            (
                platform.qubits[target_qubit].drive,
                dt_delay,
            ),
            (
                platform.qubits[control_qubit].drive,
                dt_delay,
            ),
            (
                platform.qubits[target_qubit].drive,
                Delay(duration=flux_duration),
            ),
            (
                platform.qubits[control_qubit].drive,
                Delay(duration=flux_duration),
            ),
        ]
    )
    # R90 (angle to be swept)
    theta_sequence += target_natives.R(theta=np.pi / 2, phi=0)
    theta_pulse = theta_sequence[-1][1]
    # X
    if setup == "X":
        theta_sequence += control_natives.RX()

    sequence += theta_sequence

    ro_target_delay = Delay(duration=flux_duration)
    ro_control_delay = Delay(duration=flux_duration)

    ro_sequence = PulseSequence(
        [
            (
                platform.qubits[target_qubit].acquisition,
                Delay(duration=drive_duration),
            ),
            (
                platform.qubits[control_qubit].acquisition,
                Delay(duration=drive_duration),
            ),
            (
                platform.qubits[target_qubit].acquisition,
                ro_target_delay,
            ),
            (
                platform.qubits[control_qubit].acquisition,
                ro_control_delay,
            ),
            (
                platform.qubits[target_qubit].acquisition,
                Delay(duration=theta_sequence.duration),
            ),
            (
                platform.qubits[control_qubit].acquisition,
                Delay(duration=theta_sequence.duration),
            ),
            target_natives.MZ()[0],
            control_natives.MZ()[0],
        ]
    )

    sequence += ro_sequence

    return sequence, flux_pulse, theta_pulse, [ro_target_delay, ro_control_delay]


def _acquisition(
    params: VirtualZPhasesParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> VirtualZPhasesData:
    r"""
    Acquisition for VirtualZPhases.

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
    data = VirtualZPhasesData(thetas=theta_absolute.tolist(), native=params.native)
    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)

        for target_q, control_q in (
            (ordered_pair[0], ordered_pair[1]),
            (ordered_pair[1], ordered_pair[0]),
        ):
            for setup in ("I", "X"):
                (
                    sequence,
                    _,
                    theta_pulse,
                    _,
                ) = create_sequence(
                    platform,
                    setup,
                    target_q,
                    control_q,
                    ordered_pair,
                    params.native,
                    params.dt,
                )
                sweeper = Sweeper(
                    parameter=Parameter.relative_phase,
                    range=(params.theta_start, params.theta_end, params.theta_step),
                    pulses=[theta_pulse],
                )
                results = platform.execute(
                    [sequence],
                    [[sweeper]],
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.CYCLIC,
                )

                ro_target = list(
                    sequence.channel(platform.qubits[target_q].acquisition)
                )[-1]
                ro_control = list(
                    sequence.channel(platform.qubits[control_q].acquisition)
                )[-1]
                result_target = results[ro_target.id]
                result_control = results[ro_control.id]

                data.register_qubit(
                    VirtualZPhasesType,
                    (target_q, control_q, setup),
                    dict(
                        target=result_target,
                        control=result_control,
                    ),
                )
    return data


def sinusoid(x, amplitude, offset, phase):
    """Sinusoidal fit function."""
    return np.sin(x + phase) * amplitude + offset


def phase_diff(phase_1, phase_2):
    """Return the phase difference of two sinusoids, normalized in the range [0, pi]."""
    return np.arccos(np.cos(phase_1 - phase_2))


def fit_sinusoid(thetas, data):
    """Fit sinusoid to the given data."""
    pguess = [
        np.max(data) - np.min(data),
        np.mean(data),
        np.pi,
    ]

    popt, _ = curve_fit(
        sinusoid,
        thetas,
        data,
        p0=pguess,
        bounds=(
            (0, -np.max(data), 0),
            (np.max(data), np.max(data), 2 * np.pi),
        ),
    )

    return popt.tolist()


def _fit(
    data: VirtualZPhasesData,
) -> VirtualZPhasesResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phase = {}
    angle = {}
    leakage = {}
    for pair in pairs:
        virtual_phase[pair] = {}
        leakage[pair] = {}
        for target, control, setup in data[pair]:
            target_data = data[pair][target, control, setup].target
            try:
                params = fit_sinusoid(np.array(data.thetas), target_data)
                fitted_parameters[target, control, setup] = params

            except Exception as e:
                log.warning(f"CZ fit failed for pair ({target, control}) due to {e}.")

        try:
            for target_q, control_q in (
                pair,
                list(pair)[::-1],
            ):
                angle[target_q, control_q] = phase_diff(
                    fitted_parameters[target_q, control_q, "X"][2],
                    fitted_parameters[target_q, control_q, "I"][2],
                )
                virtual_phase[pair][target_q] = -fitted_parameters[
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
    return VirtualZPhasesResults(
        native=data.native,
        angle=angle,
        virtual_phase=virtual_phase,
        fitted_parameters=fitted_parameters,
        leakage=leakage,
    )


def _plot(data: VirtualZPhasesData, fit: VirtualZPhasesResults, target: QubitPairId):
    """Plot routine for VirtualZPhases."""
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
                x=np.array(thetas),
                y=target_prob,
                name=f"{setup} sequence",
                legendgroup=setup,
            ),
            row=1,
            col=1 if fig == fig1 else 2,
        )

        fig.add_trace(
            go.Scatter(
                x=np.array(thetas),
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
                    x=angle_range,
                    y=sinusoid(
                        angle_range,
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
                            f"{fit.native} angle [rad]",
                            "Virtual Z phase [rad]",
                            "Leakage [a.u.]",
                        ],
                        [
                            np.round(fit.angle[target_q, control_q], 4),
                            np.round(
                                fit.virtual_phase[tuple(sorted(target))][target_q], 4
                            ),
                            np.round(fit.leakage[tuple(sorted(target))][control_q], 4),
                        ],
                    )
                )
            )

    fig1.update_layout(
        title_text=f"Phase correction Qubit {qubits[0]}",
        showlegend=True,
        xaxis1_title="Virtual phase[rad]",
        xaxis2_title="Virtual phase [rad]",
        yaxis_title="State 1 Probability",
    )

    fig2.update_layout(
        title_text=f"Phase correction Qubit {qubits[1]}",
        showlegend=True,
        xaxis1_title="Virtual phase[rad]",
        xaxis2_title="Virtual phase[rad]",
        yaxis_title="State 1 Probability",
    )

    return [fig1, fig2], "".join(fitting_report)  # target and control qubit


def _update(
    results: VirtualZPhasesResults, platform: CalibrationPlatform, target: QubitPairId
):
    # FIXME: quick fix for qubit order
    target = tuple(sorted(target))
    update.virtual_phases(
        results.virtual_phase[target], results.native, platform, target
    )


correct_virtual_z_phases = Routine(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""Virtual phases correction protocol."""
