"""Virtual correction experiment for two qubit gates, tune landscape."""

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
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.protocols.characterization.two_qubit_interaction.chevron import order_pair
from qibocal.protocols.characterization.utils import table_dict, table_html


@dataclass
class VirtualZParameters(Parameters):
    """VirtualZ runcard inputs."""

    theta_start: float
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float
    """Step size for the theta sweep in radians."""
    flux_pulse_amplitude: Optional[float] = None
    """Amplitude of flux pulse implementing CZ."""
    native_gate: Optional[str] = "CZ"
    """Native gate to implement, CZ or iSWAP."""
    dt: Optional[float] = 20
    """Time delay between flux pulses and readout."""
    parking: bool = True
    """Wether to park non interacting qubits or not."""


@dataclass
class VirtualZResults(Results):
    """CzVirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId],]
    """Fitted parameters"""
    native_gate_angle: dict[QubitPairId, float]
    """Angle."""
    virtual_phase: dict[QubitPairId, dict[QubitId, float]]
    """Virtual Z phase correction."""


VirtualZType = np.dtype([("target", np.float64), ("control", np.float64)])


@dataclass
class VirtualZData(Data):
    """VirtualZ data."""

    data: dict[tuple, npt.NDArray[VirtualZType]] = field(default_factory=dict)
    thetas: list = field(default_factory=list)
    vphases: dict[QubitPairId, dict[QubitId, float]] = field(default_factory=dict)
    amplitudes: dict[tuple[QubitId, QubitId], float] = field(default_factory=dict)

    def register_qubit(self, target, control, setup, prob_target, prob_control):
        ar = np.empty(prob_target.shape, dtype=VirtualZType)
        ar["target"] = prob_target
        ar["control"] = prob_control
        self.data[target, control, setup] = np.rec.array(ar)

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
    gate,
    parking: bool,
    dt: float,
    amplitude: float = None,
) -> tuple[
    PulseSequence, dict[QubitId, Pulse], dict[QubitId, Pulse], dict[QubitId, Pulse]
]:
    """Create the experiment PulseSequence."""

    sequence = PulseSequence()

    Y90_pulse = platform.create_RX90_pulse(
        target_qubit, start=0, relative_phase=np.pi / 2
    )
    RX_pulse_start = platform.create_RX_pulse(control_qubit, start=0, relative_phase=0)

    if gate == "CZ":
        native_gate, virtual_z_phase = platform.create_CZ_pulse_sequence(
            (ordered_pair[1], ordered_pair[0]),
            start=max(Y90_pulse.finish, RX_pulse_start.finish),
        )
    elif gate == "iSWAP":
        native_gate, virtual_z_phase = platform.create_iSWAP_pulse_sequence(
            (ordered_pair[1], ordered_pair[0]),
            start=max(Y90_pulse.finish, RX_pulse_start.finish),
        )

    # TODO: Do we need an independent amplitude similar for coupler amplitude ?
    if amplitude is not None:
        native_gate.get_qubit_pulses(ordered_pair[1])[0].amplitude = amplitude

    theta_pulse = platform.create_RX90_pulse(
        target_qubit,
        start=native_gate.finish + dt,
        relative_phase=virtual_z_phase[target_qubit],
    )

    RX_pulse_end = platform.create_RX_pulse(
        control_qubit,
        start=native_gate.finish + dt,
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
        native_gate.get_qubit_pulses(ordered_pair[1]),
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
        for pulse in native_gate:
            if pulse.qubit not in ordered_pair:
                pulse.duration = theta_pulse.finish
                sequence.add(pulse)

    # TODO: Do we need an independent amplitude similar for coupler amplitude ?
    return (
        sequence,
        virtual_z_phase,
        theta_pulse,
        native_gate.get_qubit_pulses(ordered_pair[1])[0].amplitude,
    )


def _acquisition(
    params: VirtualZParameters,
    platform: Platform,
    targets: Qubits,
) -> VirtualZData:
    r"""
    Acquisition for VirtualZ.

    Check the two-qubit landscape created by a flux pulse of a given duration
    and amplitude.
    The system is initialized with a Y90 pulse on the low frequency qubit and either
    an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to
    the high frequency qubit in order to perform a two-qubit interaction. The Id/X gate
    is undone in the high frequency qubit and a theta90 pulse is applied to the low
    frequency qubit before measurement. That is, a pi-half pulse around the relative phase
    parametereized by the angle theta.
    Measurements on the low frequency qubit yield the the 2Q-phase of the gate and the
    remnant single qubit Z phase aquired during the execution to be corrected.
    Population of the high frequency qubit yield the leakage to the non-computational states
    during the execution of the flux pulse.
    """

    theta_absolute = np.arange(params.theta_start, params.theta_end, params.theta_step)

    data = VirtualZData(thetas=theta_absolute.tolist())
    for pair in targets:
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
                    data.amplitudes[ord_pair],
                ) = create_sequence(
                    platform,
                    setup,
                    target_q,
                    control_q,
                    ord_pair,
                    params.native_gate,
                    params.dt,
                    params.parking,
                    params.flux_pulse_amplitude,
                )
                data.vphases[ord_pair] = dict(virtual_z_phase)
                theta = np.arange(
                    virtual_z_phase[target_q] + params.theta_start,
                    virtual_z_phase[target_q] + params.theta_end,
                    params.theta_step,
                    dtype=float,
                )
                sweeper = Sweeper(
                    Parameter.relative_phase,
                    theta,
                    pulses=[theta_pulse],
                    type=SweeperType.ABSOLUTE,
                )
                results = platform.sweep(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        acquisition_type=AcquisitionType.INTEGRATION,
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                    sweeper,
                )

                result_target = results[target_q].magnitude
                result_control = results[control_q].magnitude

                data.register_qubit(
                    target=target_q,
                    control=control_q,
                    setup=setup,
                    prob_target=result_target,
                    prob_control=result_control,
                )
    return data


def fit_function(x, p0, p1, p2):
    """Sinusoidal fit function."""
    # return p0 + p1 * np.sin(2*np.pi*p2 * x + p3)
    return np.sin(x + p2) * p0 + p1


def _fit(
    data: VirtualZData,
) -> VirtualZResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    fitted_parameters = {}
    pairs = data.pairs
    virtual_phase = {}
    native_gate_angle = {}
    for pair in pairs:
        virtual_phase[pair] = {}
        for target, control, setup in data[pair]:
            target_data = data[pair][target, control, setup].target
            pguess = [
                np.max(target_data) - np.min(target_data),
                np.mean(target_data),
                3.14,
            ]

            try:
                popt, _ = curve_fit(
                    fit_function,
                    np.array(data.thetas) + data.vphases[pair][target],
                    target_data,
                    p0=pguess,
                    bounds=((0, 0, 0), (2.5, 2.5, 2 * np.pi)),
                )
                fitted_parameters[target, control, setup] = popt.tolist()

            except:
                log.warning("landscape_fit: the fitting was not succesful")
                fitted_parameters[target, control, setup] = [0] * 3

        for target_q, control_q in (
            pair,
            list(pair)[::-1],
        ):
            native_gate_angle[target_q, control_q] = abs(
                fitted_parameters[target_q, control_q, "X"][2]
                - fitted_parameters[target_q, control_q, "I"][2]
            )
            virtual_phase[pair][target_q] = -fitted_parameters[
                target_q, control_q, "I"
            ][2]

    return VirtualZResults(
        native_gate_angle=native_gate_angle,
        virtual_phase=virtual_phase,
        fitted_parameters=fitted_parameters,
    )


# TODO: remove str
def _plot(data: VirtualZData, fit: VirtualZResults, target):
    """Plot routine for VirtualZ."""
    pair_data = data[target]
    targets = next(iter(pair_data))[:2]
    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {targets[0]}",
            f"Qubit {targets[1]}",
        ),
    )
    reports = []
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Qubit {targets[0]}",
            f"Qubit {targets[1]}",
        ),
    )

    fitting_report = ""
    thetas = data.thetas
    for target, control, setup in pair_data:
        target_prob = pair_data[target, control, setup].target
        control_prob = pair_data[target, control, setup].control
        fig = fig1 if (target, control) == targets else fig2
        fig.add_trace(
            go.Scatter(
                x=np.array(thetas) + data.vphases[targets][target],
                y=target_prob,
                name=f"{setup} sequence",
                legendgroup=setup,
            ),
            row=1,
            col=1 if fig == fig1 else 2,
        )

        fig.add_trace(
            go.Scatter(
                x=np.array(thetas) + data.vphases[targets][control],
                y=control_prob,
                name=f"{setup} sequence",
                legendgroup=setup,
            ),
            row=1,
            col=2 if fig == fig1 else 1,
        )
        if fit is not None:
            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            fitted_parameters = fit.fitted_parameters[target, control, setup]
            fig.add_trace(
                go.Scatter(
                    x=angle_range + data.vphases[targets][target],
                    y=fit_function(
                        angle_range + data.vphases[targets][target],
                        *fitted_parameters,
                    ),
                    name="Fit",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1 if fig == fig1 else 2,
            )

            fitting_report = table_html(
                table_dict(
                    [target, target, targets[1]],
                    [" angle", "Virtual Z phase", "Flux pulse amplitude"],
                    [
                        np.round(fit.native_gate_angle[target, control], 4),
                        np.round(fit.virtual_phase[tuple(sorted(target))][target], 4),
                        np.round(data.amplitudes[targets]),
                    ],
                )
            )

    fig1.update_layout(
        title_text=f"Phase correction Qubit {targets[0]}",
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis1_title="theta [rad] + virtual phase[rad]",
        xaxis2_title="theta [rad] + virtual phase [rad]",
        yaxis_title="MSR[V]",
    )

    fig2.update_layout(
        title_text=f"Phase correction Qubit {targets[1]}",
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis1_title="theta [rad] + virtual phase[rad]",
        xaxis2_title="theta [rad] + virtual phase[rad]",
        yaxis_title="MSR[V]",
    )

    return [fig1, fig2], fitting_report


def _update(results: VirtualZResults, platform: Platform, qubit_pair: QubitPairId):
    if qubit_pair[0] > qubit_pair[1]:
        qubit_pair = (qubit_pair[1], qubit_pair[0])
    update.virtual_phases(results.virtual_phase[qubit_pair], platform, qubit_pair)


native_gate_virtualz = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""virtual Z correction routine."""
