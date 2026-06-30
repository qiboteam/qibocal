"""CZ virtual correction experiment for two qubit gates, tune landscape."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

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
    PulseId,
    PulseSequence,
    Sweeper,
    VirtualZ,
)

from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.calibration.calibration import QubitId
from qibocal.protocols.utils import table_dict, table_html

from ... import update
from ...update import replace
from .utils import fit_sinusoid, fit_virtualz, order_pair, phase_diff, sinusoid

__all__ = ["correct_virtual_z_phases", "create_sequence", "fit_sinusoid", "phase_diff"]


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
    dt: float | None = 16
    """Time delay between flux pulses and readout."""
    gate_repetition: int = 1
    """Number of CZ repetition"""
    sweep: bool = True
    """Toggle sweeping vs unrolling.


    .. tip::

       Sweeping phases has been proven fragile in certain platforms. For this reason, we
       are now providing the double implementation.
       The difference it is just an internal detail, and ideally the result is supposed
       to be the exact same when sweeping phases is properly supported. Sweeping is
       often a bit faster.

       The recommended usage is just to toggle this to ``False`` when the user is
       confident the sweeping is not working properly. Or just to check whether that is
       actually the problem or not.

    """


@dataclass
class VirtualZPhasesResults(Results):
    """VirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId], object]
    """Fitted parameters"""
    native: str
    """Native two qubit gate."""
    gate_repetition: int
    leakage: dict[QubitPairId, dict[QubitId, float]]
    """Leakage on control qubit for pair."""
    angle: dict[QubitPairId, float] | None = None
    """Native angle."""
    virtual_phase: dict[QubitPairId, dict[QubitId, float]] | None = None
    """Virtual Z phase correction."""

    def __contains__(self, key: QubitPairId):
        """Check if key is in class.
        While key is a QubitPairId both chsh and chsh_mitigated contain
        an additional key which represents the basis chosen.
        """
        return key in [(target, control) for target, control in self.angle]


@dataclass
class VirtualZPhasesData(Data):
    """VirtualZPhases data."""

    gate_repetition: int
    data: dict[tuple[QubitId, QubitId, Literal["I", "X"]], npt.NDArray[np.float64]]
    thetas: list[float]
    native: str = "CZ"

    def __getitem__(
        self, pair: QubitPairId
    ) -> dict[tuple[QubitId, QubitId, Literal["I", "X"]], npt.NDArray[np.float64]]:
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }

    @property
    def order_pairs(self) -> list[QubitPairId]:
        pairs: list[QubitPairId] = []
        for key in self.data.keys():
            pairs.append((key[0], key[1]))
        return np.unique(pairs, axis=0).tolist()


def create_sequence(
    platform: CalibrationPlatform,
    setup: Literal["I", "X"],
    target_qubit: QubitId,
    control_qubit: QubitId,
    ordered_pair: tuple[QubitId, QubitId],
    native: Literal["CZ", "iSWAP"],
    dt: float,
    vzphase: float = 0.0,
    flux_pulse_max_duration: float | None = None,
    flux_pulse: Pulse | None = None,
) -> tuple[PulseSequence, Pulse, VirtualZ]:
    """
    Create the pulse sequence for the calibration of two-qubit gate virtual phases.

    This function constructs a pulse sequence for a given two-qubit native gate `native` (CZ or iSWAP)
    on the specified qubits. The sequence includes:
    - A preliminary RX90 pulse on the `target_qubit`.
    - An optional X pulse on the `control_qubit` based on the `setup` type.
    - A flux pulse implementing the two-qubit native gate.
    - A delay of duration `dt` before the final X90 pulse on the target qubit.
    - Measurement pulses.
    It is possible to specify the maximum duration for the flux pulses with the
    `flux_pulse_max_duration` parameter.

    The function returns:
    - The full experiment pulse sequence.
    - The applied flux pulse.
    - The final `VirtualZPhase` pulses to be used for phase sweeping.
    """

    target_natives = platform.natives.single_qubit[target_qubit]
    control_natives = platform.natives.single_qubit[control_qubit]

    sequence = PulseSequence()
    # X90
    sequence += target_natives.R(theta=np.pi / 2)
    # X
    if setup == "X":
        assert control_natives.RX is not None
        sequence += control_natives.RX()

    flux_channel = platform.qubits[ordered_pair[1]].flux
    assert flux_channel is not None, (
        "Flux channel must be defined for the control qubit."
    )
    # CZ
    if flux_pulse is None:
        cz_sequence = platform.natives.two_qubit[ordered_pair].CZ
        flux_pulse = list(cz_sequence.channel(flux_channel))[0]
    assert isinstance(flux_pulse, Pulse), "Flux pulse must be a Pulse object."

    if flux_pulse_max_duration is not None:
        flux_pulse = replace(flux_pulse, duration=flux_pulse_max_duration)

    sequence |= PulseSequence(
        [
            (flux_channel, Delay(duration=dt)),
            (flux_channel, flux_pulse),
            (flux_channel, Delay(duration=dt)),
        ]
    )

    vz = VirtualZ(phase=vzphase)
    sequence.append((platform.qubits[target_qubit].drive, vz))

    sequence |= target_natives.R(theta=np.pi / 2)

    # X gate for the leakage
    if setup == "X":
        sequence += control_natives.RX()

    sequence |= PulseSequence(
        [
            target_natives.MZ()[0],
            control_natives.MZ()[0],
        ]
    )

    return sequence, flux_pulse, vz


def _acquisition(
    params: VirtualZPhasesParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> VirtualZPhasesData:
    r"""
    Acquisition for VirtualZPhases.

    Check the two-qubit landscape created by a flux pulse of a given duration
    and amplitude.
    The system is initialized with a X90 pulse on the low frequency qubit and either
    an Id or an X gate on the high frequency qubit. Then the flux pulse is applied to
    the high frequency qubit in order to perform a two-qubit interaction.
    A $X_{\beta}90$ pulse is applied to the low frequency qubit before measurement.
    That is, a pi-half pulse around the relative phase parametereized by the angle theta.
    Measurements on the low frequency qubit yield the 2Q-phase of the gate and the
    remnant single qubit Z phase aquired during the execution to be corrected.
    Population of the high frequency qubit yield the leakage to the non-computational states
    during the execution of the flux pulse.
    """
    assert params.native == "CZ", "This protocol supports only CZ gate."

    data: dict[tuple[QubitId, QubitId, Literal["I", "X"]], npt.NDArray[np.float64]] = {}
    sequences: list[PulseSequence] = []
    vzs: list[VirtualZ] = []
    readouts: dict[
        tuple[QubitId, QubitId, Literal["I", "X"]], list[tuple[PulseId, PulseId]]
    ] = defaultdict(list)

    # The virtual phase values are the opposite of beta, this is
    # because, according to the circuit we would like to reproduce
    # after the CZ, an RZ is applied. The RZ gate with `theta` angle
    # is compiled into  a VirtualPhase pulse with phase `-theta`.
    # (See https://github.com/qiboteam/qibolab/pull/1044#issuecomment-2354622956)
    sweeper = Sweeper(
        parameter=Parameter.phase,
        range=(
            -params.gate_repetition * params.theta_start,
            -params.gate_repetition * params.theta_end,
            -params.gate_repetition * params.theta_step,
        ),
        pulses=vzs,
    )
    phases = [0.0] if params.sweep else np.arange(*sweeper.irange).tolist()

    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)

        for target_q, control_q in (
            (ordered_pair[0], ordered_pair[1]),
            (ordered_pair[1], ordered_pair[0]),
        ):
            for setup in ("I", "X"):
                for phase in phases:
                    (sequence, _, vz) = create_sequence(
                        platform,
                        setup,
                        target_q,
                        control_q,
                        ordered_pair,
                        params.native,
                        dt=params.dt,
                        vzphase=phase,
                    )

                    sequences.append(sequence)
                    vzs.append(vz)

                    ro_target = list(
                        sequence.channel(platform.qubits[target_q].acquisition)
                    )[-1]
                    ro_control = list(
                        sequence.channel(platform.qubits[control_q].acquisition)
                    )[-1]
                    readouts[target_q, control_q, setup].append(
                        (ro_target.id, ro_control.id)
                    )

    sweep = [[sweeper]] if params.sweep else []

    results = platform.execute(
        sequences,
        sweep,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    if params.sweep:
        for key, [(target, control)] in readouts.items():
            data[key] = np.stack([results[target], results[control]])
    else:
        for key, meas in readouts.items():
            res = [[results[target], results[control]] for target, control in meas]
            data[key] = np.moveaxis(res, 0, 1)

    return VirtualZPhasesData(
        data=data,
        gate_repetition=params.gate_repetition,
        thetas=np.arange(
            params.theta_start, params.theta_end, params.theta_step
        ).tolist(),
        native=params.native,
    )


def _fit(
    data: VirtualZPhasesData,
) -> VirtualZPhasesResults:
    r"""Fitting routine for the experiment.

    The used model is

    .. math::

        y = p_0 sin\Big(x + p_2\Big) + p_1.
    """
    fitted_parameters = {}
    pairs = data.order_pairs
    virtual_phase = {}
    angle = {}
    leakage = {}
    for pair in pairs:
        new_fitted_parameter, new_phases, new_angle, new_leak = fit_virtualz(
            data[pair], tuple(pair), data.thetas, data.gate_repetition
        )
        fitted_parameters |= new_fitted_parameter
        virtual_phase |= new_phases
        angle |= new_angle
        leakage |= new_leak
    return VirtualZPhasesResults(
        native=data.native,
        gate_repetition=data.gate_repetition,
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
        target_prob = pair_data[target_q, control_q, setup][0]
        control_prob = pair_data[target_q, control_q, setup][1]
        fig = fig1 if (target_q, control_q) == qubits else fig2
        fig.add_trace(
            go.Scatter(
                x=np.array(thetas),
                y=target_prob,
                mode="markers",
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
                mode="markers",
                name=f"{setup} sequence",
                legendgroup=setup,
            ),
            row=1,
            col=2 if fig == fig1 else 1,
        )
        if fit is not None:
            angle_range = np.linspace(thetas[0], thetas[-1], 100)
            fitted_parameters = fit.fitted_parameters[(target_q, control_q), setup]
            fig.add_trace(
                go.Scatter(
                    x=angle_range,
                    y=sinusoid(
                        angle_range,
                        data.gate_repetition,
                        *fitted_parameters,
                    ),
                    name="Fit",
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
                                fit.virtual_phase[target_q, control_q],
                                4,
                            ),
                            np.round(fit.leakage[target_q, control_q], 4),
                        ],
                    )
                )
            )

    fig1.update_layout(
        title_text=f"Phase correction Qubit {qubits[0]}",
        showlegend=True,
        xaxis1_title="Virtual phase[rad]",
        xaxis2_title="Virtual phase [rad]",
        yaxis2=dict(range=[0, 1], title="State 1 Probability"),
        yaxis_title="State 1 Probability",
    )

    fig2.update_layout(
        title_text=f"Phase correction Qubit {qubits[1]}",
        showlegend=True,
        xaxis1_title="Virtual phase[rad]",
        xaxis2_title="Virtual phase[rad]",
        yaxis1=dict(range=[0, 1], title="State 1 Probability"),
        yaxis_title="State 1 Probability",
    )

    return [fig1, fig2], "".join(fitting_report)  # target and control qubit


def _update(
    results: VirtualZPhasesResults, platform: CalibrationPlatform, target: QubitPairId
):
    if results.gate_repetition == 1:
        # FIXME: quick fix for qubit order
        target = tuple(sorted(target))
        update.virtual_phases(results.virtual_phase, results.native, platform, target)


correct_virtual_z_phases = Protocol(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""Virtual phases correction protocol."""
