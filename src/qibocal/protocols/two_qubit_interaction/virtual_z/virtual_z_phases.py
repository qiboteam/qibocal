"""CZ/iSWAP virtual phase correction experiment for two qubit gates."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    Parameter,
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
from qibocal.protocols.utils import Range, RangeLike, table_dict, table_html, to_range

from .... import update
from ..utils import fit_virtualz, order_pair, sinusoid
from .acquisition import create_sequence

__all__ = ["correct_virtual_z_phases"]


@dataclass
class VirtualZPhasesParameters(Parameters):
    """VirtualZ runcard inputs."""

    theta: RangeLike | None = None
    """Angles for the virtual phase sweep [rad].

    See :class:`qibocal.protocols.utils.RangeLike`. Takes precedence over
    ``theta_start``, ``theta_end`` and ``theta_step``.
    """
    theta_start: float | None = None
    """Initial angle for the low frequency qubit measurement in radians."""
    theta_end: float | None = None
    """Final angle for the low frequency qubit measurement in radians."""
    theta_step: float | None = None
    """Step size for the theta sweep in radians."""
    native: Literal["CZ", "iSWAP"] = "CZ"
    """Two-qubit interaction to be calibrated, either CZ or iSWAP."""
    dt: float = 16
    """Time delay between flux pulses and readout."""

    def theta_range(self) -> Range:
        """Range of the virtual phase sweep, as ``(start, stop, step)``."""

        def legacy_range() -> Range:
            assert self.theta_start is not None, (
                "Either theta or theta_start, theta_end, theta_step must be provided."
            )
            assert self.theta_end is not None
            assert self.theta_step is not None
            return (self.theta_start, self.theta_end, self.theta_step)

        return (
            to_range(self.theta, center=0.0)
            if self.theta is not None
            else legacy_range()
        )


@dataclass
class VirtualZPhasesResults(Results):
    """VirtualZ outputs when fitting will be done."""

    fitted_parameters: dict[tuple[str, QubitId], object]
    """Fitted parameters"""
    native: str
    """Native two qubit gate."""
    leakage: dict[QubitPairId, dict[QubitId, float]]
    """Leakage on control qubit for pair."""
    angle: dict[QubitPairId, float] | None = None
    """Native angle.

    For CZ it is the conditional phase (ideally pi). For iSWAP it is the residual
    conditional phase on :math:`|11\\rangle`, which cannot be corrected with
    virtual Z rotations (ideally 0).
    """
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
        for key in self.data:
            pairs.append((key[0], key[1]))
        return np.unique(pairs, axis=0).tolist()


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

    For the iSWAP gate the excitation is exchanged between the two qubits, therefore
    the X90 pulse is played on the partner qubit and the superposition is
    transferred by the gate onto the measured qubit (see
    :func:`~qibocal.protocols.two_qubit_interaction.virtual_z.utils.create_iswap_sequence`).
    """

    data: dict[tuple[QubitId, QubitId, Literal["I", "X"]], npt.NDArray[np.float64]] = {}
    sequences: list[PulseSequence] = []
    vzs: list[VirtualZ] = []
    readouts: dict[
        tuple[QubitId, QubitId, Literal["I", "X"]], tuple[PulseId, PulseId]
    ] = {}

    thetas = np.arange(*params.theta_range())
    # RZ(theta) compiles to VirtualZ(phase=-theta). The fit returns the
    # opposite frame shift, which cancels the gate's unwanted single-qubit phase.
    vzphases = -thetas

    for pair in targets:
        # order the qubits so that the low frequency one is the first
        ordered_pair = order_pair(pair, platform)

        for oscillating_q, steady_q in (
            (ordered_pair[0], ordered_pair[1]),
            (ordered_pair[1], ordered_pair[0]),
        ):
            for setup in ("I", "X"):
                (sequence, _, vz) = create_sequence(
                    platform,
                    setup,
                    oscillating_q,
                    steady_q,
                    ordered_pair,
                    dt=params.dt,
                    native=params.native,
                )

                sequences.append(sequence)
                vzs.append(vz)

                ro_oscillating = list(
                    sequence.channel(platform.qubits[oscillating_q].acquisition)
                )[-1]
                ro_steady = list(
                    sequence.channel(platform.qubits[steady_q].acquisition)
                )[-1]
                readouts[oscillating_q, steady_q, setup] = (
                    ro_oscillating.id,
                    ro_steady.id,
                )

    sweeper = Sweeper(parameter=Parameter.phase, values=vzphases, pulses=vzs)

    results = platform.execute(
        sequences,
        [[sweeper]],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )
    for key, (oscillating_q, steady_q) in readouts.items():
        data[key] = np.stack([results[oscillating_q], results[steady_q]])

    return VirtualZPhasesData(
        data=data,
        thetas=thetas.tolist(),
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
            data[pair], tuple(pair), data.thetas, native=data.native
        )
        fitted_parameters |= new_fitted_parameter
        virtual_phase |= new_phases
        angle |= new_angle
        leakage |= new_leak
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
                            f"{fit.native} angle [rad]"
                            if fit.native == "CZ"
                            else f"{fit.native} conditional phase [rad]",
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
        yaxis2={"range": [0, 1], "title": "State 1 Probability"},
        yaxis_title="State 1 Probability",
    )

    fig2.update_layout(
        title_text=f"Phase correction Qubit {qubits[1]}",
        showlegend=True,
        xaxis1_title="Virtual phase[rad]",
        xaxis2_title="Virtual phase[rad]",
        yaxis1={"range": [0, 1], "title": "State 1 Probability"},
        yaxis_title="State 1 Probability",
    )

    return [fig1, fig2], "".join(fitting_report)  # target and control qubit


def _update(
    results: VirtualZPhasesResults, platform: CalibrationPlatform, target: QubitPairId
):
    # FIXME: quick fix for qubit order
    target = tuple(sorted(target))
    update.virtual_phases(results.virtual_phase, results.native, platform, target)


correct_virtual_z_phases = Protocol(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""Virtual phases correction protocol."""
