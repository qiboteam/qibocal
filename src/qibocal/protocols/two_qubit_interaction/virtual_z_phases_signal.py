"""CZ virtual correction experiment for two qubit gates, tune landscape."""

from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Routine

from .utils import order_pair
from .virtual_z_phases import (
    VirtualZPhasesData,
    VirtualZPhasesParameters,
    VirtualZPhasesResults,
    VirtualZPhasesType,
    _fit,
)
from .virtual_z_phases import _plot as _plot_prob
from .virtual_z_phases import _update, create_sequence


@dataclass
class VirtualZPhasesSignalParameters(VirtualZPhasesParameters):
    """VirtualZ runcard inputs."""


@dataclass
class VirtualZPhasesSignalResults(VirtualZPhasesResults):
    """VirtualZ outputs when fitting will be done."""


VirtualZPhasesType = np.dtype([("target", np.float64), ("control", np.float64)])


@dataclass
class VirtualZPhasesSignalData(VirtualZPhasesData):
    """VirtualZPhases data."""


def _acquisition(
    params: VirtualZPhasesSignalParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> VirtualZPhasesSignalData:
    r"""
    Acquisition for VirtualZPhases. See https://arxiv.org/pdf/1904.06560.pdf

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
    data = VirtualZPhasesData(native=params.native, thetas=theta_absolute.tolist())
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
                    data.amplitudes[ord_pair],
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
                    params.flux_pulse_amplitude,
                )
                theta = np.arange(
                    params.theta_start,
                    params.theta_end,
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
                        relaxation_time=params.relaxation_time,
                        acquisition_type=AcquisitionType.INTEGRATION,
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                    sweeper,
                )

                result_target = results[target_q].magnitude
                result_control = results[control_q].magnitude

                data.register_qubit(
                    VirtualZPhasesType,
                    (target_q, control_q, setup),
                    dict(
                        target=result_target,
                        control=result_control,
                    ),
                )
    return data


def _plot(
    data: VirtualZPhasesSignalData,
    fit: VirtualZPhasesSignalResults,
    target: QubitPairId,
):
    """Plot routine for VirtualZPhases."""
    figs, fitting_report = _plot_prob(data, fit, target)

    for fig in figs:
        fig.update_layout(
            yaxis_title="Signal [a.u.]",
        )

    return figs, fitting_report


correct_virtual_z_phases_signal = Routine(
    _acquisition, _fit, _plot, _update, two_qubit_gates=True
)
"""Virtual Z correction routine."""
