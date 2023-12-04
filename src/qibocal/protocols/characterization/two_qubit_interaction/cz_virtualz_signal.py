"""CZ virtual correction experiment for two qubit gates, tune landscape."""
from dataclasses import dataclass

import numpy as np
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import QubitsPairs, Routine
from qibocal.protocols.characterization.two_qubit_interaction.chevron import order_pair

from .cz_virtualz import (
    CZVirtualZData,
    CZVirtualZParameters,
    CZVirtualZResults,
    CZVirtualZType,
    _fit,
)
from .cz_virtualz import _plot as _plot_prob
from .cz_virtualz import _update, create_sequence


@dataclass
class CZVirtualZSignalParameters(CZVirtualZParameters):
    """CzVirtualZ runcard inputs."""


@dataclass
class CZVirtualZSignalResults(CZVirtualZResults):
    """CzVirtualZ outputs when fitting will be done."""


CZVirtualZType = np.dtype([("target", np.float64), ("control", np.float64)])


@dataclass
class CZVirtualZSignalData(CZVirtualZData):
    """CZVirtualZ data."""


def _acquisition(
    params: CZVirtualZSignalParameters,
    platform: Platform,
    qubits: QubitsPairs,
) -> CZVirtualZSignalData:
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
                    data.amplitudes[ord_pair],
                    data.durations[ord_pair],
                ) = create_sequence(
                    platform,
                    setup,
                    target_q,
                    control_q,
                    ord_pair,
                    params,
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
                    CZVirtualZType,
                    (target_q, control_q, setup),
                    dict(
                        target=result_target,
                        control=result_control,
                    ),
                )
    return data


def _plot(data: CZVirtualZSignalData, fit: CZVirtualZSignalResults, qubit):
    """Plot routine for CZVirtualZ."""
    figs, fitting_report = _plot_prob(data, fit, qubit)

    for fig in figs:
        fig.update_layout(
            yaxis_title="Signal [a.u.]",
        )

    return figs, fitting_report


cz_virtualz_signal = Routine(_acquisition, _fit, _plot, _update, two_qubit_gates=True)
"""CZ virtual Z correction routine."""
