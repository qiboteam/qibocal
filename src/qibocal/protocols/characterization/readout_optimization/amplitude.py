import copy
from dataclasses import dataclass, field
from os import error
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit

MAXITER = 50
ERROR_THRESHOLD = 0.03


@dataclass
class RoAmplitudeParameters(Parameters):
    """RoAmplitude runcard inputs."""

    amplitude_start: float
    """Power total width."""
    amplitude_step: float
    """Power step to be probed."""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""


RoAmplitudeType = np.dtype(
    [
        ("amp", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("state", int),
        ("errors", np.float64),
    ]
)
"""Custom dtype for Optimization RO frequency."""


@dataclass
class RoAmplitudeData(Data):
    """Data class for twpa power protocol."""

    data: dict[QubitId, npt.NDArray[RoAmplitudeType]] = field(default_factory=dict)

    def append_data(self, qubit, state, amp, i, q, errors):
        """Append elements to data for single qubit."""
        ar = np.empty(i.shape, dtype=RoAmplitudeType)
        ar["amp"] = amp
        ar["i"] = i
        ar["q"] = q
        ar["state"] = state
        ar["errors"] = errors
        if qubit in self.data.keys():
            self.data[qubit] = np.append(self.data[qubit], np.rec.array(ar))
        else:
            self.data[qubit] = np.rec.array(ar)

    def unique_amplitudes(self, qubit: QubitId) -> np.ndarray:
        return np.unique(self.data[qubit]["amp"])


@dataclass
class RoAmplitudeResults(Results):
    """Result class for twpa power protocol."""

    lowest_errors: dict[QubitId, list]
    best_amp: dict[QubitId, list]


def _acquisition(
    params: RoAmplitudeParameters,
    platform: Platform,
    qubits: Qubits,
) -> RoAmplitudeData:
    r"""
    Data acquisition for TWPA power optmization.
    This protocol perform a classification protocol for twpa powers
    in the range [twpa_power - amplitude_width / 2, twpa_power + amplitude_width / 2]
    with step amplitude_step.

    Args:
        params (:class:`RoAmplitudeParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`TwpaFrequencyData`)
    """

    data = RoAmplitudeData()
    platform_copy = copy.deepcopy(platform)
    for qubit in qubits:
        errors = []
        n = 0
        error = 1
        while error < ERROR_THRESHOLD or n <= MAXITER:
            platform_copy.qubits[qubit].readout.amplitude = (
                params.amplitude_start + n * params.amplitude_step
            )
            n += 1
            sequence_0 = PulseSequence()
            sequence_1 = PulseSequence()

            qd_pulses = platform.create_RX_pulse(qubit, start=0)
            ro_pulses = platform.create_qubit_readout_pulse(
                qubit, start=qd_pulses.finish
            )
            sequence_0.add(ro_pulses)
            sequence_1.add(qd_pulses)
            sequence_1.add(ro_pulses)

            state0_results = platform.execute_pulse_sequence(
                sequence_0,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                ),
            )

            state1_results = platform.execute_pulse_sequence(
                sequence_1,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.INTEGRATION,
                ),
            )
            result0 = state0_results[ro_pulses.serial]
            result1 = state1_results[ro_pulses.serial]

            i_values = np.concatenate((result0.voltage_i, result1.voltage_i))
            q_values = np.concatenate((result0.voltage_q, result1.voltage_q))
            iq_values = np.stack((i_values, q_values), axis=-1)
            nshots = int(len(i_values) / 2)
            states = [0] * nshots + [1] * nshots
            model = QubitFit()
            model.fit(iq_values, np.array(states))
            error = 1 - model.fidelity
            errors.append(error)
            data.append_data(
                qubit=qubit,
                amp=platform_copy.qubits[qubit].readout.amplitude,
                state=states,
                i=i_values,
                q=q_values,
                errors=error,
            )
    return data


def _fit(data: RoAmplitudeData) -> RoAmplitudeResults:
    qubits = data.qubits
    best_amps = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["errors"])
        lowest_err[qubit] = data_qubit["errors"][index_best_err]
        best_amps[qubit] = data_qubit["amp"][index_best_err]

    return RoAmplitudeResults(lowest_err, best_amps)


def _plot(data: RoAmplitudeData, fit: RoAmplitudeResults, qubit):
    """Plotting function for Optimization RO frequency."""
    figures = []
    opacity = 1
    fitting_report = None
    fig = make_subplots(
        rows=1,
        cols=1,
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=data[qubit]["amp"],
                y=data[qubit]["errors"],
                opacity=opacity,
                showlegend=True,
            ),
            row=1,
            col=1,
        )

        fitting_report = "" + (
            f"{qubit} | Best Readout Amplitude : {fit.best_amp[qubit]:,.4f}<br>"
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Readout Amplitude (GHz)",
        yaxis_title="Errors",
    )

    figures.append(fig)

    return figures, fitting_report


ro_amplitude = Routine(_acquisition, _fit, _plot)
"""Twpa power Routine  object."""
