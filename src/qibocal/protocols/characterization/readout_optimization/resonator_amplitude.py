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


@dataclass
class ResonatorAmplitudeParameters(Parameters):
    """ResonatorAmplitude runcard inputs."""

    amplitude_step: float
    """Amplituude step to be probed."""
    amplitude_start: float = 0.0
    """Amplitude start."""
    amplitude_stop: float = 1.0
    """Amplitude stop value"""
    nshots: Optional[int] = None
    """Number of shots."""
    relaxation_time: Optional[int] = None
    """Relaxation time (ns)."""
    error_threshold: float = 0.003
    """Probability error threshold to stop the best amplitude search"""


ResonatorAmplitudeType = np.dtype(
    [
        ("amp", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("state", int),
        ("errors", np.float64),
    ]
)
"""Custom dtype for Optimization RO amplitude."""


@dataclass
class ResonatorAmplitudeData(Data):
    """Data class for `resoantor_amplitude` protocol."""

    data: dict[QubitId, npt.NDArray[ResonatorAmplitudeType]] = field(
        default_factory=dict
    )

    def append_data(self, qubit, state, amp, i, q, errors):
        """Append elements to data for single qubit."""
        ar = np.empty(i.shape, dtype=ResonatorAmplitudeType)
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
class ResonatorAmplitudeResults(Results):
    """Result class for `resonator_amplitude` protocol."""

    lowest_errors: dict[QubitId, list]
    best_amp: dict[QubitId, list]


def _acquisition(
    params: ResonatorAmplitudeParameters,
    platform: Platform,
    qubits: Qubits,
) -> ResonatorAmplitudeData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol perform a classification protocol for amplitude optimization
    with step amplitude_step.

    Args:
        params (:class:`ResonatorAmplitudeParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        qubits (dict): dict of target :class:`Qubit` objects to be characterized

    Returns:
        data (:class:`ResonatorAmplitudeData`)
    """

    data = ResonatorAmplitudeData()
    for qubit in qubits:
        n = 0
        error = 1
        old_amp = platform.qubits[qubit].native_gates.MZ.amplitude
        new_amp = params.amplitude_start
        while error > params.error_threshold and new_amp <= params.amplitude_stop:
            platform.qubits[qubit].native_gates.MZ.amplitude = new_amp
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
            error = model.probability_error
            data.append_data(
                qubit=qubit,
                amp=new_amp,
                state=states,
                i=i_values,
                q=q_values,
                errors=error,
            )
            platform.qubits[qubit].native_gates.MZ.amplitude = old_amp
            n += 1
            new_amp += params.amplitude_step
    return data


def _fit(data: ResonatorAmplitudeData) -> ResonatorAmplitudeResults:
    qubits = data.qubits
    best_amps = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["errors"])
        lowest_err[qubit] = data_qubit["errors"][index_best_err]
        best_amps[qubit] = data_qubit["amp"][index_best_err]

    return ResonatorAmplitudeResults(lowest_err, best_amps)


def _plot(data: ResonatorAmplitudeData, fit: ResonatorAmplitudeResults, qubit):
    """Plotting function for Optimization RO amplitude."""
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
                mode="lines+markers",
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
        xaxis_title="Readout Amplitude",
        yaxis_title="Probability Error",
    )

    figures.append(fig)

    return figures, fitting_report


resonator_amplitude = Routine(_acquisition, _fit, _plot)
"""Resonator Amplitude Routine  object."""
