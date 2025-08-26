from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, Delay, PulseSequence

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import table_dict, table_html
from qibocal.update import replace

__all__ = ["resonator_amplitude"]


@dataclass
class ResonatorAmplitudeParameters(Parameters):
    """ResonatorAmplitude runcard inputs."""

    amplitude_step: float
    """Amplituude step to be probed."""
    amplitude_start: float = 0.0
    """Amplitude start."""
    amplitude_stop: float = 1.0
    """Amplitude stop value"""
    error_threshold: float = 0.003
    """Probability error threshold to stop the best amplitude search"""


ResonatorAmplitudeType = np.dtype(
    [
        ("error", np.float64),
        ("amp", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization RO amplitude."""


@dataclass
class ResonatorAmplitudeData(Data):
    """Data class for `resoantor_amplitude` protocol."""

    data: dict[tuple, npt.NDArray[ResonatorAmplitudeType]] = field(default_factory=dict)


@dataclass
class ResonatorAmplitudeResults(Results):
    """Result class for `resonator_amplitude` protocol."""

    lowest_errors: dict[QubitId, list]
    """Lowest probability errors"""
    best_amp: dict[QubitId, list]
    """Amplitude with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""


def _acquisition(
    params: ResonatorAmplitudeParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorAmplitudeData:
    r"""
    Data acquisition for resoantor amplitude optmization.
    This protocol sweeps the readout amplitude performing a classification routine
    and evaluating the error probability at each step. The sweep will be interrupted
    if the probability error is less than the `error_threshold`.

    Args:
        params (:class:`ResonatorAmplitudeParameters`): input parameters
        platform (:class:`CalibrationPlatform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ResonatorAmplitudeData`)
    """

    data = ResonatorAmplitudeData()
    for qubit in targets:
        error = 1
        natives = platform.natives.single_qubit[qubit]

        ro_channel, ro_pulse = natives.MZ()[0]
        new_amp = params.amplitude_start
        while error > params.error_threshold and new_amp <= params.amplitude_stop:
            new_ro = replace(ro_pulse, amplitude=new_amp)
            sequence_0 = PulseSequence()
            sequence_1 = PulseSequence()

            qd_channel, qd_pulse = natives.RX()[0]

            sequence_1.append((qd_channel, qd_pulse))
            sequence_1.append((ro_channel, Delay(duration=qd_pulse.duration)))
            sequence_1.append((ro_channel, new_ro))

            sequence_0.append((ro_channel, new_ro))

            state0_results = platform.execute(
                [sequence_0],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            )

            state1_results = platform.execute(
                [sequence_1],
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            )
            result0 = state0_results[new_ro.id]
            result1 = state1_results[new_ro.id]

            # TODO: move QubitFit() and analysis in _fit()
            iq_values = np.concatenate((result0, result1))
            nshots = params.nshots
            states = [0] * nshots + [1] * nshots
            model = QubitFit()
            model.fit(iq_values, np.array(states))
            error = model.probability_error
            data.register_qubit(
                ResonatorAmplitudeType,
                (qubit),
                dict(
                    amp=np.array([new_amp]),
                    error=np.array([error]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )
            new_amp += params.amplitude_step
    return data


def _fit(data: ResonatorAmplitudeData) -> ResonatorAmplitudeResults:
    qubits = data.qubits
    best_amps = {}
    best_angle = {}
    best_threshold = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["error"])
        lowest_err[qubit] = data_qubit["error"][index_best_err]
        best_amps[qubit] = data_qubit["amp"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return ResonatorAmplitudeResults(lowest_err, best_amps, best_angle, best_threshold)


def _plot(
    data: ResonatorAmplitudeData, fit: ResonatorAmplitudeResults, target: QubitId
):
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
                x=data[target]["amp"],
                y=data[target]["error"],
                opacity=opacity,
                showlegend=True,
                mode="lines+markers",
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                target,
                "Best Readout Amplitude [a.u.]",
                np.round(fit.best_amp[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Amplitude [a.u.]",
        yaxis_title="Probability Error",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ResonatorAmplitudeResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_amplitude(results.best_amp[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_amplitude = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Amplitude Routine  object."""
