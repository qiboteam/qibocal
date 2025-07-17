from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, Delay, PulseSequence, Sweeper, Parameter

from qibocal import update
from qibocal.auto.operation import Data, Parameters, QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import table_dict, table_html, readout_frequency
from qibocal.update import replace

__all__ = ["resonator_frequency"]


@dataclass
class ResonatorFrequencyParameters(Parameters):
    """ResonatorFrequency runcard inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""

ResonatorFrequencyType = np.dtype(
    [
        ("error", np.float64),
        ("freq", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization RO frequency."""


@dataclass
class ResonatorFrequencyData(Data):
    """Data class for `resonator_frequnecy` optimization protocol."""

    data: dict[tuple, npt.NDArray[ResonatorFrequencyType]] = field(default_factory=dict)


@dataclass
class ResonatorFrequencyResults(Results):
    """Result class for `resonator_frequency` optimization protocol."""

    lowest_errors: dict[QubitId, list]
    """Lowest probability errors"""
    best_freq: dict[QubitId, list]
    """Frequency with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""


def _acquisition(
    params: ResonatorFrequencyParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> ResonatorFrequencyData:
    r"""
    Data acquisition for resoantor frequency optmization.
    This protocol sweeps the readout frequency performing a classification routine
    and evaluating the error probability at each step.

    Args:
        params (:class:`ResonatorFrequencyParameters`): input parameters
        platform (:class:`CalibrationPlatform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ResonatorFrequencyData`)

    """

    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )
    
    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    
    ro_pulses = {}
    for q in targets:
        natives = platform.natives.single_qubit[q]

        qd_channel, qd_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        sequence_1.append((qd_channel, qd_pulse))
        sequence_1.append((ro_channel, Delay(duration=qd_pulse.duration)))
        
        sequence_0.append((ro_channel, ro_pulse))
        sequence_1.append((ro_channel, ro_pulse))
        ro_pulses[q] = ro_pulse

    ro_channel, ro_pulse = natives.MZ()[0]
    
    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform)
            + delta_frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    results = {}
    results[0] = platform.execute(
            [sequence_0],
            [sweepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        )

    results[1] = platform.execute(
            [sequence_1],
            [sweepers],
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
            acquisition_type=AcquisitionType.INTEGRATION,
        )
    data:ResonatorFrequencyData = ResonatorFrequencyData()

    for q in targets:
        frequency = readout_frequency(q, platform)+ delta_frequency_range

        for k, freq in enumerate(frequency):
            
            # iq_values = []
            # for state in results.keys():
            #     result = results[state][ro_pulses[q].id]
            #     iq_values.append(result[:,k])

            result0 = results[0][ro_pulses[q].id][:,k]
            result1 = results[1][ro_pulses[q].id][:,k]
            iq_values = np.concatenate((result0, result1))
            
            nshots = params.nshots
            states = [0] * nshots + [1] * nshots
            model = QubitFit()
            model.fit(iq_values, np.array(states))
            error = model.probability_error

            data.register_qubit(
                ResonatorFrequencyType,
                (q),
                dict(
                    error=np.array([error]),
                    freq=np.array([freq]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )
          
    return data


def _fit(data: ResonatorFrequencyData) -> ResonatorFrequencyResults:
    qubits = data.qubits
    best_freqs = {}
    best_angle = {}
    best_threshold = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_best_err = np.argmin(data_qubit["error"])
        lowest_err[qubit] = data_qubit["error"][index_best_err]
        best_freqs[qubit] = data_qubit["freq"][index_best_err]
        best_angle[qubit] = data_qubit["angle"][index_best_err]
        best_threshold[qubit] = data_qubit["threshold"][index_best_err]

    return ResonatorFrequencyResults(lowest_err, best_freqs, best_angle, best_threshold)


def _plot(
    data: ResonatorFrequencyData, fit: ResonatorFrequencyResults, target: QubitId
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
                x=data[target]["freq"],
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
                "Best Readout Frequency [a.u.]",
                np.round(fit.best_freq[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Readout Frequency [a.u.]",
        yaxis_title="Probability Error",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(
    results: ResonatorFrequencyResults, platform: CalibrationPlatform, target: QubitId
):
    update.readout_frequency(results.best_freq[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


resonator_frequency = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Frequency Routine  object."""
