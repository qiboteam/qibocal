from dataclasses import dataclass, field
from os import error

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, ExecutionParameters, AveragingMode
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.fitting.classifier.qubit_fit import QubitFit
from qibocal.protocols.utils import table_dict, table_html


@dataclass
class ReadoutDurationParameters(Parameters):
    """Readout duration runcard inputs."""

    duration_step: int = 4
    """duration step to be probed."""
    duration_start: int = 100
    """duration start."""
    duration_stop: float = 600
    """duration stop value"""
    
    @property
    def duration_range(self) -> npt.NDArray[np.float64]:
        """Return the range of duration values to be tested."""
        return np.arange(
            self.duration_start, self.duration_stop + self.duration_step, self.duration_step
        ).astype(np.float64)


ReadoutDurationType = np.dtype(
    [
        ("duration", np.float64),
        ("assignment_fidelity", np.float64),
        ("angle", np.float64),
        ("threshold", np.float64),
    ]
)
"""Custom dtype for Optimization of MZ duration."""



@dataclass
class ReadoutDurationData(Data):
    """Data class for `readout_duration` protocol."""

    data: dict[tuple, npt.NDArray[ReadoutDurationType]] = field(default_factory=dict)


@dataclass
class ReadoutDurationResults(Results):
    """Result class for `readout_duration` protocol."""

    lowest_errors: dict[QubitId, list]
    """Lowest probability errors"""
    best_duration: dict[QubitId, list]
    """duration with lowest error"""
    best_angle: dict[QubitId, float]
    """IQ angle that gives lower error."""
    best_threshold: dict[QubitId, float]
    """Thershold that gives lower error."""


def _acquisition(
    params: ReadoutDurationParameters,
    platform: Platform,
    targets: list[QubitId],
) -> ReadoutDurationData:
    r"""
    Data acquisition for readout duration optmization.
    This protocol sweeps the readout acquisition duration performing a classification routine
    and evaluating the error probability at each step.
    
    May only work with QBLOX

    Args:
        params (:class:`ReadoutDurationParameters`): input parameters
        platform (:class:`Platform`): Qibolab's platform
        targets (list): list of QubitIds to be characterized

    Returns:
        data (:class:`ReadoutDurationData`)
    """
    from qibolab.channels import Channel
    from qibolab.qubits import Qubit
    
    data = ReadoutDurationData()
    ro_pulses = {}
    qd_pulses = {}
    sweep_results = {}

    # Run the experiment for each qubit
    for qubit in targets:
        sequence_0 = PulseSequence()
        sequence_1 = PulseSequence()

        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence_0.add(ro_pulses[qubit])
        sequence_1.add(qd_pulses[qubit])
        sequence_1.add(ro_pulses[qubit])

        # Create a duration sweeper for all qubits
        sweeper_duration = Sweeper(
            parameter = Parameter.duration,
            values = params.duration_range,
            pulses = [ro_pulses[qubit]],
            type =SweeperType.ABSOLUTE,
        )
        sweep_results[(qubit,0)] = platform.sweep(
            sequence_0,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            ),
            sweeper_duration,
        )
        
        sweep_results[(qubit,1)] = platform.sweep(
            sequence_1,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
            ),
            sweeper_duration,
        )
    
    # Retrieve the results for every qubit, preprocess and save
    for qubit in targets:
        ro_pulse = ro_pulses[qubit]
        for k, duration in enumerate(params.duration_range):
            i_values = []
            q_values = []
            states = []
            for i, results in enumerate([sweep_results[(qubit,0)], sweep_results[(qubit,1)]]):
                result = results[ro_pulse.serial]
                i_values.extend(result.voltage_i[:, k])
                q_values.extend(result.voltage_q[:, k])
                states.extend([i] * len(result.voltage_i[:, k]))

            model = QubitFit()
            model.fit(np.stack((i_values, q_values), axis=-1), np.array(states))
            data.register_qubit(
                ReadoutDurationType,
                (qubit),
                dict(
                    duration=np.array([(duration)]),
                    assignment_fidelity=np.array([model.assignment_fidelity]),
                    angle=np.array([model.angle]),
                    threshold=np.array([model.threshold]),
                ),
            )

    return data


def _fit(data: ReadoutDurationData) -> ReadoutDurationResults:
    qubits = data.qubits
    best_durations = {}
    best_angle = {}
    best_threshold = {}
    lowest_err = {}
    for qubit in qubits:
        data_qubit = data[qubit]
        index_af = np.argmax(data_qubit["assignment_fidelity"])
        lowest_err[qubit] = 1-data_qubit["assignment_fidelity"][index_af]
        best_durations[qubit] = data_qubit["duration"][index_af]
        best_angle[qubit] = data_qubit["angle"][index_af]
        best_threshold[qubit] = data_qubit["threshold"][index_af]

    return ReadoutDurationResults(lowest_err, best_durations, best_angle, best_threshold)


def _plot(
    data: ReadoutDurationData, fit: ReadoutDurationResults, target: QubitId
):
    """Plotting function for Optimization MZ duration."""
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
                x=data[target]["duration"],
                y=data[target]["assignment_fidelity"],
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
                "Best MZ pulse duration [a.u.]",
                np.round(fit.best_duration[target], 4),
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="MZ duration [a.u.]",
        yaxis_title="Assignement Fidelity",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: ReadoutDurationResults, platform: Platform, target: QubitId):
    update.readout_mz_duration(results.best_duration[target], platform, target)
    update.iq_angle(results.best_angle[target], platform, target)
    update.threshold(results.best_threshold[target], platform, target)


mz_duration_sweeper = Routine(_acquisition, _fit, _plot, _update)
"""Resonator Readout MZ duration Routine  object."""
