from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.two_qubit_interaction.cross_resonance.cross_resonance import CrossResonanceParameters

CrossResonanceType = np.dtype(
    [
        ("magnitude", np.float64),
        ("phase", np.float64),
        ("length", np.int64),
    ]
)
"""Custom dtype for Cross Resonance Gate Calibration using Sequences."""


@dataclass
class CrossResonanceSeqParameters(CrossResonanceParameters):
    """Cross Resonance Gate Calibration using Sequences runcard inputs."""


@dataclass
class CrossResonanceSeqResults(Results):
    """Cross Resonance Gate Calibration using Sequences outputs."""


@dataclass
class CrossResonanceSeqData(Data):
    """Data structure for Cross Resonance Gate Calibration using Sequences."""
    data: dict[QubitId, npt.NDArray[CrossResonanceType]] = field(default_factory=dict)
    """Raw data acquired."""

def _acquisition(
    params: CrossResonanceSeqParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceSeqData:
    """Data acquisition for Cross Resonance Gate Calibration using Sequences."""

    data = CrossResonanceSeqData()
    
    for pair in targets:
        for setup in ["I", "X"]:

            # sweep the parameter
            for duration in params.duration_range:
                target, control = pair
                sequence = PulseSequence()
                control_drive_freq = platform.qubits[control].native_gates.RX.frequency

                if setup == "X":
                    rx_control = platform.create_RX_pulse(control, 0)
                    cr_pulse = platform.create_RX_pulse(target, start= rx_control.finish)
                    sequence.add(rx_control)
                else:
                    cr_pulse = platform.create_RX_pulse(target, 0)

                cr_pulse.frequency = control_drive_freq
                cr_pulse.duration = duration
                if params.pulse_amplitude is not None:
                    cr_pulse.amplitude = params.pulse_amplitude
                sequence.add(cr_pulse)

                ro_pulse = platform.create_qubit_readout_pulse(target, start=cr_pulse.finish)
                sequence.add(ro_pulse)
                    
                results = platform.execute_pulse_sequence(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        relaxation_time=params.relaxation_time,
                        #acquisition_type=AcquisitionType.DISCRIMINATION,
                        acquisition_type=AcquisitionType.INTEGRATION,
                        #averaging_mode=AveragingMode.SINGLESHOT,
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                )

                # Store Results
                #result = results[target].probability(state=0)
                result = results[target]
                print(result)
                data.register_qubit(
                    CrossResonanceType,
                    (target, control, setup),
                    dict(
                        #prob=np.array([result]),
                        length=np.array([duration]),
                        magnitude=np.array([result.magnitude]),
                        phase=np.array([result.phase]),
                    ),
                )
    return data


def _fit(
    data: CrossResonanceSeqData,
) -> CrossResonanceSeqResults:
    """Post-processing function for Cross Resonance Gate Calibration using Sequences."""
    return CrossResonanceSeqResults()


def _plot(data: CrossResonanceSeqData, target: QubitPairId, fit: CrossResonanceSeqResults):
    """Plotting function for Cross Resonance Gate Calibration using Sequences."""

    control_idle_data = data.data[target[0], target[1], "I"]
    control_excited_data = data.data[target[0], target[1], "X"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=control_idle_data.length,
            y=control_idle_data.magnitude, 
            name="Control at 0"
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=control_excited_data.length,
            y=control_excited_data.magnitude,
            name="Control at 1",
        ),
    )

    fig.update_layout(
            showlegend=True,
            xaxis_title="Gate duration (ns)",
            yaxis_title="Signal [a.u.]"
    )

    return [fig], ""


cross_resonance_sequences = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
