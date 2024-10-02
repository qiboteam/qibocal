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
STATES = [0,1]
"""Custom dtype for Cross Resonance Gate Calibration using Sequences."""


@dataclass
class CrossResonanceSeqParameters(CrossResonanceParameters):
    """Cross Resonance Gate Calibration using Sequences runcard inputs."""


@dataclass
class CrossResonanceSeqResults(Results):
    """Cross Resonance Gate Calibration using Sequences outputs."""


@dataclass
class CrossResonanceSeqData(Data):
    """Data structure for Cross Resonance Gate Calibration using Sequences.
    targets: [target, control]
    0(I):
        Q_C: Pulse(omega_T, t)  - MZ
        Q_T: wait               - MZ
    1(X):
        Q_C: RX   - Pulse(omega_T, t)  - MZ
        Q_T:      - wait               - MZ
    """
    data: dict[QubitId, npt.NDArray[CrossResonanceType]] = field(default_factory=dict)
    """Raw data acquired."""

def _acquisition(
    params: CrossResonanceSeqParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceSeqData:
    """Data acquisition for Cross Resonance Gate Calibration using Sequences."""
    from qibolab.pulses import Pulse, Rectangular, PulseType, Gaussian
    from qibolab.native import NativePulse

    data = CrossResonanceSeqData()
    
    
    for pair in targets:
        for tgt_setup in STATES:
            for ctr_setup in STATES:
            # sweep the parameter
                for duration in params.duration_range:
                    target, control = pair
                    tgt_native_rx:NativePulse = platform.qubits[target].native_gates.RX.pulse(start=0)
                    ctr_native_rx:NativePulse = platform.qubits[control].native_gates.RX.pulse(start=0)

                    sequence = PulseSequence()
                    next_start = 0

                    if tgt_setup == STATES[1]:
                        sequence.add(tgt_native_rx)
                        next_start = tgt_native_rx.finish

                    if ctr_setup == STATES[1]:
                        sequence.add(ctr_native_rx)
                        next_start = max(ctr_native_rx.finish, next_start)
                    
                    cr_pulse: Pulse = Pulse(start=next_start,
                                    duration=duration,
                                    amplitude=ctr_native_rx.amplitude,
                                    frequency=tgt_native_rx.frequency,   # control frequency
                                    relative_phase=0,
                                    shape=Rectangular(),
                                    qubit=control,
                                    channel= ctr_native_rx.channel ,type=PulseType.DRIVE
                                    )

                    if params.pulse_amplitude is not None:
                        cr_pulse.amplitude = params.pulse_amplitude
                    
                    sequence.add(cr_pulse)

                    for qubit in pair:
                        sequence.add(platform.create_qubit_readout_pulse(qubit, start=cr_pulse.finish))
                            
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
                    for qubit in pair:
                        mag = results[qubit].magnitude
                        phi = results[qubit].phase
                        data.register_qubit(
                            CrossResonanceType,
                            (qubit, target, control, tgt_setup, ctr_setup),
                            dict(
                                magnitude=[mag],
                                phase = [phi],
                                length=[duration],
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
    figs = []
    pair_labels = ["Target", "Control"]
    for ro_qubit, label in zip(target, pair_labels):
        fig = go.Figure()
        for ctr_setup in STATES:
            i_data = data.data[ro_qubit, target[0], target[1], STATES[0], ctr_setup]
            x_data = data.data[ro_qubit, target[0], target[1], STATES[1], ctr_setup]
            fig.add_trace(
                go.Scatter(
                    x=i_data.length, y=i_data.magnitude, 
                    name= f"Target: |{STATES[0]}>, Control: |{ctr_setup}>",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=x_data.length, y=x_data.magnitude, 
                    name= f"Target: |{STATES[1]}>, Control: |{ctr_setup}>",
                    mode='lines', line={'dash': 'dash'}, 
                ),
            )
            fig.update_layout(
                title=f"Qubit {ro_qubit}: {label}",
                xaxis_title="CR Pulse Length (ns)",
                yaxis_title="Excited State Probability",
            )

        figs.append(fig)

    return figs, ""


cross_resonance_sequences = Routine(_acquisition, _fit, _plot)
"""CrossResonance Routine object."""
