from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibolab.native import NativePulse
from qibolab.qubits import QubitId, QubitPairId
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses import Gaussian, Drag, Rectangular, GaussianSquare

from .utils import STATES

CrossResonanceType = np.dtype(
    [
        ("prob", np.float64),
        ("length", np.int64),
    ]
)
"""Custom dtype for Cross Resonance Gate Calibration."""

@dataclass
class CrossResonanceParameters(Parameters):
    """Cross Resonance Gate Calibration runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    pulse_amplitude: Optional[float] = None
    """CR pulse amplitude [ns]."""
    shape: Optional[str] = "Rectangular()"
    """CR pulse shape."""
    
    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )
    """Pulse duration range."""
    @property
    def pulse_shape(self):
        return eval(self.shape)
    """Cross Resonance Pulse shape."""


@dataclass
class CrossResonanceResults(Results):
    """Cross Resonance Gate Calibration outputs."""


@dataclass
class CrossResonanceData(Data):
    """Data structure for Cross Resonance Gate Calibration."""

    targets: list[QubitPairId] = field(default_factory=list)
    """Targets for the Cross Resonance Gate Calibration stored as pair of [target, control]."""

    data: dict[QubitId, npt.NDArray[CrossResonanceType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceData:
    """Data acquisition for Cross Resonance Gate Calibration."""

    data = CrossResonanceData(targets=targets)

    for pair in targets:
        for tgt_setup in STATES:
            for ctr_setup in STATES:
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
                                duration=params.pulse_duration_start,
                                amplitude=ctr_native_rx.amplitude,
                                frequency=tgt_native_rx.frequency,   # control frequency
                                relative_phase=0,
                                shape=params.pulse_shape,
                                qubit=control,
                                channel= ctr_native_rx.channel ,type=PulseType.DRIVE
                                )

                if params.pulse_amplitude is not None:
                    cr_pulse.amplitude = params.pulse_amplitude
                
                sequence.add(cr_pulse)

                for qubit in pair:
                    sequence.add(platform.create_qubit_readout_pulse(qubit, start=cr_pulse.finish))

                sweeper_duration = Sweeper(
                    parameter = Parameter.duration,
                    values = params.duration_range,
                    pulses=[cr_pulse],
                    type=SweeperType.ABSOLUTE,
                )

                results = platform.sweep(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        relaxation_time=params.relaxation_time,
                        acquisition_type=AcquisitionType.DISCRIMINATION,
                        averaging_mode=AveragingMode.SINGLESHOT,
                    ),
                    sweeper_duration,
                )

                # store the results
                for qubit in pair:
                    probability = results[qubit].probability(state=1)
                    data.register_qubit(
                        CrossResonanceType,
                        (qubit, target, control, tgt_setup, ctr_setup),
                        dict(
                            prob=probability,
                            length=params.duration_range,
                        ),
                    )

    return data


def _fit(
    data: CrossResonanceData,
) -> CrossResonanceResults:
    """Post-processing function for Cross Resonance Gate Calibration."""
    return CrossResonanceResults()


def _plot(data: CrossResonanceData, target: QubitPairId, fit: CrossResonanceResults):
    """Plotting function for Cross Resonance Gate Calibration."""

    figs = []
    for ro_qubit in target:
        fig = go.Figure()
        for ctr_setup in STATES:
            i_data = data.data[ro_qubit, target[0], target[1], STATES[0], ctr_setup]
            x_data = data.data[ro_qubit, target[0], target[1], STATES[1], ctr_setup]
            fig.add_trace(
                go.Scatter(
                    x=i_data.length, y=i_data.prob, 
                    name= f"Target: |{STATES[0]}>, Control: |{ctr_setup}>",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=x_data.length, y=x_data.prob, 
                    name= f"Target: |{STATES[1]}>, Control: |{ctr_setup}>",
                    mode='lines', line={'dash': 'dash'}, 
                ),
            )
            fig.update_layout(
                title=f"Qubit {ro_qubit}",
                xaxis_title="CR Pulse Length (ns)",
                yaxis_title="Excited State Probability",
            )

        figs.append(fig)

    return figs, ""

cross_resonance_length = Routine(_acquisition, _fit, _plot)
"""CrossResonance Length Routine object."""
