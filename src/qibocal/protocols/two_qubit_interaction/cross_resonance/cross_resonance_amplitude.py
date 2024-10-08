from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibolab.pulses import PulseSequence
from qibolab.native import NativePulse
from qibolab.qubits import QubitId, QubitPairId
from qibolab.pulses import Pulse, Rectangular, PulseType, Gaussian

from qibocal.auto.operation import Data, Parameters, Results, Routine

CrossResonanceAmplitudeType = np.dtype(
    [
        ("prob", np.float64),
        ("amp", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for Cross Resonance Amplitude Gate Calibration."""

STATES = [0,1]

@dataclass
class CrossResonanceAmplitudeParameters(Parameters):
    """Cross Resonance Amplitude Gate Calibration runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    pulse_length: Optional[int] = None
    """CR pulse duration [ns]."""
    pulse_amplitude: Optional[float] = None
    """CR pulse amplitude [ns]."""

    @property
    def amplitude_factor_range(self):
        return np.arange(
            self.min_amp_factor, self.max_amp_factor, self.step_amp_factor
        )


@dataclass
class CrossResonanceResults(Results):
    """Cross Resonance Gate Calibration outputs."""


@dataclass
class CrossResonanceData(Data):
    """Data structure for rCross Resonance Gate Calibration."""

    data: dict[QubitId, npt.NDArray[CrossResonanceAmplitudeType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceAmplitudeParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceData:
    """Data acquisition for Cross Resonance Gate Calibration."""

    data = CrossResonanceData()
    


    for pair in targets:
        for ctr_setup  in STATES:
            for tgt_setup in STATES:
                target, control = pair
                tgt_native_rx:NativePulse = platform.qubits[target].native_gates.RX.pulse(start=0)
                ctr_native_rx:NativePulse = platform.qubits[control].native_gates.RX.pulse(start=0)

                if params.pulse_length is not None:
                    pulse_duration = params.pulse_length
                else:
                    pulse_duration = tgt_native_rx.duration

                sequence = PulseSequence()
                next_start = 0
                if tgt_setup == STATES[1]:
                    sequence.add(tgt_native_rx)
                    next_start = tgt_native_rx.finish

                if ctr_setup == STATES[1]:
                    sequence.add(ctr_native_rx)
                    next_start = max(ctr_native_rx.finish, next_start)
                
                cr_pulse: Pulse = Pulse(start=next_start,
                                duration=pulse_duration,
                                amplitude=ctr_native_rx.amplitude,
                                frequency=tgt_native_rx.frequency,   # control frequency
                                relative_phase=0,
                                shape=Gaussian(5),
                                qubit=control,
                                channel= ctr_native_rx.channel ,type=PulseType.DRIVE
                                )

                if params.pulse_amplitude is not None:
                    cr_pulse.amplitude = params.pulse_amplitude
                
                sequence.add(cr_pulse)

                for qubit in pair:
                    sequence.add(platform.create_qubit_readout_pulse(qubit, start=cr_pulse.finish))

                sweeper_amplitude = Sweeper(
                    parameter = Parameter.amplitude,
                    values = params.amplitude_factor_range,
                    pulses=[cr_pulse],
                    type=SweeperType.FACTOR,
                )

                results = platform.sweep(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        relaxation_time=params.relaxation_time,
                        acquisition_type=AcquisitionType.DISCRIMINATION,
                        averaging_mode=AveragingMode.SINGLESHOT,
                    ),
                    sweeper_amplitude,
                )

                # store the results
                for qubit in pair:
                    prob = results[qubit].probability(state=1)
                    data.register_qubit(
                        CrossResonanceAmplitudeType,
                        (qubit, target, control, tgt_setup, ctr_setup),
                        dict(
                            prob=prob.tolist(),
                            amp=cr_pulse.amplitude * params.amplitude_factor_range,
                            error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
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

            cr_parameters = getattr(i_data, 'amp')
            fig.add_trace(
                go.Scatter(
                    x=cr_parameters, y=i_data.prob, 
                    name= f"Target: |{STATES[0]}>, Control: |{ctr_setup}>",
                ),
            )
            fig.add_trace(
                go.Scatter(
                    x=cr_parameters, y=x_data.prob, 
                    name= f"Target: |{STATES[1]}>, Control: |{ctr_setup}>",
                    mode='lines', line={'dash': 'dash'}, 
                ),
            )
            fig.update_layout(
                title=f"Qubit {ro_qubit}",
                xaxis_title="CR Pulse Amplitude (a.u.)",
                yaxis_title="Excited State Probability",
            )

        figs.append(fig)

    return figs, ""

cross_resonance_amplitude = Routine(_acquisition, _fit, _plot)
"""CrossResonance Amplitude Routine object."""
