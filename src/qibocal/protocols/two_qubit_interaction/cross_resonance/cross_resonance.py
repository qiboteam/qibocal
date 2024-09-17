from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Results, Routine

CrossResonanceType = np.dtype(
    [
        ("prob", np.float64),
        ("length", np.int64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class CrossResonanceParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""

    pulse_amplitude: Optional[float] = None

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )


@dataclass
class CrossResonanceResults(Results):
    """ResonatorSpectroscopy outputs."""


@dataclass
class CrossResonanceData(Data):
    """Data structure for resonator spectroscopy with attenuation."""

    data: dict[QubitId, npt.NDArray[CrossResonanceType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceData:
    """Data acquisition for resonator spectroscopy."""

    data = CrossResonanceData()

    for pair in targets:
        for setup in ["I", "X"]:
            target, control = pair
            sequence = PulseSequence()
            control_drive_freq = platform.qubits[control].native_gates.RX.frequency

            if setup == "X":
                rx_control = platform.create_RX_pulse(control, 0)
                pulse = platform.create_RX_pulse(target, rx_control.finish)
                sequence.add(rx_control)
            else:
                pulse = platform.create_RX_pulse(target, 0)

            pulse.frequency = control_drive_freq
            if params.pulse_amplitude is not None:
                pulse.amplitude = params.pulse_amplitude
            sequence.add(pulse)
            sequence.add(
                platform.create_qubit_readout_pulse(target, start=pulse.finish)
            )

            sweeper_duration = Sweeper(
                Parameter.duration,
                params.duration_range,
                pulses=[pulse],
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
            prob1 = results[target].probability(state=0)
            data.register_qubit(
                CrossResonanceType,
                (target, control, setup),
                dict(
                    prob=prob1,
                    length=params.duration_range,
                ),
            )
    # finally, save the remaining data
    return data


def _fit(
    data: CrossResonanceData,
) -> CrossResonanceResults:
    """Post-processing function for ResonatorSpectroscopy."""
    return CrossResonanceResults()


def _plot(data: CrossResonanceData, target: QubitPairId, fit: CrossResonanceResults):
    """Plotting function for ResonatorSpectroscopy."""
    # TODO: share colorbar
    control_idle_data = data.data[target[0], target[1], "I"]
    control_excited_data = data.data[target[0], target[1], "X"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=control_idle_data.length, y=control_idle_data.prob, name="Control at 0"
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=control_excited_data.length,
            y=control_excited_data.prob,
            name="Control at 1",
        ),
    )

    return [fig], ""

cross_resonance = Routine(_acquisition, _fit, _plot)
"""CrossResonance Sequences Routine object."""
