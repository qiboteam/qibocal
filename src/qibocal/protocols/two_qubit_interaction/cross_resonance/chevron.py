from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal.auto.operation import Data, Parameters, Results, Routine

CrossResonanceChevronType = np.dtype(
    [
        ("prob", np.float64),
        ("length", np.int64),
        ("amp", np.float64),
    ]
)
"""Custom dtype for cross resonance chevron."""

STATES = ["I", "X"]

@dataclass
class CrossResonanceChevronParameters(Parameters):
    """Cross Resonance Chevron runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    amplitude_min_factor: float
    """Amplitude minimum."""
    amplitude_max_factor: float
    """Amplitude maximum."""
    amplitude_step_factor: float
    """Amplitude step."""

    pulse_amplitude: Optional[float] = None

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )

    @property
    def amplitude_factor_range(self):
        return np.arange(
            self.amplitude_min_factor,
            self.amplitude_max_factor,
            self.amplitude_step_factor,
        )


@dataclass
class CrossResonanceChevronResults(Results):
    """Cross Resonance Chevron outputs."""


@dataclass
class CrossResonanceChevronData(Data):
    """Data structure for cross resonance chevron."""

    data: dict[QubitId, npt.NDArray[CrossResonanceChevronType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, dtype, key, prob, length, amp):
        """Store output for single qubit."""
        size = len(length) * len(amp)
        amplitude, duration = np.meshgrid(amp, length)
        ar = np.empty(size, dtype=dtype)
        ar["length"] = duration.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob"] = prob.ravel()
        self.data[key] = np.rec.array(ar)


def _acquisition(
    params: CrossResonanceChevronParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> CrossResonanceChevronData:
    """Data acquisition for Cross Resonance Chevron."""

    data = CrossResonanceChevronData()

    for pair in targets:
        for setup in STATES:
            target, control = pair
            sequence = PulseSequence()
            target_drive_freq = platform.qubits[target].native_gates.RX.frequency

            if setup == STATES[1]:
                rx_control = platform.create_RX_pulse(control, 0)
                pulse = platform.create_RX_pulse(control, rx_control.finish)
                sequence.add(rx_control)
            else:
                pulse = platform.create_RX_pulse(control, 0)

            pulse.frequency = target_drive_freq
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

            sweeper_amplitude = Sweeper(
                Parameter.amplitude,
                pulse.amplitude * params.amplitude_factor_range,
                pulses=[pulse],
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
                sweeper_duration,
                sweeper_amplitude,
            )

            # store the results
            prob1 = results[target].probability(state=1)
            data.register_qubit(
                dtype=CrossResonanceChevronType,
                key=(target, control, setup),
                prob=prob1,
                length=params.duration_range,
                amp=pulse.amplitude * params.amplitude_factor_range,
            )
    # finally, save the remaining data
    return data


def _fit(
    data: CrossResonanceChevronData,
) -> CrossResonanceChevronResults:
    """Post-processing function for ResonatorSpectroscopy."""
    return CrossResonanceChevronResults()


def _plot(
    data: CrossResonanceChevronData,
    target: QubitPairId,
    fit: CrossResonanceChevronResults,
):
    pair = target
    figs = []

    """Plotting function for Cross Resonance Chevron ."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Control Q{pair[1]} = |{STATES[0]}>",
            f"Control Q{pair[1]} = |{STATES[1]}>",
        ),
    )

    for i, setup in enumerate(STATES):
        qubit_data = data.data[target[0], target[1], setup]
        fig.add_trace(
            go.Heatmap(
                x=qubit_data.length,
                y=qubit_data.amp,
                z=qubit_data.prob,
                name=f"Control at {setup}",
                coloraxis="coloraxis"
            ),
            row=1,
            col=i+1,
        )
    fig.update_layout(coloraxis={'colorscale':'Plasma'})
    
    figs.append(fig)
    return figs, ""


cross_resonance_chevron = Routine(_acquisition, _fit, _plot)
"""CrossResonanceChevron Routine object."""
