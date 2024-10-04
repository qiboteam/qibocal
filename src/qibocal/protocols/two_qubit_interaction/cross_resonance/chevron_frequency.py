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
from qibocal.protocols.utils import HZ_TO_GHZ
from qibocal.auto.operation import Data, Parameters, Results, Routine

CrossResonanceChevronFrequencyType = np.dtype(
    [
        ("length", np.float64),
        ("freq", np.int64),
        ("prob", np.float64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class CrossResonanceChevronFrequencyParameters(Parameters):
    """Cross resonance chevron frequency runcard inputs."""

    pulse_duration_start: float
    """Initial pi pulse duration [ns]."""
    pulse_duration_end: float
    """Final pi pulse duration [ns]."""
    pulse_duration_step: float
    """Step pi pulse duration [ns]."""
    freq_width: int
    """Frequency range."""
    freq_step: int
    """Frequency step size."""

    pulse_amplitude: Optional[float] = None

    @property
    def duration_range(self):
        return np.arange(
            self.pulse_duration_start, self.pulse_duration_end, self.pulse_duration_step
        )
    
    @property
    def frequency_range(self):
        return np.arange(
            -self.freq_width/2,
            self.freq_width/2,
            self.freq_step,
        )
STATES = [0, 1]

@dataclass
class CrossResonanceChevronFrequencyResults(Results):
    """Chevron wih frequency Cross pulse calibration outputs."""


@dataclass
class CrossResonanceChevronFrequencyData(Data):
    """Data structure for chevron wih frequency."""

    data: dict[QubitId, npt.NDArray[CrossResonanceChevronFrequencyType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, dtype, key, prob, freq, length):
        """Store output for single qubit."""
        size = len(freq) * len(length)
        frequency, duration = np.meshgrid(freq, length)
        ar = np.empty(size, dtype=dtype)
        ar["freq"] = frequency.ravel()
        ar["length"] = duration.ravel()
        ar["prob"] = prob.ravel()
        self.data[key] = np.rec.array(ar)


def _acquisition(
    params: CrossResonanceChevronFrequencyParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> CrossResonanceChevronFrequencyData:
    """Data acquisition for chevron wih frequency."""

    data = CrossResonanceChevronFrequencyData()
    for pair in targets:
        target, control = pair
        for setup in STATES:      
            sequence = PulseSequence()
            target_drive_freq = platform.qubits[target].native_gates.RX.frequency
            
            # add a RX control pulse if the setup is |1>
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

            # add readout pulses
            for qubit in pair:
                sequence.add(platform.create_qubit_readout_pulse(qubit, start=pulse.finish))
                
            # create a duration sweeper for the pulse duration
            sweeper_duration = Sweeper(
                Parameter.duration,
                params.duration_range,
                pulses=[pulse],
                type=SweeperType.ABSOLUTE,
            )

            # create a frequency sweeper for the pulse frequency
            sweeper_frequency = Sweeper(
                Parameter.frequency,
                params.frequency_range,
                pulses=[pulse],
                type=SweeperType.OFFSET,
            )

            # run the sweep 
            results = platform.sweep(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=AveragingMode.SINGLESHOT,
                ),
                sweeper_duration,
                sweeper_frequency,
            )

            # store the results for each qubit in the pair in the data object
            # NOTE: Change this to use standard qibocal>auto>operation>Data>register_qubit
            for qubit in pair:
                data.register_qubit(
                    dtype=CrossResonanceChevronFrequencyType,
                    key=(qubit, target, control, setup),
                    prob=results[qubit].probability(state=1),
                    length=params.duration_range,
                    freq=pulse.frequency + params.frequency_range,
                )

    # return the data
    return data


def _fit(
    data:CrossResonanceChevronFrequencyData,
) -> CrossResonanceChevronFrequencyResults:
    """Post-processing function for chevron with frequency."""
    return CrossResonanceChevronFrequencyResults()


def _plot(
    data: CrossResonanceChevronFrequencyData,
    target: QubitPairId,
    fit: CrossResonanceChevronFrequencyResults,
):
    pair = target
    """Plotting function for chevron wih frequency and duration."""
    figs = []
    for qubit in pair:
        fig = make_subplots(
                rows=1,
                cols=2,
                subplot_titles=(
                    f"Q{qubit} , Control |{STATES[0]}>",
                    f"Q{qubit} , Control |{STATES[1]}>",
                ),
            )
        for i, setup in enumerate(STATES):           
            qubit_data = data.data[qubit, pair[0], pair[1], setup]
            fig.add_trace(
                go.Heatmap(
                    x=qubit_data.length,
                    y=qubit_data.freq * HZ_TO_GHZ,
                    z=qubit_data.prob,
                    name=f"Control at |{setup}>",
                    coloraxis="coloraxis"
                ),
                row=1,
                col=i+1,
            )
            fig.update_xaxes(title_text="Pulse Duration [ns]", row=1, col=i+1)
            fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=i+1)
        fig.update_layout(coloraxis={'colorscale':'Plasma'})
        figs.append(fig)

            
    return figs, ""

cross_resonance_chevron_frequency = Routine(_acquisition, _fit, _plot)
"""Cross resonance chevron with frequency routine object."""