from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence, Pulse, GaussianSquare, PulseType
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibocal.protocols.utils import HZ_TO_GHZ
from qibocal.auto.operation import Data, Parameters, Results, Routine

from .utils import STATES

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
    """ResonatorSpectroscopy runcard inputs."""

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

@dataclass
class CrossResonanceChevronFrequencyResults(Results):
    """Chevron wih Frequency Cross Resonance Calibration outputs."""


@dataclass
class CrossResonanceChevronFrequencyData(Data):
    """Data structure for Chevron wih Frequency."""

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
    """Data acquisition for Chevron wih Frequency."""

    data = CrossResonanceChevronFrequencyData()
    for pair in targets:
        target, control = pair
        for setup in STATES:      
            sequence = PulseSequence()
            target_drive_freq = platform.qubits[target].native_gates.RX.frequency
            rx_control = platform.create_RX_pulse(control, 0)
    
            next_start = 0
            if setup == STATES[1]:   
                next_start = rx_control.finish # add a RX control pulse if the setup is |X>
                sequence.add(rx_control)
            
            # Cross resonance pulse
            cr_pulse: Pulse = Pulse(start = next_start,
                                 duration = rx_control.duration,
                                 amplitude = rx_control.amplitude,
                                 frequency = target_drive_freq,
                                 relative_phase=0,
                                 channel=rx_control.channel,
                                 shape = GaussianSquare(0.9,5),
                                 type = PulseType.DRIVE,
                                 qubit = control,
                                 )
            
            if params.pulse_amplitude is not None:
                cr_pulse.amplitude = params.pulse_amplitude
            sequence.add(cr_pulse)

            # add readout pulses
            for qubit in pair:
                sequence.add(platform.create_qubit_readout_pulse(qubit, start=cr_pulse.finish))
                
            # create a duration sweeper for the pulse duration
            sweeper_duration = Sweeper(
                Parameter.duration,
                params.duration_range,
                pulses=[cr_pulse],
                type=SweeperType.ABSOLUTE,
            )

            # create a frequency sweeper for the pulse frequency
            sweeper_frequency = Sweeper(
                Parameter.frequency,
                params.frequency_range,
                pulses=[cr_pulse],
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
            for qubit in pair:
                data.register_qubit(
                    dtype=CrossResonanceChevronFrequencyType,
                    key=(qubit, target, control, setup),
                    prob=results[qubit].probability(state=1),
                    length=params.duration_range,
                    freq=cr_pulse.frequency + params.frequency_range,
                )

    # return the data
    return data


def _fit(
    data:CrossResonanceChevronFrequencyData,
) -> CrossResonanceChevronFrequencyResults:
    """Post-processing function for Chevron wih Frequency."""
    return CrossResonanceChevronFrequencyResults()


def _plot(
    data: CrossResonanceChevronFrequencyData,
    target: QubitPairId,
    fit: CrossResonanceChevronFrequencyResults,
):
    pair = target
    """Plotting function for Chevron wih Frequency and Duration."""
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
"""CrossResonanceChevronFrequency Routine object."""