from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibocal.protocols.utils import HZ_TO_GHZ
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses import Gaussian, Drag, Rectangular, GaussianSquare

from .utils import STATES

CrossResonanceChevronFrequencyType = np.dtype(
    [
        ("amp", np.float64),
        ("freq", np.int64),
        ("prob", np.float64),
    ]
)
"""Custom dtype for resonator spectroscopy."""


@dataclass
class CrossResonanceChevronFrequencyParameters(Parameters):
    """ResonatorSpectroscopy runcard inputs."""

    min_amp_factor: float
    """Initial CR pulse amplitude [a.u.]."""
    max_amp_factor: float
    """Final CR pulse amplitude [a.u.]."""
    step_amp_factor: float
    """Step CR pulse amplitude [a.u.]."""
    freq_width: int
    """Frequency range."""
    freq_step: int
    """Frequency step size."""
    pulse_amplitude: Optional[float] = None
    """Maximum pulse amplitude [a.u.]."""
    pulse_duration: Optional[int] = None
    """Pulse duration [ns]."""
    shape: Optional[str] = "Rectangular()"
    """CR pulse shape parameters."""
    @property
    def pulse_shape(self):
        return eval(self.shape)
    """Cross Resonance Pulse shape."""
    
    @property
    def amplitude_factor(self):
        return np.arange(
            self.min_amp_factor, self.max_amp_factor, self.step_amp_factor
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

    def register_qubit(self, dtype, data_key, data:dict):
        """Store output for single qubit."""

        prob = data["prob"]
        freq = data["freq"]
        amp = data["amp"]

        size = len(freq) * len(amp)
        frequency, amplitude = np.meshgrid(freq, amp)
        ar = np.empty(size, dtype=dtype)
        ar["freq"] = frequency.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob"] = prob.ravel()
        self.data[data_key] = np.rec.array(ar)


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
            
            # add a RX control pulse if the setup is |1>
            next_start=0
            if setup == STATES[1]:
                next_start = rx_control.finish
                sequence.add(rx_control)

            cr_pulse: Pulse = Pulse(start = next_start,
                                 duration = rx_control.duration,
                                 amplitude = rx_control.amplitude,
                                 frequency = target_drive_freq,
                                 relative_phase=0,
                                 channel=rx_control.channel,
                                 shape = params.pulse_shape,
                                 type = PulseType.DRIVE,
                                 qubit = control,
                                 )

            cr_pulse.frequency = target_drive_freq
            if params.pulse_amplitude is not None:
                cr_pulse.amplitude = params.pulse_amplitude
            if params.pulse_duration is not None:
                cr_pulse.duration = params.pulse_duration

            sequence.add(cr_pulse)

            # add readout pulses
            for qubit in pair:
                sequence.add(platform.create_qubit_readout_pulse(qubit, start=cr_pulse.finish))
                
            # create a duration sweeper for the pulse duration
            sweeper_amplitude = Sweeper(
                Parameter.amplitude,
                params.amplitude_factor,
                pulses=[cr_pulse],
                type=SweeperType.FACTOR,
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
                sweeper_amplitude,
                sweeper_frequency,
            )

            # store the results for each qubit in the pair in the data object
            # NOTE: Change this to use standard qibocal>auto>operation>Data>register_qubit
            for qubit in pair:
                prob = results[qubit].probability(state=1)
                data.register_qubit(
                    CrossResonanceChevronFrequencyType,
                    (qubit, target, control, setup),
                    dict(
                        prob=prob,
                        amp=params.amplitude_factor*cr_pulse.amplitude,
                        freq=cr_pulse.frequency + params.frequency_range,
                        )
                )

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
                    x=qubit_data.amp,
                    y=qubit_data.freq * HZ_TO_GHZ,
                    z=qubit_data.prob,
                    name=f"Control at |{setup}>",
                    coloraxis="coloraxis"
                ),
                row=1,
                col=i+1,
            )
            fig.update_xaxes(title_text="Pulse Amplitude [a.u.]", row=1, col=i+1)
            fig.update_yaxes(title_text="Frequency [GHz]", row=1, col=i+1)
        fig.update_layout(coloraxis={'colorscale':'Plasma'})
        figs.append(fig)

            
    return figs, ""

cross_resonance_chevron_amplitude_frequency = Routine(_acquisition, _fit, _plot)
"""CrossResonanceChevronFrequency Routine object."""