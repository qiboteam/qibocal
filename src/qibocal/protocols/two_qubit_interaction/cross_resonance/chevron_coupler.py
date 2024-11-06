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
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses import Gaussian, Drag, Rectangular, GaussianSquare

from qibocal.protocols.two_qubit_interaction.utils import order_pair


from .utils import STATES

CrossResonanceChevronType = np.dtype(
    [
        ("flux", np.int64),
        ("amp", np.float64),
        ("prob", np.float64),
    ]
)
"""Custom dtype for cross resonance chevron calibration."""

@dataclass
class CrossResonanceChevronParameters(Parameters):
    """cross resonance chevron runcard inputs."""

    coupler_amplitude_min_factor: float
    """Initial coupler flux pulse amplitude."""
    coupler_amplitude_max_factor: float
    """Final coupler flux pulse amplitude."""
    coupler_amplitude_step_factor: float
    """Step coupler flux pulse amplitude."""
    
    amplitude_min_factor: float
    """CR pulse amplitude minimum."""
    amplitude_max_factor: float
    """CR pulse amplitude maximum."""
    amplitude_step_factor: float
    """CR pulse amplitude step."""
    
    pulse_amplitude: Optional[float] = None
    pulse_duration: Optional[int] = None
    shape: Optional[str] = "Rectangular()"
    """CR pulse shape parameters."""
    @property
    def pulse_shape(self):
        return eval(self.shape)
    """Cross Resonance Pulse shape."""

    @property
    def amplitude_factor_range(self):
        return np.arange(
            self.amplitude_min_factor,
            self.amplitude_max_factor,
            self.amplitude_step_factor,
        )

    @property
    def coupler_amplitude_range(self):
        return np.arange(
            self.coupler_amplitude_min_factor,
            self.coupler_amplitude_max_factor,
            self.coupler_amplitude_step_factor,
    )

@dataclass
class CrossResonanceChevronResults(Results):
    """cross resonance chevron outputs."""


@dataclass
class CrossResonanceChevronData(Data):
    """Data structure for cross resonance chevron."""

    data: dict[QubitId, npt.NDArray[CrossResonanceChevronType]] = field(
        default_factory=dict
    )
    """Raw data acquired."""

    def register_qubit(self, dtype, key, prob, flux, amp):
        """Store output for single qubit."""
        size = len(flux) * len(amp)
        amplitude, flux_amplitude = np.meshgrid(amp, flux)
        ar = np.empty(size, dtype=dtype)
        ar["flux"] = flux_amplitude.ravel()
        ar["amp"] = amplitude.ravel()
        ar["prob"] = prob.ravel()
        self.data[key] = np.rec.array(ar)


def _acquisition(
    params: CrossResonanceChevronParameters,
    platform: Platform,
    targets: list[QubitPairId],
) -> CrossResonanceChevronData:
    """Data acquisition for cross resonance chevron."""
    """Run a CR gate with variable amplitude for different coupling flux bias."""
    data = CrossResonanceChevronData()

    for pair in targets:
        for setup in STATES:
            target, control = pair
            ordered_pair = order_pair(pair, platform)

            sequence = PulseSequence()
            target_drive_freq = platform.qubits[target].native_gates.RX.frequency
            rx_control = platform.create_RX_pulse(control, 0)

            next_start = 0
            if setup == STATES[1]:
                next_start = rx_control.finish
                sequence.add(rx_control)

            native_gate, _ = platform.create_CZ_pulse_sequence(
                (target,control),
                start=sequence.finish,
            )

            flux_pulses = [p for p in native_gate.coupler_pulses(*pair)][:1]
            flux_amplitude = getattr(flux_pulses[0], "amplitude")

            # Cross resonance pulse
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
            
            if params.pulse_amplitude is not None:
                cr_pulse.amplitude = params.pulse_amplitude
            if params.pulse_duration is not None:
                cr_pulse.duration = params.pulse_duration

            sequence.add(cr_pulse)
            sequence.add(
                platform.create_qubit_readout_pulse(target, start=cr_pulse.finish)
            )

            sweeper_coupler = Sweeper(
                Parameter.amplitude,
                params.coupler_amplitude_range,
                pulses=[flux_pulses],
                type=SweeperType.FACTOR,
            )

            sweeper_amplitude = Sweeper(
                Parameter.amplitude,
                cr_pulse.amplitude * params.amplitude_factor_range,
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
                sweeper_coupler,
                sweeper_amplitude,
            )

            # store the results
            prob = results[target].probability(state=1)
            data.register_qubit(
                dtype   = CrossResonanceChevronType,
                key     = (target, control, setup),
                prob    = prob,
                flux    = flux_amplitude*params.coupler_amplitude_range,
                amp     = cr_pulse.amplitude*params.amplitude_factor_range,
            )
    # finally, save the remaining data
    return data


def _fit(
    data: CrossResonanceChevronData,
) -> CrossResonanceChevronResults:
    """Post-processing function for Chevron with CR amplitude and coupler flux bias."""
    return CrossResonanceChevronResults()


def _plot(
    data: CrossResonanceChevronData,
    target: QubitPairId,
    fit: CrossResonanceChevronResults,
):
    pair = target
    figs = []

    """Plotting function for Cross Resonance Chevron with flux bias sweep."""
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
                x=qubit_data.flux,
                y=qubit_data.amp,
                z=qubit_data.prob,
                name=f"Control at {setup}",
                coloraxis="coloraxis"
            ),
            row=1,
            col=i+1,
        )
        fig.update_xaxes(title_text="Flux Pulse Amplitude [a.u.]", row=1, col=i+1)
        fig.update_yaxes(title_text="CR Pulse Amplitude [a.u.]", row=1, col=i+1)
    fig.update_layout(coloraxis={'colorscale':'Plasma'})
    
    figs.append(fig)
    return figs, ""


cross_resonance_chevron_coupler = Routine(_acquisition, _fit, _plot)
"""CrossResonanceChevron Coupler Routine object."""
