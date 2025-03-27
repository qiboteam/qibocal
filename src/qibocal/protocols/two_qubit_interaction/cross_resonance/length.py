from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import itertools

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibolab.native import NativePulse
from qibolab.qubits import QubitId, QubitPairId
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses import Gaussian, Drag, Rectangular, GaussianSquare

from .utils import STATES, BASIS, ro_projection_pulse, cr_plot, Setup, Basis

CrossResonanceLengthType = np.dtype(
    [
        ("prob", np.float64),
        ("duration", np.int64),
        ("error", np.int64),
    ]
)
"""Custom dtype for Cross Resonance Gate Calibration with Swept pulse duration."""

@dataclass
class CrossResonanceLengthParameters(Parameters):
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
    projections: Optional[list[str]] = field(default_factory=lambda: [BASIS[2]])
    """Measurement porjection"""
    tgt_setups: Optional[list[str]] = field(default_factory=lambda: [STATES[0]])
    """Setup for the experiment."""
    
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
class CrossResonanceLengthResults(Results):
    """Cross Resonance Gate Calibration outputs."""


@dataclass
class CrossResonanceLengthData(Data):
    """Data structure for Cross Resonance Gate Calibration."""

    targets: list[QubitPairId] = field(default_factory=list)
    """Targets for the Cross Resonance Gate Calibration stored as pair of [target, control]."""

    data: dict[(QubitPairId, QubitId, Setup, Setup, Basis), 
               npt.NDArray[CrossResonanceLengthType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceLengthParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceLengthData:
    """Data acquisition for Cross Resonance Gate Calibration."""

    data = CrossResonanceLengthData(targets=targets)

    for pair in targets:
        target, control = pair
        for tgt_setup, ctr_setup, basis  in itertools.product(params.tgt_setups, STATES, params.projections):
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

                # Add readout pulses
                projection_pulse , ro_pulses = {}, {}
                for ro_qubit in pair:
                    projection_pulse[ro_qubit], ro_pulses[ro_qubit] = ro_projection_pulse(
                        platform, ro_qubit, start=cr_pulse.finish, projection=basis  
                    )
                    sequence.add(projection_pulse[ro_qubit])
                    sequence.add(ro_pulses[ro_qubit]) 

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
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                    sweeper_duration,
                )

                # store the results
                for ro_qubit in pair:
                    probability = results[ro_qubit].probability(state=1)
                    data.register_qubit(
                        CrossResonanceLengthType,
                        (pair, ro_qubit, tgt_setup, ctr_setup, basis),
                        dict(
                            prob=probability,
                            duration=params.duration_range,
                            error=np.sqrt(probability * (1 - probability) / params.nshots).tolist(),
                        ),
                    )
    return data

def _fit(
    data: CrossResonanceLengthData,
) -> CrossResonanceLengthResults:
    """Post-processing function for Cross Resonance Gate Calibration."""
    return CrossResonanceLengthResults()

def _plot(data: CrossResonanceLengthData, target: QubitPairId, fit: CrossResonanceLengthResults
          ) -> tuple[list[go.Figure], str]:
    """Plotting function for Cross Resonance Gate Calibration."""
    return cr_plot(data,target, 'duration'), ""

cross_resonance_length = Routine(_acquisition, _fit, _plot)
"""CrossResonance Duration Routine object."""
