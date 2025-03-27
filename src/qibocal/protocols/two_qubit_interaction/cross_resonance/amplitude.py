from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import itertools

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses import Gaussian, Drag, Rectangular, GaussianSquare

from qibolab.native import NativePulse
from qibolab.qubits import QubitId, QubitPairId

from qibocal.auto.operation import Data, Parameters, Results, Routine
from .utils import STATES, BASIS, ro_projection_pulse, cr_plot, Setup, Basis

CrossResonanceType = np.dtype(
    [
        ("prob", np.float64),
        ("amp", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for Cross Resonance Amplitude Gate Calibration."""

@dataclass
class CrossResonanceParameters(Parameters):
    """Cross Resonance Amplitude Gate Calibration runcard inputs."""

    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    pulse_duration: Optional[int] = None
    """CR pulse duration [ns]."""
    pulse_amplitude: Optional[float] = None
    """CR pulse amplitude [ns]."""
    shape: Optional[str] = "Rectangular()"
    """CR pulse shape parameters."""
    projections: Optional[list[str]] = field(default_factory=lambda: [BASIS[-1]])
    """Measurement porjection"""
    tgt_setup: Optional[list[str]] = field(default_factory=lambda: [STATES[0]])
    """Setup for the experiment."""

    @property
    def amplitude_factor_range(self):
        return np.arange(
            self.min_amp_factor, self.max_amp_factor, self.step_amp_factor
        )
    """Amplitude factor range."""
    @property
    def pulse_shape(self):
        return eval(self.shape)
    """Cross Resonance Pulse shape."""


@dataclass
class CrossResonanceResults(Results):
    """Cross Resonance Gate Calibration outputs."""


@dataclass
class CrossResonanceData(Data):
    """Data structure for Cross Resonance Gate Calibration in Pulse Amplitude."""

    targets: list[QubitPairId] = field(default_factory=list)
    """Targets for the Cross Resonance Gate Calibration stored as pair of [target, control]."""

    data: dict[(QubitPairId, QubitId, Setup, Setup, Basis), 
               npt.NDArray[CrossResonanceType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceData:
    """Data acquisition for Cross Resonance Gate Calibration."""
    
    data = CrossResonanceData(targets=targets)
    for pair in targets:
        target, control = pair
        for ctr_setup, tgt_setup, basis in itertools.product(STATES, params.tgt_setup, params.projections):
            ctr_native_rx = platform.qubits[control].native_gates.RX.pulse(start=0)
            tgt_native_rx = platform.qubits[target].native_gates.RX.pulse(start=0)
            
            sequence = PulseSequence()

            if ctr_setup == STATES[1]:
                sequence.add(ctr_native_rx)

            if tgt_setup == STATES[1]:
                sequence.add(tgt_native_rx)

            next_start = max(tgt_native_rx.finish, ctr_native_rx.finish)
            
            cr_pulse: Pulse = Pulse(start = next_start,
                            duration = ctr_native_rx.duration,
                            amplitude = ctr_native_rx.amplitude,
                            frequency = tgt_native_rx.frequency,
                            relative_phase=0,
                            channel=ctr_native_rx.channel,
                            shape = params.pulse_shape,
                            type = PulseType.DRIVE,
                            qubit = control,
                            )

            if params.pulse_duration is not None:
                cr_pulse.duration = params.pulse_duration
            if params.pulse_amplitude is not None:
                cr_pulse.amplitude = params.pulse_amplitude
            
            sequence.add(cr_pulse)

            projection_pulse, ro_pulses = {}, {}
            for ro_qubit in pair:
                # sequence.add(platform.create_qubit_readout_pulse(qubit=qubit, start=cr_pulse.finish))
                projection_pulse[ro_qubit], ro_pulses[ro_qubit] = ro_projection_pulse(
                    platform, ro_qubit, start=cr_pulse.finish, projection=basis  
                )
                sequence.add(projection_pulse[ro_qubit]) 
                sequence.add(ro_pulses[ro_qubit]) 

            sweeper_amplitude = Sweeper(
                parameter = Parameter.amplitude,
                values = cr_pulse.amplitude*params.amplitude_factor_range,
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
                sweeper_amplitude,
            )
            
            # store the results
            for ro_qubit in pair:
                probability = results[ro_qubit].probability(state=1)
                data.register_qubit(
                    CrossResonanceType,
                    data_keys= (pair, ro_qubit, tgt_setup, ctr_setup, basis),
                    data_dict= dict(
                        prob=probability,
                        amp=cr_pulse.amplitude * params.amplitude_factor_range,
                        error=np.sqrt(probability * (1 - probability) / params.nshots).tolist(),
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
    return cr_plot(data,target, 'amp'), ""

cross_resonance_amplitude = Routine(_acquisition, _fit, _plot)
"""CrossResonance Amplitude Routine object."""
