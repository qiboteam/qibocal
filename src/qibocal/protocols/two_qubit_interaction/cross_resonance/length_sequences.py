from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import itertools

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.two_qubit_interaction.cross_resonance.length import (
    CrossResonanceLengthParameters,
    CrossResonanceLengthResults,
    CrossResonanceLengthData,
    CrossResonanceLengthType,
)
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses import Gaussian, Drag, Rectangular, GaussianSquare

from .utils import STATES, cr_plot

def _acquisition(
    params: CrossResonanceLengthParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceLengthData:
    """Data acquisition for Cross Resonance Gate Calibration using Sequences."""
    from qibolab.native import NativePulse

    data = CrossResonanceLengthData()
    
    basis = 'Z'
    
    for pair in targets:
        target, control = pair
        for tgt_setup, ctr_setup in itertools.product(params.tgt_setups, STATES):
            probability, length, error = {target: [], control: []}, {target: [], control: []}, {target: [], control: []}
            for duration in params.duration_range:
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
                            duration=duration,
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

                for ro_qubit in pair:
                    sequence.add(platform.create_qubit_readout_pulse(ro_qubit, start=cr_pulse.finish))
                        
                results = platform.execute_pulse_sequence(
                    sequence,
                    ExecutionParameters(
                        nshots=params.nshots,
                        relaxation_time=params.relaxation_time,
                        acquisition_type=AcquisitionType.DISCRIMINATION,
                        # averaging_mode=AveragingMode.SINGLESHOT,
                        averaging_mode=AveragingMode.CYCLIC,
                    ),
                )
                for ro_qubit in pair:
                    probability[ro_qubit].append(results[ro_qubit].probability(state=1))
                    length[ro_qubit].append(duration)
                    error[ro_qubit].append(np.sqrt(probability[ro_qubit][-1]*(1-probability[ro_qubit][-1])/params.nshots))
                
            for ro_qubit in pair:
                data.register_qubit(
                    CrossResonanceLengthType,
                    (pair, ro_qubit, tgt_setup, ctr_setup, basis),
                    dict(
                        prob=probability[ro_qubit],
                        length=length[ro_qubit],
                        error=error[ro_qubit],
                    ),
    )
        
    return data


def _fit(
    data: CrossResonanceLengthData,
) -> CrossResonanceLengthResults:
    """Post-processing function for Cross Resonance Gate Calibration using Sequences."""
    return CrossResonanceLengthResults()


def _plot(data: CrossResonanceLengthData, target: QubitPairId, fit: CrossResonanceLengthResults
          ) -> tuple[list[go.Figure], str]:
    """Plotting function for Cross Resonance Gate Calibration using Sequences."""
    return cr_plot(data,target, 'duration'), ""



cross_resonance_length_sequences = Routine(_acquisition, _fit, _plot, two_qubit_gates=True)
"""CrossResonance Routine object."""
