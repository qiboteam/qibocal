from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibolab.native import NativePulse
from qibolab.qubits import QubitId, QubitPairId
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibolab.pulses import Pulse, PulseSequence, PulseShape, PulseType, Gaussian, Drag, Rectangular, GaussianSquare, DrivePulse

from .utils import STATES, BASIS, ro_projection_pulse,cr_plot, Setup, Basis
from .length import (
    CrossResonanceLengthType, 
    CrossResonanceLengthParameters, 
    CrossResonanceLengthData,
    CrossResonanceLengthResults,
)
from qibo.backends import matrices
from typing import Literal, List

projections: list[str] = BASIS
"""Projections to measure for the Cross Resonance CNOT calibration."""


@dataclass
class CrossResonanceCnotLengthData(CrossResonanceLengthData):
    """Data structure for Cross Resonance CNOT calibration."""

    native: str = "CNOT"
    """Native gate for the calibration."""

    def __getitem__(self, pair):
        return {
            index: value
            for index, value in self.data.items()
            if set(pair).issubset(index)
        }


@dataclass
class CrossResonanceCNOTParameters(CrossResonanceLengthParameters):
    """Cross Resonance Gate Calibration runcard inputs."""

    target_amplitude: float = 0.0
    """Target pulse amplitude for ZZ correction."""

    echo : bool = False
    """Echo pulse for the control qubit."""


@dataclass
class  CrossResonanceCnotLengthResults(CrossResonanceLengthResults):
    """Cross Resonance Gate Calibration outputs."""

    # exp_t: dict[QubitPairId, dict[str, str, dict[list[float],list[float],list[float]] ]] = field(default_factory=dict)
    #  #          pair(Target, Control), control state, basis, expectation value
    # """Expectation values for each basis in the Pauli group as a function of the pulse length."""

    #measured_density_matrix: dict[QubitId, int, list] = field(default_factory=dict)
    #"""Complex measured density matrix."""


def _acquisition(
    params: CrossResonanceCNOTParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceCnotLengthData:
    """Data acquisition for Cross Resonance Gate Calibration.
    The gate consists on a pi/2 pulse on the target qubit followed by a CR pulse on the control qubit.
    The control qubit is prepared in either the |0> or |1> state and the target qubit is prepared in the |+> state.
        Target:     --[X/2]---[Pulse(omega_t, amp_t, phi=0)]---------[Pulse(omega_t, amp_t, phi=pi)]---[RO]--
               
        Control:    ---[X*]---[Pulse(omega_t, amp_c, phi=0)]---[X*]--[Pulse(omega_t, amp_c, phi=pi)]---[RO]--
    """

    data = CrossResonanceCnotLengthData(targets = targets)    
    if isinstance(params.pulse_shape, str):
        shape = PulseShape.eval(params.pulse_shape)
    else:
        shape = Rectangular()

    for pair in targets:
        for ctr_setup in STATES:
            for basis in BASIS:
                target, control = pair
                tgt_native_rx:NativePulse = platform.qubits[target].native_gates.RX90.pulse(start=0)
                ctr_native_rx:NativePulse = platform.qubits[control].native_gates.RX.pulse(start=0)

                sequence = PulseSequence()
                next_start = 0
                
                sequence.add(tgt_native_rx)
                next_start = tgt_native_rx.finish

                if ctr_setup == STATES[1]:
                    sequence.add(ctr_native_rx)
                    next_start = max(ctr_native_rx.finish, next_start)
                
                # cr_pulse_tgt = tgt_native_rx.copy()
                # cr_pulse_tgt.amplitude = params.target_amplitude
                # cr_pulse_tgt.start = next_start
                # cr_pulse_tgt.duration = params.pulse_duration_start
                # cr_pulse_tgt.shape = shape

                cr_pulse: Pulse = Pulse(start=next_start,
                                duration=params.pulse_duration_start,
                                amplitude=ctr_native_rx.amplitude,
                                frequency=tgt_native_rx.frequency,   # control frequency
                                relative_phase=0,
                                shape=params.pulse_shape,
                                qubit=control,
                                channel= ctr_native_rx.channel ,
                                type=PulseType.DRIVE
                                )
                if params.pulse_amplitude is not None:
                    cr_pulse.amplitude = params.pulse_amplitude
                
                sequence.add(cr_pulse)    
                next_start = cr_pulse.finish
                
                ## Echo Pulse
                # if ctr_setup == STATES[1]:
                #     ctr_rx2:NativePulse = platform.qubits[control].native_gates.RX.pulse(start=next_start)
                #     sequence.add(platform.qubits[control].native_gates.RX.pulse(start=next_start))
                #     next_start = max(ctr_rx2.finish, next_start)

                # cr_pulse_tgt_echo = tgt_native_rx.copy()
                # cr_pulse_tgt_echo.amplitude = params.target_amplitude
                # cr_pulse_tgt_echo.start = cr_pulse.finish
                # cr_pulse_tgt_echo.duration = params.pulse_duration_start/2
                # cr_pulse_tgt_echo.shape = shape
                # cr_pulse_tgt_echo.relative_phase = 180

                # cr_pulse_echo: Pulse = Pulse(start=next_start,
                #                 duration=params.pulse_duration_start,
                #                 amplitude=ctr_native_rx.amplitude,
                #                 frequency=tgt_native_rx.frequency,   # target frequency at control qubit
                #                 relative_phase=180,
                #                 shape=shape,
                #                 channel= ctr_native_rx.channel, 
                #                 qubit=control,
                #                 type= PulseType.DRIVE,
                #                 )
                
                # if params.pulse_amplitude is not None:
                #     cr_pulse_echo.amplitude = params.pulse_amplitude
                # next_start = cr_pulse_echo.finish
                # sequence.add(cr_pulse_echo)
                
                # sequence.add(cr_pulse_tgt)
                # sequence.add(cr_pulse_tgt_echo)

                sequence.add(cr_pulse)
                
                # Add Readout pulses  
                ro_qubit = target  
                projection_pulses, ro_pulses= {}, {}
                projection_pulses[ro_qubit], ro_pulses[ro_qubit] = ro_projection_pulse(
                    platform, ro_qubit, start=cr_pulse.finish, projection=basis  
                )
                sequence.add(projection_pulses[ro_qubit])
                sequence.add(ro_pulses[ro_qubit]) 

                sweeper_duration = Sweeper(
                    parameter = Parameter.duration,
                    values = params.duration_range,
                    pulses = [cr_pulse],# cr_pulse_echo, cr_pulse_tgt,cr_pulse_tgt_echo],
                    type = SweeperType.ABSOLUTE,
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

                # Store the results
                probability = results[ro_qubit].probability(state=1)
                data.register_qubit(
                    CrossResonanceLengthType,
                    data_keys=(pair, ro_qubit, 'X90', ctr_setup, basis),
                    data_dict=dict(
                            prob=probability,
                            duration=params.duration_range,
                            error = np.sqrt(probability*(1-probability)/params.nshots).tolist()
                        ),
                )

    return data


def _fit(data: CrossResonanceCnotLengthData) -> CrossResonanceCnotLengthResults:
    """Post-processing function for Cross Resonance Gate Calibration."""
    return CrossResonanceCnotLengthResults()


def _plot(data: CrossResonanceCnotLengthData, target: QubitPairId, fit: CrossResonanceCnotLengthResults):
    """Plotting function for Cross Resonance Gate Calibration."""
    pair = tuple(target)
    figs = cr_plot(data,target, 'duration')
    return figs, ""

cross_resonance_cnot_length = Routine(_acquisition, _fit, _plot)
"""CrossResonance Length CNOT calibration object."""
