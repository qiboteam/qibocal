from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibolab.native import NativePulse
from qibolab.qubits import QubitId, QubitPairId
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibolab.pulses import Pulse, PulseSequence, PulseShape, PulseType, Gaussian, Drag, Rectangular, GaussianSquare, DrivePulse

from .utils import STATES, BASIS, ro_projection_pulse
from .length import CrossResonanceType, CrossResonanceParameters
from qibo.backends import matrices
from typing import Literal


type_basis = Literal[BASIS]
"""Measurement basis type for the Cross Resonance CNOT calibration."""

type_setup = Literal[STATES]
"""Measurement basis type for the Cross Resonance CNOT calibration."""

projections: list[type_basis] = BASIS
"""Projections to measure for the Cross Resonance CNOT calibration."""



@dataclass
class CrossResonanceCNOTData(Data):
    """Data structure for Cross Resonance CNOT calibration."""

    data: dict[(QubitPairId, type_setup, type_basis), 
               npt.NDArray[CrossResonanceType]] = field(default_factory=dict)
    """Raw data acquired."""


@dataclass
class CrossResonanceCNOTParameters(CrossResonanceParameters):
    """Cross Resonance Gate Calibration runcard inputs."""
    
    pulse_shape: str = None
    """Pulse shape for the CR pulse. Options: 'gaussian', 'drag', 'rectangular'."""

    target_amplitude: float = 0.0
    """Traget pulse amplitude for correction."""


@dataclass
class  CrossResonanceCNOTResults(Results):
    """Cross Resonance Gate Calibration outputs."""

    exp_t: dict[QubitPairId, dict[str, str, dict[list[float],list[float],list[float]] ]] = field(default_factory=dict)
     #          pair(Target, Control), control state, basis, expectation value
    """Expectation values for each basis in the Pauli group as a function of the pulse length."""

    measured_density_matrix: dict[QubitId, int, list] = field(default_factory=dict)
    """Complex measured density matrix."""


def _acquisition(
    params: CrossResonanceCNOTParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceCNOTData:
    """Data acquisition for Cross Resonance Gate Calibration.
    The gate consists on a pi/2 pulse on the target qubit followed by a CR pulse on the control qubit.
    The control qubit is prepared in either the |0> or |1> state and the target qubit is prepared in the |+> state.
        Target:     --[X/2]---[Pulse(omega_t, amp_t)]---[RO]--
                               
        Control:    ----------[Pulse(omega_t, amp_c)]------------[RO]--
    """

    print("Starting Cross Resonance CNOT calibration")
    data = CrossResonanceCNOTData()
    
    ro_pulses = {}

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
                
                if params.pulse_shape is None:
                    shape = Rectangular()

                cr_pulse: DrivePulse = DrivePulse(start=next_start,
                                duration=params.pulse_duration_start,
                                amplitude=ctr_native_rx.amplitude,
                                frequency=tgt_native_rx.frequency,   # target frequency at control qubit
                                relative_phase=0,
                                shape=shape,
                                channel= ctr_native_rx.channel, 
                                qubit=control,
                                )
                
                if params.pulse_amplitude is not None:
                    cr_pulse.amplitude = params.pulse_amplitude
                
                cr_pulse_tgt = tgt_native_rx.copy()
                cr_pulse_tgt.amplitude = params.target_amplitude
                cr_pulse_tgt.start = next_start

                sequence.add(cr_pulse)
                sequence.add(cr_pulse_tgt)

                # Add readout pulses    
                projection_pulse, ro_pulses = ro_projection_pulse(
                    platform, target, start=cr_pulse.finish, projection=basis  
                )
                sequence.add(projection_pulse)
                sequence.add(ro_pulses) 

                sweeper_duration = Sweeper(
                    parameter = Parameter.duration,
                    values = params.duration_range,
                    pulses = [cr_pulse, cr_pulse_tgt],
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

                # store the results
                probability = results[target].probability(state=1)
                
                data.register_qubit(
                    CrossResonanceType,
                    data_keys=(
                        QubitPairId(target, control), 
                        ctr_setup, 
                        basis
                    ),
                    data_dict=dict(
                            prob=probability,
                            length=params.duration_range,
                        ),
                )


    return data


def _fit(data: CrossResonanceCNOTData) -> CrossResonanceCNOTResults:
    """Post-processing function for Cross Resonance Gate Calibration."""
    measured_density_matrix = {}
    #raise NotImplementedError("Fitting function not implemented yet.")

    list_pairs = list(set([key[0] for key in data.data.keys()]))

    print(f"Starting Cross Resonance CNOT calibration fitting on targets {list_pairs}")
    exp_t = {}
    # Calculate expectation values as a function of the pulse length
    for i, (target, control) in enumerate(list_pairs):
        for ctr_setup in STATES:  
                _exp_t = {} # expectation values for each basis in the Pauli group as a function of the pulse length
                x = {} # pulse length
                for basis in BASIS:
                    if ((target,control), ctr_setup, basis) not in data.data:
                        print(f"All basis need to be measured for fitting {target}.")
                        break
                    x = data.data[(target,control), ctr_setup, basis].length
                    _exp_t[basis] = {'duration': x.tolist(),
                                    'exp_real': np.real(1 - 2 * data.data[(target,control), ctr_setup, basis].prob).tolist(),
                                    'exp_imag': np.imag(1 - 2 * data.data[(target,control), ctr_setup, basis].prob).tolist()}
                                    
                exp_t[(target, control), ctr_setup] = _exp_t
    
    # # Calculate the expectation values for a rabi pulse
    # results.exp_t = exp_t
    # for i, pair in enumerate(data.targets):
    #     target = pair[0]
    #     for ctr_setup in STATES:  
    #         try:
    #             exp = {}

    #             # Calculate the expectation values for a pulse length
    #             # For this we can either fit the oscillation to a decaying sinusoidal function or directly measure 
    #             # the expectation values for the X, Y and Z basis at a fixed pulse length
    #             from scipy.optimize import curve_fit
    #             # fit_function = lambda x, A, B, f, phi: (A * np.cos(2 * pi * f * x - phi) + B)
    #             fit_function = lambda x, a0, a, b, c, d: a0 + a * np.cos(c * x + d)**2 #* np.exp(-b * x)
    #             for basis in BASIS:
    #                 # Fit decaying sinusoidal function
    #                 tau = max(x[basis])/2
    #                 rabi_duration = tau/2
    #                 inital_guess = [0.5, 0.5, 1/tau, 1/(2*np.pi*rabi_duration), np.pi/2]
    #                 # define ranges for the parameters
    #                 bounds = ([0.0, 0.0, 0, 0, -np.pi], [1.0, 1.0,  10/tau, 10/(2*np.pi*tau), np.pi])

    #                 print(f"Fitting {basis} basis, max_time = {max(x[basis])} ns")
                    
    #                 # Remove warnings
    #                 import warnings
    #                 warnings.filterwarnings("ignore")
    #                 popt, pcov = curve_fit(fit_function, x[basis], exp_t[basis], p0=inital_guess, bounds=bounds)
    #                 warnings.filterwarnings("default")

    #                 print(f"Fit parameters: {popt[0]:.2f}, {popt[1]:.2f}, {popt[2]:.2f}, {popt[3]:.2f}, {popt[4]:.2f}")
    #                 if pcov[0,0] > 0.1:
    #                     print(f"Fit failed for {basis} basis")
    #                 else:
    #                     exp[basis] = popt[2]
    #                     print(f"Fit for {basis} basis: {exp[basis]} GHz ({2*np.pi/exp[basis]} ns)")

    #             measured_density_matrix = 0.5 * (
    #                 np.eye(2) + 
    #                 exp["X"] * np.array([[0, 1], [1, 0]]) + 
    #                 exp["Y"] * np.array([[0, -1j], [1j, 0]]) + 
    #                 exp["Z"] * np.array([[1, 0], [0, -1]])
    #             )

    #             result[ctr_setup, qubit, 'exp'] = exp
    #             result[ctr_setup, qubit, 'measured_density_matrix'] =  np.real(measured_density_matrix).tolist()
    #         except:
    #             print(f"Failed to fit {target} qubit")
    #             continue
                
    results = CrossResonanceCNOTResults(exp_t=exp_t) #, measured_density_matrix={})
    return results


def _plot(data: CrossResonanceCNOTData, target: QubitPairId, fit: CrossResonanceCNOTResults):
    """Plotting function for Cross Resonance Gate Calibration."""
    print(fit)
    figs = []
    for ro_qubit in target:
        fig = make_subplots(rows=len(BASIS), cols=1)
        show_legend = True
        # Make a subplot for each projection
        for projection in BASIS:
            # Check if it exists
            if (ro_qubit, target[0], target[1], STATES[0], projection) not in data.data:
                continue
            
            # Add a subplot
            for ctr_setup in STATES:
                if fit is not None:
                    _data = fit.exp_t[ro_qubit, target[0], target[1], ctr_setup][projection]
                else:
                    _data = data.data[ro_qubit, target[0], target[1], ctr_setup, projection]
                fig.add_trace(
                    go.Scatter(
                        x=_data.length, y=_data.prob, 
                        name= f"Control in |{ctr_setup}>",
                        mode="lines+markers",
                        line = dict(color="blue" if ctr_setup == STATES[0] else "red"),
                        showlegend=show_legend,
                    ),
                    row = BASIS.index(projection)+1, col =1,
                )
            show_legend = False

        for projection in BASIS:
            if fit is not None:
                fig.update_yaxes(range=[-1, 1],
                             title_text=f"<{projection}(t)>", 
                             row=BASIS.index(projection)+1, col=1)
            else:
                fig.update_yaxes(range=[0, 1],
                             title_text=f"<{projection}(t)>", 
                             row=BASIS.index(projection)+1, col=1)

        fig.update_layout(
            title=f"Qubit {ro_qubit}",
            xaxis_title="CR Pulse Length (ns)",
            height=800, 
        )

        figs.append(fig)
    return figs, ""

cross_resonance_cnot = Routine(_acquisition, _fit, _plot)
"""CrossResonance Length CNOT calibration object."""
