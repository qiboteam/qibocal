from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import itertools

from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from qibolab.native import NativePulse
from qibolab.qubits import QubitId, QubitPairId
from qibolab.pulses import Pulse, PulseSequence, PulseType
from qibolab.pulses import Gaussian, Drag, Rectangular, GaussianSquare


from qibocal.config import log
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.utils import guess_period, fallback_period, chi2_reduced, table_html, table_dict
from qibocal.protocols.rabi.utils import fit_length_function, rabi_length_function

from .utils import STATES, BASIS, ro_projection_pulse, cr_plot, Setup, Basis

CrossResonanceLengthType = np.dtype(
    [
        ("prob", np.float64),
        ("duration", np.int64),
        ("error", np.float64),
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
    tgt_setup: Optional[list[str]] = field(default_factory=lambda: [STATES[0]])
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

    def __dict__(self):
        """Convert the object to a dictionary."""
        return {
            prop: getattr(self, prop) for prop in self.__dataclass_fields__
            if prop not in ["pulse_shape", "duration_range"]
        }


@dataclass
class CrossResonanceLengthResults(Results):
    """Cross Resonance Gate Calibration outputs."""
    amplitude: dict[QubitPairId, dict[Setup, float]]
    """Pi pulse amplitude. Same for all bases in each qubit pair test. If specified in the runcard, same for all pairs."""
    duration: dict[QubitPairId, dict[Setup, float]] = field(default_factory=dict)
    """Fitted pi pulse duration for each qubit."""
    fitted_parameters: dict[QubitPairId, dict[Setup, Setup, Basis, Union[float, list[float]]]] = field(default_factory=dict)
    """Raw fitting output."""
    chi2: dict[QubitPairId, dict[Setup, Setup, Union[float, list[float]]]] = field(default_factory=dict)
    """Reduced chi2 for each fitting."""
    Jeff: dict[QubitPairId, Union[float, list[float]]] = field(default_factory=dict)
    """Effective coupling strength for each qubit pair."""

@dataclass
class CrossResonanceLengthData(Data):
    """Data structure for Cross Resonance Gate Calibration."""

    targets: list[QubitPairId] = field(default_factory=list)
    """Targets for the Cross Resonance Gate Calibration stored as pair of [target, control]."""

    parameters: dict = field(default_factory=dict)
    """Parameters for the Cross Resonance Gate Calibration."""

    amplitude: dict[QubitPairId, float] = field(default_factory=dict)
    """Amplitude of the qubit drive pulse."""

    data: dict[(QubitPairId, QubitId, Setup, Setup, Basis), 
               npt.NDArray[CrossResonanceLengthType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: CrossResonanceLengthParameters, platform: Platform, targets: list[QubitPairId]
) -> CrossResonanceLengthData:
    """Data acquisition for Cross Resonance Gate Calibration."""

    parameters = params.__dict__()
    parameters["ctr_setup"] = STATES
    log.info(f"Cross Resonance Gate Calibration parameters: {parameters}")
    data = CrossResonanceLengthData(
        targets=targets,
        parameters=parameters,
    )
    amplitude = {}
    for pair in targets:
        target, control = pair
        for tgt_setup, ctr_setup, basis  in itertools.product(params.tgt_setup, STATES, params.projections):
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
                    amplitude[pair] = params.pulse_amplitude
                else:
                    amplitude[pair] = cr_pulse.amplitude
                
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
                    probability = np.array(results[ro_qubit].probability(state=1))
                    data.register_qubit(
                        CrossResonanceLengthType,
                        (pair, ro_qubit, tgt_setup, ctr_setup, basis),
                        dict(
                            prob=probability.tolist(),
                            duration=params.duration_range,
                            error=np.sqrt(probability * (1 - probability) / params.nshots).tolist(),
                        ),
                    )
                    
    data.amplitude = amplitude
    
    return data

def _fit(
    data: CrossResonanceLengthData,
) -> CrossResonanceLengthResults:
    """Post-processing function for Cross Resonance Gate Calibration."""
    tgt_setups = data.parameters['tgt_setup']
    ctr_setups = data.parameters['ctr_setup'] if 'ctr_setup' in data.parameters else STATES
    projections = data.parameters['projections']
    amplitude = data.amplitude

    # Initialize dictionaries to store the results
    fit_popt = {}
    fit_chi2 = {}
    fit_duration = {}
    fit_amplitude = {}
    fit_Jeff = {}
    # Loop over the qubit pairs and their setups
    for i, pair in enumerate(data.targets):
        target, control = pair

        _popt = {}
        _chi2 = {}
        _duration = {}
        _amplitude = {}
        
        for tgt_setup, basis, ctr_setup in itertools.product(tgt_setups, projections, ctr_setups):
            # Get the data for the current qubit pair and setup


            _data: CrossResonanceLengthData = data[(tuple(pair), target, tgt_setup, ctr_setup, basis)]
            raw_x = _data.duration
            min_x = np.min(raw_x)
            max_x = np.max(raw_x)
            y = _data.prob
            x = (raw_x - min_x) / (max_x - min_x)

            # Fit the data
            period = fallback_period(guess_period(x, y))
            pguess = [0.5, 0.5, period, 0, 0]
            try:
                popt, perr, pi_pulse_parameter = fit_length_function(x, y, pguess, sigma=_data.error, signal = False, x_limits=(min_x, max_x))

                _popt[tgt_setup, ctr_setup, basis] = popt
                _chi2[tgt_setup, ctr_setup, basis] = [chi2_reduced(y, rabi_length_function(raw_x, *popt),_data.error),
                                                               np.sqrt(2 / len(y))]
                
                if basis == BASIS[-1]:
                    _amplitude[ctr_setup] = amplitude[tuple(pair)]
                    _duration[ctr_setup]= [pi_pulse_parameter, perr[2] * (max_x - min_x) / 2]
                    print(pi_pulse_parameter)# popt[2] / 2


            except Exception as e:
                log.error(f"Error fitting data for {tuple(pair)} |{tgt_setup},{ctr_setup}>, <{basis}>: {e}")
                continue

        if STATES[0] in _duration and STATES[1] in _duration:
            fit_Jeff[tuple(pair)] = (1/_duration['X'][0]- 1/_duration['I'][0])*1e9/2
            print(f"Jeff for {tuple(pair)}: {fit_Jeff[tuple(pair)]*1e-6} MHz")

        fit_popt[tuple(pair)] = _popt
        fit_chi2[tuple(pair)] = _chi2
        fit_duration[tuple(pair)] = _duration
        fit_amplitude[tuple(pair)] = _amplitude

    ret =  CrossResonanceLengthResults(amplitude = fit_amplitude, 
                                       duration = fit_duration, 
                                       Jeff = fit_Jeff,
                                       chi2 = fit_chi2, 
                                       fitted_parameters = fit_popt)
    return ret

def _plot(data: CrossResonanceLengthData, fit: CrossResonanceLengthResults, target: QubitPairId
          ) -> tuple[list[go.Figure], str]:
    """Plotting function for Cross Resonance Gate Calibration."""
    target = tuple(target)
    fit_table = ""
    # Create a table with the results
    if fit is not None:
        fit_values = [1/np.array(fit.duration[target][ctr_setup][0])*1e9 for ctr_setup in STATES]+[fit.amplitude[target][STATES[1]], fit.Jeff[target]]
        fit_names = [f"Rabi freq. Ctrl: |{ctr_setup}> [Hz]" for ctr_setup in STATES] + ["Ctr Amp.", "J_{eff}"]

        fit_table = table_html(
            table_dict(
                qubit = target[0],
                names = fit_names,
                values = fit_values,
                display_error=False
            ),
        )
        
    return cr_plot(data,target,'duration',fit = fit), fit_table

cross_resonance_length = Routine(_acquisition, _fit, _plot, two_qubit_gates=True)
"""CrossResonance Duration Routine object."""
