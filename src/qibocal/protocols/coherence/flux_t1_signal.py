from dataclasses import dataclass, field
from typing import Union, Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from qibocal.auto.history import History
from qibocal.protocols.flux_dependence.qubit_flux_dependence import QubitFluxResults

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine

from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from qibocal.protocols.flux_dependence.utils import transmon_frequency

from ..utils import table_dict, table_html
from . import utils

from pathlib import Path

T1FluxType = np.dtype(
    [("qubit_frequency", np.float64), ("T1", np.float64)]
)
"""Custom dtype for T1Flux routines."""

#TODO: Make this parameters a dict of 4 parameters classes for each routine ???
@dataclass
class T1FluxSignalParameters(Parameters):
    """T1 runcard inputs."""

    folder_flux_map: Path
    
    biases_start: list[float]
    biases_end: list[float]
    biases_step: list[float]
    
    # Qubit spectroscopy
    freq_width: int
    """Width [Hz] for frequency sweep relative  to the qubit frequency."""
    freq_step: int
    """Frequency [Hz] step for sweep."""
    drive_duration: int
    """Drive pulse duration [ns]. Same for all qubits."""
    
    #Rabi amp signal
    min_amp_factor: float
    """Minimum amplitude multiplicative factor."""
    max_amp_factor: float
    """Maximum amplitude multiplicative factor."""
    step_amp_factor: float
    """Step amplitude multiplicative factor."""
    
    #T1 signal
    delay_before_readout_start: int
    """Initial delay before readout [ns]."""
    delay_before_readout_end: int
    """Final delay before readout [ns]."""
    delay_before_readout_step: int
    """Step delay before readout [ns]."""
    single_shot: bool = False
    """If ``True`` save single shot signal data."""
    
    # Optional qubit spectroscopy
    drive_amplitude: Optional[float] = None
    """Drive pulse amplitude (optional). Same for all qubits."""
    hardware_average: bool = True
    """By default hardware average will be performed."""
    # Optional rabi amp signal
    pulse_length: Optional[float] = None
    """RX pulse duration [ns]."""

@dataclass
class T1FluxSignalData(Data):
    """T1 acquisition outputs."""

    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def average(self):
        if len(next(iter(self.data.values())).shape) > 1:
            return utils.average_single_shots(self.__class__, self.data)
        return self


def _acquisition(
    params: T1FluxSignalParameters, platform: Platform, targets: list[QubitId]
) -> T1FluxSignalData:
    r"""Data acquisition for T1 experiment.
    """

    executor = Executor.create(name="myexec", platform=platform)
    from myexec import close, init, rabi_amplitude_signal, t1_signal, qubit_spectroscopy
    
    # init("test_T1vsFlux_map/qubit_flux_dependancy", force=True, targets=targets)
    
    data = T1FluxSignalData()
    
    for target in targets:
        
        # data_T1 = {}
        
        import pdb; pdb.set_trace()
        
        #TODO: Get this parameters from the platform
        results = QubitFluxResults.load(params.folder_flux_map)
        fitted_parameters = results.fitted_parameters
        params_qubit = fitted_parameters[target]
        fit_function=transmon_frequency
        
        i = 0
        for bias in params.biases:
            i +=1
            # Change the flux
            platform.qubits[target].flux.offset = bias
             
            # Change the qubit frequency
            qubit_frequency=fit_function(bias, **params_qubit) *1e9
            platform.qubits[target].drive_frequency = qubit_frequency
            platform.qubits[target].native_gates.RX.frequency = qubit_frequency
            
            init("test_T1vsFlux/flux{i}", force=True, targets=targets)

            #NOTE: Look and correct from the 1st estimate qubit frequency
            qubit_spectroscopy_output = qubit_spectroscopy(
                freq_width=params.freq_width,
                freq_step= params.freq_step,
                drive_duration= params.drive_duration,
                drive_amplitude=params.drive_amplitude,
            )
            
            qubit_spectroscopy_output.update_platform(platform)
            
            # Set maximun drive amplitude
            platform.qubits[target].native_gates.RX.amplitude = 0.5 #FIXME: For QM this should be 0.5
            platform.qubits[target].native_gates.RX.duration = params.pulse_length
            if qubit_spectroscopy_output.results.frequency:
                platform.qubits[target].native_gates.RX.frequency = qubit_spectroscopy_output.results.frequency[target]
            else:
                platform.qubits[target].native_gates.RX.frequency = qubit_frequency
                

            rabi_output = rabi_amplitude_signal(
                min_amp_factor=params.min_amp_factor,
                max_amp_factor=params.max_amp_factor,
                step_amp_factor=params.step_amp_factor,
                pulse_length=platform.qubits[target].native_gates.RX.duration,
            )
            
            if rabi_output.results.amplitude[target] > .5:
                print(f"Rabi fit has pi pulse amplitude {rabi_output.results.amplitude[target]}, greater than 0.5 not possible for QM. Skipping to next bias point.")
                continue
            rabi_output.update_platform(platform)
            
            t1_output = t1_signal(
                delay_before_readout_start=params.delay_before_readout_start,
                delay_before_readout_end=params.delay_before_readout_end,
                delay_before_readout_step=params.delay_before_readout_step,
            )            
            
            data.register_qubit(
                T1FluxType,
                (target, bias),
                dict(frequency=platform.qubits[target].native_gates.RX.frequency, t1=t1_output.results.t1[target][0]),
            )
            
            # close()
            report(executor.path, executor.history)
            executor.history = History()
            

    # stop and disconnect platform
    close()

    return data


def _fit():
    pass


def _plot(data: T1FluxSignalData, target: QubitId, fit=None):
    """Plotting function for T1 experiment."""
    
    import pdb; pdb.set_trace()
    
    data = data.average

    qubit_data = data[target]
    waits = qubit_data.wait


    figure = go.Figure()
    
    
    
    figure.add_trace(
        go.Scatter(
            x=data_qf,
            y=data_t1,
            opacity=1,
            name="Signal",
            showlegend=True,
            legendgroup="Signal",
        )
    )

    # last part
    figure.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHZ]",
        yaxis_title="Signal [a.u.]",
    )

    return figure


def _update(results, platform: Platform, target: QubitId):
    pass


t1flux_signal = Routine(_acquisition, _fit, _plot, _update)
"""T1 Flux Signal Routine object."""
