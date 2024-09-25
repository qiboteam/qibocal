from dataclasses import asdict, dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId, QubitPairId
from qibolab.sweeper import Parameter, Sweeper, SweeperType

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.protocols.utils import (
    HZ_TO_GHZ,
    lorentzian,
    lorentzian_fit,
    table_dict,
    table_html,
)


@dataclass
class DispersiveShiftParameters(Parameters):
    """Dispersive shift inputs."""

    freq_width: int
    """Width [Hz] for frequency sweep relative to the readout frequency [Hz]."""
    freq_step: int
    """Frequency step for sweep [Hz]."""

    @property
    def delta_frequency_range(self):
        return np.arange(
            -self.freq_width / 2, self.freq_width / 2, self.freq_step
        )

@dataclass
class DispersiveShiftResults(Results):
    """Dispersive shift outputs."""

    frequency_state_zero: dict[QubitId, float]
    """State zero frequency."""
    frequency_state_one: dict[QubitId, float]
    """State one frequency."""
    fitted_parameters_state_zero: dict[QubitId, list[float]]
    """Fitted parameters state zero."""
    fitted_parameters_state_one: dict[QubitId, list[float]]
    """Fitted parameters state one."""
    best_freq: dict[QubitId, float]
    """Readout frequency that maximizes the distance of ground and excited states in iq-plane"""

    @property
    def state_zero(self):
        return {key: value for key, value in asdict(self).items() if "zero" in key}

    @property
    def state_one(self):
        return {key: value for key, value in asdict(self).items() if "one" in key}


DispersiveShiftType = np.dtype(
    [
        ("freq", np.float64),
        ("i", np.float64),
        ("q", np.float64),
        ("signal", np.float64),
        ("phase", np.float64),
    ]
)
"""Custom dtype for dispersive shift."""


@dataclass
class DispersiveShiftData(Data):
    """Dispersive shift acquisition outputs."""

    resonator_type: str
    """Resonance type."""

    data: dict[tuple[QubitPairId, int], npt.NDArray[DispersiveShiftType]] = field(
        default_factory=dict
    )

STATES = [0,1]
def _acquisition(
    params: DispersiveShiftParameters, platform: Platform, targets: list[QubitPairId]
) -> DispersiveShiftData:
    r"""
    Data acquisition for the two qubit dispersive shift experiment.
    Perform spectroscopy on one of the qubits while the second qubit is in ground and excited state, showing
    the first qubit shift produced by the coupling between the them qubit. 
    This procedure is repeated useful to determine the qubit driven dispersive shift, which is a measure of the
    strength of the coupling between the qubits, given by $2\chi = J^2/\Delta$ 

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (Platform): Qibolab platform object
        targets (list): list of target qubit pairs to perform the experiment on

    """

    # create 2 sequences of pulses for the experiment:
    # sequence[0]: Q0: I - MZ
    #             Q1: I
    # sequence[1]: Q0: I - MZ
    #             Q1: RX
 
    sequence = PulseSequence()

    # Cannot be done in parallel because of possible overlaps between qubits
    for pair in targets: 
        target, control = pair

        for setup in STATES:
            # add a RX control pulse if the setup is |1>
            if setup == STATES[1]:
                rx_control = platform.create_RX_pulse(control, 0)
                sequence.add(rx_control)
                qd_pulse= platform.create_RX_pulse(target, rx_control.finish)  
            else:
                qd_pulse = platform.create_RX_pulse(target, 0)
            sequence.add(qd_pulse)

            ro_pulses = platform.create_qubit_readout_pulse(target, start=qd_pulse.finish)
            sequence.add(ro_pulses)

                            # create a DataUnits objects to store the results
            data = DispersiveShiftData(resonator_type=platform.resonator_type)
            sweeper = Sweeper(
                Parameter.frequency,
                params.delta_frequency_range,
                pulses=qd_pulse,
                type=SweeperType.OFFSET,
            )

            result = platform.sweep(
                sequence,
                params.execution_parameters,
                sweeper,
            )

            # retrieve the results for every qubit
            data.register_qubit(
                DispersiveShiftType,
                (pair, setup),
                dict(
                    freq=qd_pulse.frequency + params.delta_frequency_range,
                    signal=result.magnitude,
                    phase=result.phase,
                    i=result.voltage_i,
                    q=result.voltage_q,
                ),
            )
    return data


def _fit(data: DispersiveShiftData) -> DispersiveShiftResults:
    """Post-Processing for dispersive shift"""
    
    return DispersiveShiftResults()


def _plot(data: DispersiveShiftData, target: QubitPairId, fit: DispersiveShiftResults):
    """Plotting function for dispersive shift."""
    figures = []
    pair = target
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Signal [a.u.]",
            "phase [rad]",
        ),
    )
    # iterate over multiple data folders

    fitting_report = ""


    for setup in STATES:
        qubit_data = data[pair, setup]
        qubit_fit= fit[setup] if fit is not None else None

        opacity = 1
        frequencies = qubit_data.freq * HZ_TO_GHZ
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=qubit_data.signal,
                opacity=opacity,
                name=f"State {setup}",
                showlegend=True,
                legendgroup=f"State {setup}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=qubit_data.phase,
                opacity=opacity,
                showlegend=False,
                legendgroup=f"State {setup}",
            ),
            row=1,
            col=2,
        )

        if fit is not None:
            freqrange = np.linspace(
                min(frequencies),
                max(frequencies),
                2 * len(qubit_data),
            )
            params = qubit_fit[pair][setup]
            fig.add_trace(
                go.Scatter(
                    x=freqrange,
                    y=lorentzian(freqrange, *params),
                    name=f"State {setup} Fit",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=[
                    fit.best_freq[pair] * HZ_TO_GHZ,
                    fit.best_freq[pair] * HZ_TO_GHZ,
                ],
                y=[
                    np.min(np.concatenate((data[pair, STATES[0]].signal, data[pair, STATES[0]].signal))),
                    np.max(np.concatenate((data[pair, STATES[0]].signal, data[pair, STATES[1]].signal))),
                ],
                mode="lines",
                line=go.scatter.Line(color="orange", width=3, dash="dash"),
                name="Best frequency",
            ),
            row=1,
            col=1,
        )

        fig.add_vline(
            x=fit.best_freq[pair] * HZ_TO_GHZ,
            line=dict(color="orange", width=3, dash="dash"),
            row=1,
            col=1,
        )
        fitting_report = table_html(
            table_dict(
                pair,
                [
                    "State Zero Frequency [Hz]",
                    "State One Frequency [Hz]",
                    "Chi [Hz]",
                    "Best Frequency [Hz]",
                ],
                np.round(
                    [
                        qubit_fit[pair][STATES[0]],
                        qubit_fit[pair][STATES[1]],
                        (
                            qubit_fit[pair][STATES[0]]
                            - qubit_fit[pair][STATES[1]]
                        )
                        / 2,
                        fit.best_freq[pair],
                    ]
                ),
            )
        )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )

    figures.append(fig)

    return figures, fitting_report


# def _update(results: DispersiveShiftResults, platform: Platform, target: QubitId):
#     update.readout_frequency(results.best_freq[target], platform, target)


cross_resonance_dispersive_shift = Routine(_acquisition, _fit, _plot)
"""Dispersive shift Routine object."""
