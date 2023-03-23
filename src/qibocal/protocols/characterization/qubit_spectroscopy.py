from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color
from .utils import lorentzian_fit, lorentzian

@dataclass
class QubitSpectroscopyParameters(Parameters):
    freq_width: int
    freq_step: int
    drive_duration: int
    drive_amplitude: Optional[float] = None
    nshots: int = 1024
    relaxation_time: int = 50
    software_averages: int = 1

@dataclass
class QubitSpectroscopyResults(Results):
    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="drive_frequency"))
    amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    fitted_parameters: Dict[List[Tuple], List]


class QubitSpectroscopyData(DataUnits):
    def __init__(self):
        super().__init__(
            name="data",
            quantities={"frequency": "Hz"},
            options=["qubit", "iteration", "resonator_type", "amplitude"],

        )

def _acquisition(
    platform: AbstractPlatform,
    qubits: Qubits,
    params: QubitSpectroscopyParameters
) -> QubitSpectroscopyData:
   
    # reload instrument settings from runcard
    platform.reload_settings()
    # create a sequence of pulses for the experiment:
    # long drive probing pulse - MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    qd_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(
            qubit, start=0, duration=params.drive_duration
        )
        if params.drive_amplitude is not None:
            qd_pulses[qubit].amplitude = params.drive_amplitude
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(-params.freq_width // 2, params.freq_width // 2, params.freq_step)
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit frequency
    data = QubitSpectroscopyData()

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(params.software_averages):
        results = platform.sweep(
            sequence, sweeper, nshots = params.nshots, relaxation_time = params.relaxation_time
        )

        # retrieve the results for every qubit
        for qubit, ro_pulse in ro_pulses.items():
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulse.serial]
            r = result.to_dict()
            # store the results
            r.update(
                {
                    "frequency[Hz]": delta_frequency_range + qd_pulses[qubit].frequency,
                    "qubit": len(delta_frequency_range) * [qubit],
                    "iteration": len(delta_frequency_range) * [iteration],
                    "amplitude": len(delta_frequency_range)
                    * [ro_pulses[qubit].amplitude],
                }
            )
            data.add_data_from_dict(r)

        # finally, save the remaining data and fits
    return data

def _fit (data: QubitSpectroscopyData) -> QubitSpectroscopyResults:
    return QubitSpectroscopyResults(*lorentzian_fit(data))

def _plot(data: QubitSpectroscopyData, fit: QubitSpectroscopyResults, qubit):
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (uV)",
            "phase (rad)",
        ),
    )
    data.df = data.df[data.df["qubit"] == qubit].drop(columns=["i", "q", "qubit"])
    iterations = data.df["iteration"].unique()

    fitting_report = ""
    report_n = 0

    if len(iterations) > 1:
        opacity = 0.3
    else:
        opacity = 1
    for iteration in iterations:
        frequencies = data.df["frequency"].pint.to("GHz").pint.magnitude.unique()
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data["frequency"].pint.to("GHz").pint.magnitude,
                y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(2 * report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}: Data",
                showlegend=not bool(iteration),
                legendgroup=f"q{qubit}/r{report_n}: Data",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=iteration_data["frequency"].pint.to("GHz").pint.magnitude,
                y=iteration_data["phase"].pint.to("rad").pint.magnitude,
                marker_color=get_color(2 * report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}: Data",
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}: Data",
            ),
            row=1,
            col=2,
        )
    if len(iterations) > 1:
        data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=data.df.groupby("frequency")["MSR"]
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                marker_color=get_color(2 * report_n),
                name=f"q{qubit}/r{report_n}: Average",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=data.df.groupby("frequency")["phase"]
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                marker_color=get_color(2 * report_n),
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=2,
        )
    if len(data) > 0:
        freqrange = np.linspace(
            min(frequencies),
            max(frequencies),
            2 * len(frequencies),
        )
        params = fit.fitted_parameters[qubit]

        fig.add_trace(
            go.Scatter(
                x=freqrange,
                y=lorentzian(freqrange, **params),
                name=f"q{qubit}/r{report_n} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(4 * report_n + 2),
            ),
            row=1,
            col=1,
        )
        fitting_report = (
            fitting_report
            + f"q{qubit}/r{report_n} | frequency: {fit.frequency[qubit]*1e9:,.0f} Hz<br>"
            + f"q{qubit}/r{report_n} | amplitude: {fit.amplitude[qubit]} <br>"
        )
        
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Frequency (GHz)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Frequency (GHz)",
        yaxis2_title="Phase (rad)",
    )
    figures.append(fig)

    return figures, fitting_report

qubit_spectroscopy = Routine(_acquisition, _fit, _plot)
