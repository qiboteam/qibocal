from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import lmfit
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color
from .utils import lorentzian_fit, lorentzian


@dataclass
class ResonatorSpectroscopyParameters(Parameters):
    freq_width: int
    freq_step: int
    amplitude: float
    nshots: int
    relaxation_time: int
    software_averages: int


@dataclass
class ResonatorSpectroscopyResults(Results):
    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="readout_frequency"))
    amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="readout_amplitude"))
    fitted_parameters: Dict[List[Tuple], List]


class ResonatorSpectroscopyData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: ResonatorSpectroscopyParameters
) -> ResonatorSpectroscopyData:
    # reload instrument settings from runcard
    platform.reload_settings()
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        ro_pulses[qubit].amplitude = params.amplitude
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        pulses=[ro_pulses[qubit] for qubit in qubits],
    )

    data = ResonatorSpectroscopyData()

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(params.software_averages):
        results = platform.sweep(
            sequence,
            sweeper,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )

        # retrieve the results for every qubit
        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            # store the results
            r = result.to_dict()
            r.update(
                {
                    "frequency[Hz]": delta_frequency_range + ro_pulses[qubit].frequency,
                    "qubit": len(delta_frequency_range) * [qubit],
                    "iteration": len(delta_frequency_range) * [iteration],
                    "resonator_type": len(delta_frequency_range)
                    * [platform.resonator_type],
                    "amplitude": len(delta_frequency_range)
                    * [ro_pulses[qubit].amplitude],
                }
            )
            data.add_data_from_dict(r)
    # finally, save the remaining data
    return data


def _fit(data: ResonatorSpectroscopyData) -> ResonatorSpectroscopyResults:
    return ResonatorSpectroscopyResults(*lorentzian_fit(data))


def _plot(data: ResonatorSpectroscopyData, fit: ResonatorSpectroscopyResults, qubit):
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
            + f"q{qubit}/r{report_n} | amplitude: {fit.amplitude[qubit][0]} <br>"
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


resonator_spectroscopy = Routine(_acquisition, _fit, _plot)
