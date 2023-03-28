from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.data import DataUnits

from .utils import lorentzian_fit, spectroscopy_plot

# resonator_spectroscopy_flux: # wide
#   freq_width: 2_000_000
#   freq_step: 200_000
#   bias_width: 0.03 # absolute min is -40 mA
#   bias_step:  0.003
#   fluxlines: qubits
#   software_averages: 2
#   points: 1


@dataclass
class ResonatorFluxParameters(Parameters):
    freq_width: int
    freq_step: int
    bias_width: float
    bias_step: float
    fluxlines: int
    nshots: int
    relaxation_time: int
    software_averages: int


@dataclass
class ResonatorFluxResults(Results):
    ...
    # sweetspot: Dict[List[Tuple], str] = field(metadata=dict(update="sweetspot"))
    # fitted_parameters : Dict[List[Tuple], List]


class ResonatorFluxData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz", "bias": "V"},
            options=["qubit", "fluxline", "iteration"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: ResonatorFluxParameters
) -> ResonatorFluxData:
    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()
    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    # flux bias
    if params.fluxlines == "qubits":
        params.fluxlines = list(qubits.values())

    # print(params.fluxlines[0].flux.offset)

    delta_bias_range = np.arange(
        -params.bias_width / 2, params.bias_width / 2, params.bias_step
    )
    bias_sweeper = Sweeper(Parameter.bias, delta_bias_range, qubits=params.fluxlines)
    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and flux bias
    data = ResonatorFluxData()

    # repeat the experiment as many times as defined by software_averages
    for iteration in range(params.software_averages):
        results = platform.sweep(
            sequence,
            bias_sweeper,
            freq_sweeper,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )

        # retrieve the results for every qubit
        for qubit, fluxline in zip(qubits, params.fluxlines):
            result = results[ro_pulses[qubit].serial]

            biases = np.repeat(
                delta_bias_range, len(delta_frequency_range)
            ) + platform.get_bias(fluxline.name)
            freqs = np.array(
                len(delta_bias_range)
                * list(delta_frequency_range + ro_pulses[qubit].frequency)
            ).flatten()
            # store the results
            r = {k: v.ravel() for k, v in result.to_dict().items()}
            r.update(
                {
                    "frequency[Hz]": freqs,
                    "bias[V]": biases,
                    "qubit": len(freqs) * [qubit],
                    "fluxline": len(freqs) * [fluxline.name],
                    "iteration": len(freqs) * [iteration],
                }
            )
            data.add_data_from_dict(r)

    return data


def _fit(data: ResonatorFluxData) -> ResonatorFluxResults:
    return ResonatorFluxResults()


def _plot(data: ResonatorFluxData, fit: ResonatorFluxResults, qubit):
    figures = []
    fitting_report = "No fitting data"

    report_n = 0

    data.df = data.df[data.df["qubit"] == qubit]

    iterations = data.df["iteration"].unique()
    fluxlines = data.df["fluxline"].unique()
    frequencies = data.df["frequency"].unique()

    if len(fluxlines) > 1:
        fig = make_subplots(
            rows=1,
            cols=len(fluxlines),
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=tuple(
                [f"MSR [V] - fluxline {fluxline}" for fluxline in fluxlines]
            ),
        )

        for fluxline_n, fluxline in enumerate(fluxlines):
            fluxline_df = data.df[data.df["fluxline"] == fluxline]
            fluxline_df = fluxline_df.drop(
                columns=["i", "q", "qubit", "fluxline", "iteration"]
            )

            fluxline_df = fluxline_df.groupby(
                ["frequency", "bias"], as_index=False
            ).mean()

            fig.add_trace(
                go.Heatmap(
                    x=fluxline_df["frequency"].pint.to("GHz").pint.magnitude,
                    y=fluxline_df["bias"].pint.to("V").pint.magnitude,
                    z=fluxline_df["MSR"].pint.to("uV").pint.magnitude,
                    showscale=False,
                ),
                row=1 + report_n,
                col=1 + fluxline_n,
            )
            fig.update_xaxes(
                title_text=f"q{qubit}/r{report_n}: Frequency (GHz)",
                row=1 + report_n,
                col=1 + fluxline_n,
            )

    elif len(fluxlines) == 1:
        fig = make_subplots(
            rows=1,
            cols=2,
            horizontal_spacing=0.1,
            vertical_spacing=0.1,
            subplot_titles=(
                f"MSR [V] - fluxline {fluxlines[0]}",
                f"Phase [rad] - fluxline {fluxlines[0]}",
            ),
        )

        fluxline_df = data.df[data.df["fluxline"] == fluxlines[0]]
        fluxline_df = fluxline_df.drop(
            columns=["i", "q", "qubit", "fluxline", "iteration"]
        )

        fluxline_df = fluxline_df.groupby(["frequency", "bias"], as_index=False).mean()

        fig.add_trace(
            go.Heatmap(
                x=fluxline_df["frequency"].pint.to("GHz").pint.magnitude,
                y=fluxline_df["bias"].pint.to("V").pint.magnitude,
                z=fluxline_df["MSR"].pint.to("uV").pint.magnitude,
                colorbar_x=0.46,
            ),
            row=1 + report_n,
            col=1,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)",
            row=1 + report_n,
            col=1,
        )

        fig.add_trace(
            go.Heatmap(
                x=fluxline_df["frequency"].pint.to("GHz").pint.magnitude,
                y=fluxline_df["bias"].pint.to("V").pint.magnitude,
                z=fluxline_df["phase"].pint.to("rad").pint.magnitude,
                colorbar_x=1.01,
            ),
            row=1 + report_n,
            col=2,
        )
        fig.update_xaxes(
            title_text=f"q{qubit}/r{report_n}: Frequency (Hz)",
            row=1 + report_n,
            col=2,
        )
        fig.update_yaxes(title_text="Bias (V)", row=1 + report_n, col=2)

    fig.update_yaxes(title_text="Bias (V)", row=1 + report_n, col=1)
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_flux = Routine(_acquisition, _fit, _plot)
