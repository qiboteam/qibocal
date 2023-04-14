from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from ...auto.operation import Parameters, Qubits, Results, Routine
from ...data import DataUnits


@dataclass
class ResonatorPunchoutAttenuationParameters(Parameters):
    freq_width: int
    freq_step: int
    min_att: int
    max_att: int
    step_att: int
    nshots: int
    relaxation_time: int
    software_averages: int = 1


@dataclass
class ResonatorPunchoutAttenuationResults(Results):
    readout_frequency: Dict[List[Tuple], str] = field(
        metadata=dict(update="readout_frequency")
    )
    attenuation: Dict[List[Tuple], str] = field(
        metadata=dict(update="readout_attenuation")
    )
    fitted_parameters: Dict[List[Tuple], List]
    bare_frequency: Optional[Dict[List[Tuple], str]] = field(
        metadata=dict(update="bare_resonator_frequency")
    )


class ResonatorPunchoutAttenuationData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz", "attenuation": "dB"},
            options=["qubit", "iteration"],
        )


def _acquisition(
    platform: AbstractPlatform,
    qubits: Qubits,
    params: ResonatorPunchoutAttenuationParameters,
) -> ResonatorPunchoutAttenuationData:
    # create a sequence of pulses for the experiment:
    # MZ

    # taking advantage of multiplexing, apply the same set of gates to all qubits in parallel
    sequence = PulseSequence()

    ro_pulses = {}
    for qubit in qubits:
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(qubit, start=0)
        sequence.add(ro_pulses[qubit])

    # define the parameters to sweep and their range:
    # resonator frequency
    delta_frequency_range = np.arange(
        -params.freq_width // 2, params.freq_width // 2, params.freq_step
    )
    freq_sweeper = Sweeper(
        Parameter.frequency,
        delta_frequency_range,
        [ro_pulses[qubit] for qubit in qubits],
    )

    # attenuation
    attenuation_range = np.arange(params.min_att, params.max_att, params.step_att)
    att_sweeper = Sweeper(
        Parameter.attenuation, attenuation_range, qubits=[qubit for qubit in qubits]
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = ResonatorPunchoutAttenuationData()

    # repeat the experiment as many times as defined by software_averages
    att = np.repeat(attenuation_range, len(delta_frequency_range))
    for iteration in range(params.software_averages):
        results = platform.sweep(
            sequence,
            att_sweeper,
            freq_sweeper,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )

        # retrieve the results for every qubit
        for qubit, ro_pulse in ro_pulses.items():
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulse.serial]
            # store the results
            freqs = np.array(
                len(attenuation_range)
                * list(delta_frequency_range + ro_pulse.frequency)
            ).flatten()
            r = {k: v.ravel() for k, v in result.raw.items()}
            r.update(
                {
                    "frequency[Hz]": freqs,
                    "attenuation[dB]": att,
                    "qubit": len(freqs) * [qubit],
                    "iteration": len(freqs) * [iteration],
                }
            )
            data.add_data_from_dict(r)

        # Temporary fixe to force to reset the attenuation to the original value in qblox
        # sweeper method returning to orig value not working for attenuation
        # After fitting punchout the reload_settings will be called automatically
        platform.reload_settings()
        # save data
    return data
    # TODO: calculate and save fit


def _fit(data: ResonatorPunchoutAttenuationData) -> ResonatorPunchoutAttenuationResults:
    return ResonatorPunchoutAttenuationResults({}, {}, {}, {})


def _plot(
    data: ResonatorPunchoutAttenuationData,
    fit: ResonatorPunchoutAttenuationResults,
    qubit,
):
    figures = []
    fitting_report = "No fitting data"

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=(
            "Normalised MSR",
            "phase (rad)",
        ),
    )

    report_n = 0

    data.df = data.df[data.df["qubit"] == qubit]

    iterations = data.df["iteration"].unique()
    frequencies = data.df["frequency"].pint.to("Hz").pint.magnitude.unique()
    attenuations = data.df["attenuation"].pint.to("dB").pint.magnitude.unique()
    averaged_data = (
        data.df.drop(columns=["qubit", "iteration"])
        .groupby(["frequency", "attenuation"], as_index=False)
        .mean()
    )

    def norm(x):
        x_mags = x.pint.to("V").pint.magnitude
        return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))

    normalised_data = averaged_data.groupby(["attenuation"], as_index=False)[
        ["MSR"]
    ].transform(norm)

    fig.add_trace(
        go.Heatmap(
            x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
            y=averaged_data["attenuation"].pint.to("dB").pint.magnitude,
            z=normalised_data["MSR"],
            colorbar_x=0.46,
        ),
        row=1 + report_n,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=1
    )
    fig.update_yaxes(title_text="Attenuation", row=1 + report_n, col=1)
    fig.add_trace(
        go.Heatmap(
            x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
            y=averaged_data["attenuation"].pint.to("dB").pint.magnitude,
            z=averaged_data["phase"].pint.to("rad").pint.magnitude,
            colorbar_x=1.01,
        ),
        row=1 + report_n,
        col=2,
    )
    fig.update_xaxes(
        title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=2
    )
    fig.update_yaxes(title_text="Attenuation", row=1 + report_n, col=2)
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_punchout_attenuation = Routine(_acquisition, _fit, _plot)
