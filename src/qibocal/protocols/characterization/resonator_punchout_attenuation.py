from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits

from .utils import find_min_msr, get_max_freq, get_points_with_max_freq, norm


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
    readout_attenuation: Dict[List[Tuple], str] = field(
        metadata=dict(update="readout_attenuation")
    )
    bare_frequency: Optional[Dict[List[Tuple], str]] = field(
        metadata=dict(update="bare_resonator_frequency")
    )
    lp_max_att: Dict[List[Tuple], str]
    lp_min_att: Dict[List[Tuple], str]
    hp_max_att: Dict[List[Tuple], str]
    hp_min_att: Dict[List[Tuple], str]


class ResonatorPunchoutAttenuationData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz", "attenuation": "dB"},
            options=[
                "qubit",
                "iteration",
                "resonator_type",
            ],
        )


def _acquisition(
    params: ResonatorPunchoutAttenuationParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
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
                    "resonator_type": len(freqs) * [platform.resonator_type],
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


def _fit(
    data: ResonatorPunchoutAttenuationData, fit_type="attenuation"
) -> ResonatorPunchoutAttenuationResults:
    """Fit frequency and attenuation at high and low power for a given resonator
        Args:
        data (DataUnits): data file with information on the feature response at each current point.
        qubits (list): qubits coupled to the resonator that we are probing.
        resonator_type (str): the type of readout resonator ['3D', '2D'].
        labels (list of str): list containing the lables of the quantities computed by this fitting method.
        fit_type (str): the type of punchout executed ['attenuation', 'amplitude'].

    Returns:
        data_fit (Data): Data file with labels and fit parameters (frequency at low and high power, attenuation range for low and highg power)
    """

    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()

    freq_lp_dict = {}
    lp_max_att_dict = {}
    lp_min_att_dict = {}
    freq_hp_dict = {}
    hp_max_att_dict = {}
    hp_min_att_dict = {}
    ro_att_dict = {}

    for qubit in qubits:
        averaged_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["i", "q", "qubit", "iteration"])
            .groupby(["frequency", fit_type], as_index=False)
            .mean()
        )
        try:
            normalised_data = averaged_data.groupby([fit_type], as_index=False)[
                ["MSR"]
            ].transform(norm)

            averaged_data_updated = averaged_data.copy()
            averaged_data_updated.update(normalised_data["MSR"])

            min_points = find_min_msr(averaged_data_updated, resonator_type, fit_type)
            vfunc = np.vectorize(lambda x: x.magnitude)
            min_points = vfunc(min_points)

            max_x = np.amax(min_points[:, 0])
            min_x = np.amin(min_points[:, 0])
            middle_x = (max_x + min_x) / 2

            hp_points = min_points[min_points[:, 0] < middle_x]
            lp_points = min_points[min_points[:, 0] >= middle_x]

            freq_hp = get_max_freq(hp_points)
            freq_lp = get_max_freq(lp_points)

            point_hp_max, point_hp_min = get_points_with_max_freq(min_points, freq_hp)
            point_lp_max, point_lp_min = get_points_with_max_freq(min_points, freq_lp)

            freq_lp = point_lp_max[0]
            lp_max_att = point_lp_max[1]
            lp_min_att = point_lp_min[1]
            freq_hp = point_hp_max[0]
            hp_max_att = point_hp_max[1]
            hp_min_att = point_hp_min[1]
            ro_att = round((lp_max_att + lp_min_att) / 2)
            ro_att = ro_att + 1 if ro_att % 2 == 1 else ro_att

        except:
            log.warning("resonator_punchout_fit: the fitting was not succesful")
            freq_lp = 0.0
            freq_hp = 0.0
            ro_att = 0.0
            lp_max_att = 0.0
            lp_min_att = 0.0
            hp_max_att = 0.0
            hp_min_att = 0.0

        freq_lp_dict[qubit] = freq_lp / 1e9
        freq_hp_dict[qubit] = freq_hp / 1e9
        ro_att_dict[qubit] = ro_att
        lp_max_att_dict[qubit] = lp_max_att
        lp_min_att_dict[qubit] = lp_min_att
        hp_max_att_dict[qubit] = hp_max_att
        hp_min_att_dict[qubit] = hp_min_att
        log.warning(
            f"max att: {lp_max_att} -  min att: {lp_min_att} -  readout_attenuation: {ro_att}"
        )

    return ResonatorPunchoutAttenuationResults(
        freq_lp_dict,
        ro_att_dict,
        freq_hp_dict,
        lp_max_att_dict,
        lp_min_att_dict,
        hp_max_att_dict,
        hp_min_att_dict,
    )


def _plot(
    data: ResonatorPunchoutAttenuationData,
    fit: ResonatorPunchoutAttenuationResults,
    qubit,
):
    figures = []
    fitting_report = ""

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

    if len(data) > 0:
        fig.add_trace(
            go.Scatter(
                x=[
                    fit.readout_frequency[qubit] * 1e9,
                    fit.readout_frequency[qubit] * 1e9,
                    fit.bare_frequency[qubit] * 1e9,
                    fit.bare_frequency[qubit] * 1e9,
                ],
                y=[
                    fit.lp_max_att[qubit],
                    fit.lp_min_att[qubit],
                    fit.hp_max_att[qubit],
                    fit.hp_min_att[qubit],
                ],
                mode="markers",
                marker=dict(
                    size=8,
                    color="gray",
                    symbol="circle",
                ),
            )
        )
        title_text = ""
        title_text += f"q{qubit}/r{report_n} | Resonator Frequency at Low Power:  {fit.readout_frequency[qubit] * 1e9} Hz.<br>"
        title_text += f"q{qubit}/r{report_n} | Low Power Attenuation Range: {fit.lp_max_att[qubit]} - {fit.lp_min_att[qubit]} db.<br>"
        title_text += f"q{qubit}/r{report_n} | Resonator Frequency at High Power: {fit.bare_frequency[qubit] * 1e9} Hz.<br>"
        title_text += f"q{qubit}/r{report_n} | High Power Attenuation Range: {fit.hp_max_att[qubit]} - {fit.hp_min_att[qubit]} db.<br>"

        fitting_report = fitting_report + title_text
    else:
        fitting_report = "No fitting data"

    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_punchout_attenuation = Routine(_acquisition, _fit, _plot)
