from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper

from ...auto.operation import Parameters, Qubits, Results, Routine
from ...config import log
from ...data import DataUnits
from .resonator_spectroscopy import ResonatorSpectroscopyResults
from .utils import find_min_msr, get_max_freq, get_points_with_max_freq, norm


@dataclass
class ResonatorPunchoutParameters(Parameters):
    freq_width: int
    freq_step: int
    min_amp_factor: float
    max_amp_factor: float
    step_amp_factor: float
    nshots: int
    relaxation_time: int
    software_averages: int = 1


@dataclass
class ResonatorPunchoutResults(Results):
    readout_frequency: Dict[List[Tuple], str] = field(
        metadata=dict(update="readout_frequency")
    )
    # amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="readout_amplitude"))
    bare_frequency: Optional[Dict[List[Tuple], str]] = field(
        metadata=dict(update="bare_resonator_frequency")
    )
    lp_max_att: Dict[List[Tuple], str]
    lp_min_att: Dict[List[Tuple], str]
    hp_max_att: Dict[List[Tuple], str]
    hp_min_att: Dict[List[Tuple], str]


class ResonatorPunchoutData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"frequency": "Hz", "amplitude": "dimensionless"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: ResonatorPunchoutParameters
) -> ResonatorPunchoutData:
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

    # amplitude
    amplitude_range = np.arange(
        params.min_amp_factor, params.max_amp_factor, params.step_amp_factor
    )
    amp_sweeper = Sweeper(
        Parameter.amplitude, amplitude_range, [ro_pulses[qubit] for qubit in qubits]
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include resonator frequency and attenuation
    data = ResonatorPunchoutData()

    # repeat the experiment as many times as defined by software_averages
    amps = np.repeat(amplitude_range, len(delta_frequency_range))
    for iteration in range(params.software_averages):
        results = platform.sweep(
            sequence,
            amp_sweeper,
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
                len(amplitude_range) * list(delta_frequency_range + ro_pulse.frequency)
            ).flatten()
            r = {k: v.ravel() for k, v in result.raw.items()}
            r.update(
                {
                    "frequency[Hz]": freqs,
                    "amplitude[dimensionless]": amps,
                    "qubit": len(freqs) * [qubit],
                    "resonator_type": len(freqs) * [platform.resonator_type],
                    "iteration": len(freqs) * [iteration],
                }
            )
            data.add_data_from_dict(r)

        # save data
    return data
    # TODO: calculate and save fit


def _fit(data: ResonatorPunchoutData, fit_type="amplitude") -> ResonatorPunchoutResults:
    # def punchout_fit(data: DataUnits, fit_type: str) -> Results:
    # def punchout_fit(data, qubits, resonator_type, labels, fit_type):
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

    # data_fit = Data(
    #     name=f"fits",
    #     quantities=[
    #         labels[0],
    #         labels[1],
    #         labels[2],
    #         labels[3],
    #         labels[4],
    #         labels[5],
    #         "qubit",
    #     ],
    # )
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()

    freq_lp_dict = {}
    lp_max_att_dict = {}
    lp_min_att_dict = {}
    freq_hp_dict = {}
    hp_max_att_dict = {}
    hp_min_att_dict = {}

    for qubit in qubits:
        averaged_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["i", "q", "qubit", "iteration"])
            .groupby(["frequency", fit_type], as_index=False)
            .mean()
        )
        # try:
        normalised_data = averaged_data.groupby([fit_type], as_index=False)[
            ["MSR"]
        ].transform(norm)

        averaged_data_updated = averaged_data.copy()
        averaged_data_updated.update(normalised_data["MSR"])

        min_points = find_min_msr(averaged_data_updated, resonator_type, fit_type)
        min_points = [[q.magnitude for q in tupla] for tupla in min_points]
        x = [point[0] for point in min_points]
        y = [point[1] for point in min_points]

        max_x = x[np.argmax(x)]
        min_x = x[np.argmin(x)]
        middle_x = (max_x + min_x) / 2

        hp_points = [point for point in min_points if point[0] < middle_x]
        lp_points = [point for point in min_points if point[0] >= middle_x]

        freq_hp = get_max_freq(hp_points, middle_x)
        freq_lp = get_max_freq(lp_points, middle_x)

        point_hp_max, point_hp_min = get_points_with_max_freq(min_points, freq_hp)
        point_lp_max, point_lp_min = get_points_with_max_freq(min_points, freq_lp)
        print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFF", point_hp_max, type(point_hp_max[0]))
        freq_lp = point_lp_max[0]
        lp_max_att = point_lp_max[1]
        lp_min_att = point_lp_min[1]
        freq_hp = point_hp_max[0]
        hp_max_att = point_hp_max[1]
        hp_min_att = point_hp_min[1]

        # except:
        #     log.warning("resonator_punchout_fit: the fitting was not succesful")
        #     freq_lp = 0.
        #     lp_max_att = 0.
        #     lp_min_att = 0.
        #     freq_hp = 0.
        #     hp_max_att = 0.
        #     hp_min_att = 0.
        freq_lp_dict[qubit] = freq_lp
        lp_max_att_dict[qubit] = lp_max_att
        lp_min_att_dict[qubit] = lp_min_att
        freq_hp_dict[qubit] = freq_hp
        hp_max_att_dict[qubit] = hp_max_att
        hp_min_att_dict[qubit] = hp_min_att

    return ResonatorPunchoutResults(
        freq_hp_dict,
        freq_lp_dict,
        lp_max_att_dict,
        lp_min_att_dict,
        hp_max_att_dict,
        hp_min_att_dict,
    )


def _plot(data: ResonatorPunchoutData, fit: ResonatorPunchoutResults, qubit):
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
    amplitudes = data.df["amplitude"].pint.to("dimensionless").pint.magnitude.unique()
    averaged_data = (
        data.df.drop(columns=["qubit", "iteration"])
        .groupby(["frequency", "amplitude"], as_index=False)
        .mean()
    )

    def norm(x):
        x_mags = x.pint.to("V").pint.magnitude
        return (x_mags - np.min(x_mags)) / (np.max(x_mags) - np.min(x_mags))

    normalised_data = averaged_data.groupby(["amplitude"], as_index=False)[
        ["MSR"]
    ].transform(norm)

    fig.add_trace(
        go.Heatmap(
            x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
            y=averaged_data["amplitude"].pint.to("dimensionless").pint.magnitude,
            z=normalised_data["MSR"],
            colorbar_x=0.46,
        ),
        row=1 + report_n,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=1
    )
    fig.update_yaxes(title_text="Amplitude", row=1 + report_n, col=1)
    fig.add_trace(
        go.Heatmap(
            x=averaged_data["frequency"].pint.to("Hz").pint.magnitude,
            y=averaged_data["amplitude"].pint.to("dimensionless").pint.magnitude,
            z=averaged_data["phase"].pint.to("rad").pint.magnitude,
            colorbar_x=1.01,
        ),
        row=1 + report_n,
        col=2,
    )
    fig.update_xaxes(
        title_text=f"q{qubit}/r{report_n}: Frequency (Hz)", row=1 + report_n, col=2
    )
    fig.update_yaxes(title_text="Amplitude", row=1 + report_n, col=2)
    fig.update_layout(
        showlegend=False,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
    )

    figures.append(fig)

    return figures, fitting_report


resonator_punchout = Routine(_acquisition, _fit, _plot)
