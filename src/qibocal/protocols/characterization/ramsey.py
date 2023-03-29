from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from ...auto.operation import Parameters, Qubits, Results, Routine
from ...config import log
from ...data import DataUnits
from ...plots.utils import get_color


@dataclass
class RamseyParameters(Parameters):
    delay_between_pulses_start: int
    delay_between_pulses_end: list
    delay_between_pulses_step: int
    software_averages: int = 1


@dataclass
class RamseyResults(Results):
    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="drive_frequency"))
    t2: Dict[List[Tuple], str] = field(metadata=dict(update="t2"))
    # TODO: perhaps this is not necessary for the runcard
    delta_phys: Dict[List[Tuple], str]  # = field(metadata=dict(update="freq_detuning"))
    fitted_parameters: Dict[List[Tuple], List]


class RamseyData(DataUnits):
    def __init__(self):
        super().__init__(
            name="data",
            quantities={"wait": "ns", "t_max": "ns", "qubit_freqs": "Hz"},
            options=[
                "qubit",
                "iteration",
                "sampling_rate",
                "offset_freq",
                "resonator_type",
            ],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: RamseyParameters
) -> RamseyData:
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )
    sampling_rate = platform.sampling_rate  # TODO: we may to reset this
    t_max = params.delay_between_pulses_end

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = RamseyData()

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(params.software_averages):
        # sweep the parameter
        for wait in waits:
            for qubit in qubits:
                RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
                ro_pulses[qubit].start = RX90_pulses2[qubit].finish

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)
            for qubit, ro_pulse in ro_pulses.items():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].to_dict(average=True)
                r.update(
                    {
                        "wait[ns]": wait,
                        "t_max[ns]": t_max,
                        "qubit_freqs[Hz]": qubits[qubit].drive_frequency,
                        "qubit": qubit,
                        "iteration": iteration,
                        "sampling_rate": sampling_rate,
                        "offset_freq": 0,  # pars.n_osc / t_max * sampling_rate  # Hz
                        "resonator_type": platform.resonator_type,
                    }
                )
                data.add(r)
            count += 1
    return data


def ramsey_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x * p4)


def _fit(data: RamseyData) -> RamseyResults:  # TODO: put Platform as input
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.
    Args:
        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the Ramsey model
        y (str): name of the output values for the Ramsey model
        qubits (list): A list with the IDs of the qubits
        qubits_freq (float): frequency of the qubit
        sampling_rate (float): Platform sampling rate
        offset_freq (float): Total qubit frequency offset. It contains the artificial detunning applied
                             by the experimentalist + the inherent offset in the actual qubit frequency stored in the runcard.
        labels (list of str): list containing the lables of the quantities computed by this fitting method.
    Returns:
        A ``Data`` object with the following keys
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **labels[0]**: Physical detunning of the actual qubit frequency
            - **labels[1]**: New qubit frequency after correcting the actual qubit frequency with the detunning calculated (labels[0])
            - **labels[2]**: T2
            - **qubit**: The qubit being tested
    """
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()
    sampling_rate = data.df["sampling_rate"].unique()
    offset_freq = data.df["offset_freq"].unique()
    t2s = {}
    corrected_qubit_frequencies = {}
    freqs_detuing = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]
        voltages = qubit_data_df["MSR"].pint.to("uV").pint.magnitude
        times = qubit_data_df["wait"].pint.to("ns").pint.magnitude
        qubit_freq = qubit_data_df["qubit_freqs"].pint.to("Hz").pint.magnitude.unique()
        try:
            y_max = np.max(voltages.values)
            y_min = np.min(voltages.values)
            y = (voltages.values - y_min) / (y_max - y_min)
            x_max = np.max(times.values)
            x_min = np.min(times.values)
            x = (times.values - x_min) / (x_max - x_min)
            if resonator_type == "3D":
                index = np.argmin(y)
            else:
                index = np.argmax(y)

            p0 = [
                np.mean(y),
                y_max - y_min,
                0.5 / x[index],
                np.pi / 2,
                0,
            ]
            popt = curve_fit(ramsey_fit, x, y, method="lm", p0=p0)[0]
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] / (x_max - x_min),
                popt[3] - x_min * popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            delta_fitting = popt[2] / 2 * np.pi
            delta_phys = int((delta_fitting * sampling_rate) - offset_freq)
            corrected_qubit_frequency = int(qubit_freq + delta_phys)
            t2 = 1.0 / popt[4]

        except:
            log.warning("ramsey_fit: the fitting was not succesful")
            popt = [0] * 5
            t2 = 5.0
            corrected_qubit_frequency = int(qubit_freq)
            delta_phys = 0

        fitted_parameters[qubit] = popt
        corrected_qubit_frequencies[qubit] = corrected_qubit_frequency
        t2s[qubit] = t2
        freqs_detuing[qubit] = delta_phys

    return RamseyResults(
        corrected_qubit_frequencies, t2s, freqs_detuing, fitted_parameters
    )


def _plot(data: RamseyData, fit: RamseyResults, qubit):
    figures = []

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=("MSR (V)",),
    )
    report_n = 0
    fitting_report = ""

    data.df = data.df[data.df["qubit"] == qubit]
    iterations = data.df["iteration"].unique()
    waits = data.df["wait"].pint.to("ns").pint.magnitude
    if len(iterations) > 1:
        opacity = 0.3
    else:
        opacity = 1
    for iteration in iterations:
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data["wait"].pint.magnitude,
                y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=not bool(iteration),
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=1,
        )

    if len(iterations) > 1:
        data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
        fig.add_trace(
            go.Scatter(
                x=waits,
                y=data.df.groupby("wait")["MSR"]
                .mean()
                .pint.to("uV")
                .pint.magnitude,  # pylint: disable=E1101
                marker_color=get_color(report_n),
                name=f"q{qubit}/r{report_n}: Average",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=1,
        )

    # add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(data.df["wait"]),
            max(data.df["wait"]),
            2 * len(data),
        )

        fig.add_trace(
            go.Scatter(
                x=waitrange.magnitude,
                y=ramsey_fit(
                    waitrange.magnitude,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                    float(fit.fitted_parameters[qubit][4]),
                ),
                name=f"q{qubit}/r{report_n} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(4 * report_n + 2),
            ),
            row=1,
            col=1,
        )
        fitting_report = (
            fitting_report
            + (
                f"q{qubit}/r{report_n} | delta_frequency: {fit.delta_phys[qubit]:,.0f} Hz<br>"
            )
            + (
                f"q{qubit}/r{report_n} | drive_frequency: {fit.frequency[qubit]:,.0f} Hz<br>"
            )
            + (f"q{qubit}/r{report_n} | T2: {fit.t2[qubit]:,.0f} ns.<br><br>")
        )
    report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey = Routine(_acquisition, _fit, _plot)
