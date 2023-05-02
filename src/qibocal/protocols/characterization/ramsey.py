from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

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
    n_osc: Optional[int] = 0


@dataclass
class RamseyResults(Results):
    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="drive_frequency"))
    t2: Dict[List[Tuple], str]
    delta_phys: Dict[List[Tuple], str]
    fitted_parameters: Dict[List[Tuple], List]


class RamseyData(DataUnits):
    def __init__(self, n_osc, t_max):
        super().__init__(
            name="data",
            quantities={"wait": "ns", "qubit_freqs": "Hz"},
            options=[
                "qubit",
            ],
        )

        self._n_osc = n_osc
        self._t_max = t_max

    @property
    def n_osc(self):
        return self._n_osc

    @property
    def t_max(self):
        return self._t_max


def _acquisition(
    params: RamseyParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
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
            qubit,
            start=RX90_pulses1[qubit].finish,
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

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include wait time and t_max
    data = RamseyData(params.n_osc, params.delay_between_pulses_end)

    # sweep the parameter
    for wait in waits:
        for qubit in qubits:
            RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
            ro_pulses[qubit].start = RX90_pulses2[qubit].finish
            if params.n_osc != 0:
                RX90_pulses2[qubit].relative_phase = (
                    RX90_pulses2[qubit].start
                    * (-2 * np.pi)
                    * (params.n_osc)
                    / params.delay_between_pulses_end
                )

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(sequence)
        for qubit, ro_pulse in ro_pulses.items():
            # average msr, phase, i and q over the number of shots defined in the runcard
            r = results[ro_pulse.serial].average.raw
            r.update(
                {
                    "wait[ns]": wait,
                    "qubit_freqs[Hz]": qubits[qubit].drive_frequency,
                    "qubit": qubit,
                }
            )
            data.add_data_from_dict(r)
    return data


def ramsey_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x * p4)


def _fit(data: RamseyData) -> RamseyResults:
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

            ft = np.fft.rfft(y)
            freqs = np.fft.rfftfreq(len(y), x[1] - x[0])
            mags = abs(ft)
            index = np.argmax(mags) if np.argmax(mags) != 0 else np.argmax(mags[1:]) + 1
            f = freqs[index] * 2 * np.pi
            p0 = [
                0.5,
                0.5,
                f,
                0,
                0,
            ]
            popt = curve_fit(
                ramsey_fit,
                x,
                y,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [0, 0, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
            )[0]
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] / (x_max - x_min),
                popt[3] - x_min * popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            delta_fitting = popt[2] / (2 * np.pi)
            # FIXME: check this formula
            delta_phys = +int((delta_fitting - data.n_osc / data.t_max) * 1e9)
            corrected_qubit_frequency = int(qubit_freq + delta_phys)
            t2 = 1.0 / popt[4]

        except Exception as e:
            log.warning(f"ramsey_fit: the fitting was not succesful. {e}")
            popt = [0] * 5
            t2 = 5.0
            corrected_qubit_frequency = int(qubit_freq)
            delta_phys = 0

        fitted_parameters[qubit] = popt
        corrected_qubit_frequencies[qubit] = corrected_qubit_frequency / 1e9
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
    fitting_report = ""

    qubit_data = data.df[data.df["qubit"] == qubit]

    fig.add_trace(
        go.Scatter(
            x=qubit_data["wait"].pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            marker_color=get_color(0),
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        ),
        row=1,
        col=1,
    )

    # add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(qubit_data["wait"]),
            max(qubit_data["wait"]),
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
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(1),
            ),
            row=1,
            col=1,
        )
        fitting_report = (
            fitting_report
            + (f"{qubit} | delta_frequency: {fit.delta_phys[qubit]:,.1f} Hz<br>")
            + (f"{qubit} | drive_frequency: {fit.frequency[qubit] * 1e9} Hz<br>")
            + (f"{qubit} | T2: {fit.t2[qubit]:,.0f} ns.<br><br>")
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


ramsey = Routine(_acquisition, _fit, _plot)
