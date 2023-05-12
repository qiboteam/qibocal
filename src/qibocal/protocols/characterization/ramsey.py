from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit
from scipy.stats import chisquare

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color


@dataclass
class RamseyParameters(Parameters):
    """Ramsey runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    n_osc: Optional[int] = 0
    """Number of oscillations to induce detuning (optional).
        If 0 standard Ramsey experiment is performed."""


@dataclass
class RamseyResults(Results):
    """Ramsey outputs."""

    frequency: Dict[List[Tuple], str] = field(metadata=dict(update="drive_frequency"))
    """Drive frequency for each qubit."""
    t2: Dict[List[Tuple], str]
    """T2 for each qubit (ns)."""
    t2_err: Dict[List[Tuple], str]
    """T2 error for each qubit (ns)."""
    delta_phys: Dict[List[Tuple], str]
    """Drive frequency correction for each qubit."""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitting output."""


class RamseyData(DataUnits):
    """Ramsey acquisition outputs."""

    def __init__(self, n_osc, t_max):
        super().__init__(
            name="data",
            quantities={"wait": "ns", "qubit_freqs": "Hz"},
            options=["qubit", "errors"],
        )

        self._n_osc = n_osc
        self._t_max = t_max

    @property
    def n_osc(self):
        """Number of oscillations for detuning."""
        return self._n_osc

    @property
    def t_max(self):
        """Final delay between RX(pi/2) pulses in ns."""
        return self._t_max


def _acquisition(
    params: RamseyParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> RamseyData:
    """Data acquisition for Ramsey Experiment (detuned)."""
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
    import matplotlib.pyplot as plt  # TODO: remove plotting lines

    fig, (ax1, ax2) = plt.subplots(2, 1)  # TODO: remove plotting lines
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
            msr_raw = results[ro_pulse.serial].raw["MSR[V]"] * 1e6
            if wait % 10 == 0:
                ax1.scatter(
                    [wait] * len(msr_raw), msr_raw, s=1, color=f"C{int(wait/10)}"
                )
                ax1.scatter(
                    wait,
                    np.average(msr_raw),
                    s=10,
                    color=f"C{int(wait/10)}",
                    marker="+",
                )
                if wait < 150:
                    ax2.hist(
                        msr_raw,
                        bins=100,
                        histtype="step",
                        density=True,
                        color=f"C{int(wait/10)}",
                    )
            error = np.std(msr_raw) / np.sqrt(len(msr_raw))
            r.update(
                {
                    "wait[ns]": wait,
                    "qubit_freqs[Hz]": qubits[qubit].drive_frequency,
                    "qubit": qubit,
                    "errors": error,
                }
            )
            data.add_data_from_dict(r)

    plt.savefig("ramsey.pdf")
    return data


def ramsey_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x / p4)


def _fit(data: RamseyData) -> RamseyResults:
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x / p_4}.
    """
    qubits = data.df["qubit"].unique()

    t2s = {}
    corrected_qubit_frequencies = {}
    freqs_detuing = {}
    fitted_parameters = {}
    t2s_err = {}

    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]
        voltages = qubit_data_df["MSR"].pint.to("uV").pint.magnitude
        errors = qubit_data_df["errors"]
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
                0.5,
            ]
            popt, pcov = curve_fit(
                ramsey_fit,
                x,
                y,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [0, 0, 0, -np.pi, 0],
                    [1, 1, np.inf, np.pi, np.inf],
                ),
                sigma=np.array(errors, dtype=float) / (y_max - y_min),
                # absolute_sigma=True
            )
            print(chisquare(y, ramsey_fit(x, *popt)))
            err_popt = np.sqrt(np.diag(pcov))
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min / ((x_max - x_min) * popt[4])),
                popt[2] / (x_max - x_min),
                popt[3] - x_min * popt[2] / (x_max - x_min),
                popt[4] * (x_max - x_min),
            ]
            delta_fitting = popt[2] / (2 * np.pi)
            # FIXME: check this formula
            delta_phys = +int((delta_fitting - data.n_osc / data.t_max) * 1e9)
            err_delta_phys = err_popt[2] * 1e9 / (2 * np.pi)
            corrected_qubit_frequency = int(qubit_freq + delta_phys)
            t2 = popt[4]
            t2_error = err_popt[4] * (x_max - x_min)
        except Exception as e:
            log.warning(f"ramsey_fit: the fitting was not succesful. {e}")
            popt = [0] * 5
            t2 = 5.0
            corrected_qubit_frequency = int(qubit_freq)
            delta_phys = 0
            t2_error = 0

        fitted_parameters[qubit] = popt
        corrected_qubit_frequencies[qubit] = corrected_qubit_frequency / 1e9
        t2s[qubit] = t2
        t2s_err[qubit] = t2_error
        freqs_detuing[qubit] = delta_phys

    return RamseyResults(
        corrected_qubit_frequencies, t2s, t2s_err, freqs_detuing, fitted_parameters
    )


def _plot(data: RamseyData, fit: RamseyResults, qubit):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.df[data.df["qubit"] == qubit]
    errors = qubit_data["errors"].to_numpy()
    print(errors, len(qubit_data["wait"].pint.magnitude), len(errors))
    fig.add_trace(
        go.Scatter(
            x=qubit_data["wait"].pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            error_y=dict(
                # type='data', # value of error bar given in data coordinates
                array=errors,
                visible=True,
            ),
            marker_color=get_color(0),
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        )
    )

    # add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(qubit_data["wait"].pint.to("ns").pint.magnitude),
            max(qubit_data["wait"].pint.to("ns").pint.magnitude),
            2 * len(data),
        )

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=ramsey_fit(
                    waitrange,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                    float(fit.fitted_parameters[qubit][4]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(1),
            )
        )
        fitting_report = (
            fitting_report
            + (f"{qubit} | delta_frequency: {fit.delta_phys[qubit]:,.1f} Hz<br>")
            + (f"{qubit} | drive_frequency: {fit.frequency[qubit] * 1e9} Hz<br>")
            + (
                f"{qubit} | T2: {fit.t2[qubit]:,.0f} {chr(177)} {fit.t2_err[qubit]:,.0f} ns.<br><br>"
            )
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
"""Ramsey Routine object."""
