from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color


@dataclass
class SpinEchoParameters(Parameters):
    delay_between_pulses_start: int
    delay_between_pulses_end: list
    delay_between_pulses_step: int


@dataclass
class SpinEchoResults(Results):
    t2_spin_echo: Dict[List[Tuple], str] = field(metadata=dict(update="t2_spin_echo"))
    fitted_paramters: Dict[List[Tuple], List]


class SpinEchoData(DataUnits):
    def __init__(self, resonator_type):
        super().__init__(
            "data",
            quantities={"wait": "ns"},
            options=["qubit"],
        )

        self._resonator_type = resonator_type

    @property
    def resonator_type(self):
        return self._resonator_type


def _acquisition(
    params: SpinEchoParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> SpinEchoData:
    # create a sequence of pulses for the experiment:
    # Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    ro_pulses = {}
    RX90_pulses1 = {}
    RX_pulses = {}
    RX90_pulses2 = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX_pulses[qubit] = platform.create_RX_pulse(
            qubit, start=RX90_pulses1[qubit].finish
        )
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit, start=RX_pulses[qubit].finish
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX_pulses[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # delay between pulses
    ro_wait_range = np.arange(
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = SpinEchoData(platform.resonator_type)

    # sweep the parameter
    for wait in ro_wait_range:
        # save data as often as defined by points

        for qubit in qubits:
            RX_pulses[qubit].start = RX90_pulses1[qubit].finish + wait
            RX90_pulses2[qubit].start = RX_pulses[qubit].finish + wait
            ro_pulses[qubit].start = RX90_pulses2[qubit].finish

        # execute the pulse sequence
        results = platform.execute_pulse_sequence(sequence)

        for ro_pulse in ro_pulses.values():
            # average msr, phase, i and q over the number of shots defined in the runcard
            r = results[ro_pulse.serial].average.raw
            r.update(
                {
                    "wait[ns]": wait,
                    "qubit": ro_pulse.qubit,
                }
            )
            data.add(r)
    return data


def exp(x, *p):
    return p[0] - p[1] * np.exp(-1 * x * p[2])


def _fit(data: SpinEchoData) -> SpinEchoResults:
    # TODO: improve this fitting
    qubits = data.df["qubit"].unique()
    fitted_parameters = {}
    t2s = {}

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == qubit]

        times = qubit_data["wait"].pint.to("ns").pint.magnitude
        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude

        if data.resonator_type == "3D":
            pguess = [
                max(voltages.values),
                (max(voltages.values) - min(voltages.values)),
                1 / 250,
            ]
        else:
            pguess = [
                min(voltages.values),
                (max(voltages.values) - min(voltages.values)),
                1 / 250,
            ]

        try:
            popt, pcov = curve_fit(
                exp, times.values, voltages.values, p0=pguess, maxfev=2000000
            )
            t2 = abs(1 / popt[2])
        except:
            log.warning("spin_echo_fit: the fitting was not successful")

        t2s[qubit] = t2
        fitted_parameters[qubit] = popt

    return SpinEchoResults(t2s, fitted_parameters)


def _plot(data: SpinEchoData, fit: SpinEchoResults, qubit: int):
    figures = []

    fig = go.Figure()

    # iterate over multiple data folders
    fitting_report = ""

    qubit_data = data.df[data.df["qubit"] == qubit]
    waits = data.df["wait"].pint.to("ns").pint.magnitude

    fig.add_trace(
        go.Scatter(
            x=qubit_data["wait"].pint.to("ns").pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            marker_color=get_color(0),
            opacity=1,
            name="Voltage",
            showlegend=True,
            legendgroup="Voltage",
        ),
    )

    # add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(waits),
            max(waits),
            2 * len(data),
        )
        params = fit.fitted_paramters[qubit]

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=exp(waitrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(1),
            ),
        )

        fitting_report = fitting_report + (
            f"{qubit} | T2 Spin Echo: {fit.t2_spin_echo[qubit]:,.0f} ns.<br><br>"
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


spin_echo = Routine(_acquisition, _fit, _plot)
