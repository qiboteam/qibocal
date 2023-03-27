from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from qibolab.sweeper import Parameter, Sweeper
from scipy.optimize import curve_fit

from qibocal.auto.operation import Parameters, Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.plots.utils import get_color

from .rabi import rabi


@dataclass
class RabiLengthParameters(Parameters):
    pulse_duration_start: float
    pulse_duration_end: float
    pulse_duration_step: float
    nshots: int
    relaxation_time: float
    software_averages: float


@dataclass
class RabiLengthResults(Results):
    length: Dict[List[Tuple], str] = field(metadata=dict(update="drive_length"))
    fitted_parameters: Dict[List[Tuple], List]


class RabiLengthData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"time": "ns"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: RabiLengthParameters
) -> RabiLengthData:
    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse length
    to find the drive pulse length that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_duration_start (int): Initial drive pulse length for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse length for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse length for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_duration_start (int): Initial drive pulse duration for the Rabi experiment
        pulse_duration_end (int): Maximum drive pulse duration for the Rabi experiment
        pulse_duration_step (int): Scan range step for the drive pulse duration for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **time[ns]**: Drive pulse duration in ns
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **pi_pulse_duration**: pi pulse duration
            - **pi_pulse_peak_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation length
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **qubit**: The qubit being tested
    """

    # create a sequence of pulses for the experiment
    sequence = PulseSequence()
    qd_pulses = {}
    ro_pulses = {}
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_qubit_drive_pulse(qubit, start=0, duration=4)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse duration time
    qd_pulse_duration_range = np.arange(
        params.pulse_duration_start,
        params.pulse_duration_end,
        params.pulse_duration_step,
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse length
    data = RabiLengthData()

    for iteration in range(params.software_averages):
        # sweep the parameter
        for duration in qd_pulse_duration_range:
            for qubit in qubits:
                qd_pulses[qubit].duration = duration
                ro_pulses[qubit].start = duration

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence, nshots=params.nshots)

            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].to_dict()
                r.update(
                    {
                        "time[ns]": duration,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                        "resonator_type": platform.resonator_type,
                    }
                )
                data.add(r)
    return data


def _fit(data: RabiLengthData) -> RabiLengthResults:
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()
    lengths = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration", "resonator_type"])
            .groupby("time", as_index=False)
            .mean()
        )

        length = qubit_data["time"].pint.to("ns").pint.magnitude
        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude

        if resonator_type == "3D":
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / length.values[np.argmin(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        else:
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / length.values[np.argmax(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        try:
            popt, pcov = curve_fit(
                rabi, length.values, voltages.values, p0=pguess, maxfev=10000
            )
            smooth_dataset = rabi(length.values, *popt)
            pi_pulse_length = np.abs((1.0 / popt[2]) / 2)
            # pi_pulse_peak_voltage = smooth_dataset.max()
            # t2 = 1.0 / popt[4]  # double check T1

        except:
            log.warning("rabi_fit: the fitting was not succesful")

        lengths[qubit] = pi_pulse_length
        fitted_parameters[qubit] = popt

    return RabiLengthResults(lengths, fitted_parameters)


def _plot(data: RabiLengthData, fit: RabiLengthResults, qubit):
    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "MSR (V)",
            "phase (rad)",
        ),
    )

    # iterate over multiple data folders
    report_n = 0

    data.df = data.df[data.df["qubit"] == qubit]
    iterations = data.df["iteration"].unique()
    lengths = data.df["time"].unique()
    data.df = data.df.drop(columns=["i", "q", "qubit"])

    if len(iterations) > 1:
        opacity = 0.3
    else:
        opacity = 1
    for iteration in iterations:
        lengths = data.df["time"].pint.to("ns").pint.magnitude.unique()
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data["time"].pint.to("ns").pint.magnitude,
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
        fig.add_trace(
            go.Scatter(
                x=iteration_data["time"].pint.to("ns").pint.magnitude,
                y=iteration_data["phase"].pint.to("rad").pint.magnitude,
                marker_color=get_color(report_n),
                opacity=opacity,
                name=f"q{qubit}/r{report_n}",
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}",
            ),
            row=1,
            col=2,
        )
    if len(iterations) > 1:
        data.df = data.df.drop(columns=["iteration"])  # pylint: disable=E1101
        fig.add_trace(
            go.Scatter(
                x=lengths,
                y=data.df.groupby("time")["MSR"]  # pylint: disable=E1101
                .mean()
                .pint.to("uV")
                .pint.magnitude,
                marker_color=get_color(report_n),
                name=f"q{qubit}/r{report_n}: Average",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=lengths,
                y=data.df.groupby("time")["phase"]  # pylint: disable=E1101
                .mean()
                .pint.to("rad")
                .pint.magnitude,
                marker_color=get_color(report_n),
                showlegend=False,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=2,
        )

    # add fitting trace
    if len(data) > 0:
        lengthrange = np.linspace(
            min(lengths),
            max(lengths),
            2 * len(data),
        )
        params = fit.fitted_parameters[qubit]

        fig.add_trace(
            go.Scatter(
                x=lengthrange,
                y=rabi(lengthrange, *params),
                name=f"q{qubit}/r{report_n} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fitting_report = fitting_report + (
            f"q{qubit}/r{report_n} | pi_pulse_length: {fit.length[qubit]:.3f}<br>"
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Time (ns)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


rabi_length = Routine(_acquisition, _fit, _plot)
