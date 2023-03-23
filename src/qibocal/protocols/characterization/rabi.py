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


@dataclass
class RabiAmplitudeParameters(Parameters):
    pulse_amplitude_start: float
    pulse_amplitude_end: float
    pulse_amplitude_step: float
    nshots: int
    relaxation_time: float
    software_averages: float


@dataclass
class RabiAmplitudeResults(Results):
    amplitude: Dict[List[Tuple], str] = field(metadata=dict(update="drive_amplitude"))
    fitted_parameters: Dict[List[Tuple], List]


class RabiAmplitudeData(DataUnits):
    def __init__(self):
        super().__init__(
            "data",
            {"amplitude": "dimensionless"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: RabiAmplitudeParameters
) -> RabiAmplitudeData:
    r"""
    In the Rabi experiment we apply a pulse at the frequency of the qubit and scan the drive pulse amplitude
    to find the drive pulse amplitude that creates a rotation of a desired angle.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        pulse_amplitude_start (int): Initial drive pulse amplitude for the Rabi experiment
        pulse_amplitude_end (int): Maximum drive pulse amplitude for the Rabi experiment
        pulse_amplitude_step (int): Scan range step for the drive pulse amplitude for the Rabi experiment
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **amplitude[dimensionless]**: Drive pulse amplitude
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **pi_pulse_amplitude**: pi pulse amplitude
            - **pi_pulse_peak_voltage**: pi pulse's maximum voltage
            - **popt0**: offset
            - **popt1**: oscillation amplitude
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
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].finish
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # qubit drive pulse amplitude
    qd_pulse_amplitude_range = np.arange(
        params.pulse_amplitude_start,
        params.pulse_amplitude_end,
        params.pulse_amplitude_step,
    )
    sweeper = Sweeper(
        Parameter.amplitude,
        qd_pulse_amplitude_range,
        [qd_pulses[qubit] for qubit in qubits],
    )

    # create a DataUnits object to store the results,
    # DataUnits stores by default MSR, phase, i, q
    # additionally include qubit drive pulse amplitude
    data = RabiAmplitudeData()

    for iteration in range(params.software_averages):
        # sweep the parameter
        results = platform.sweep(
            sequence,
            sweeper,
            nshots=params.nshots,
            relaxation_time=params.relaxation_time,
        )
        for qubit in qubits:
            # average msr, phase, i and q over the number of shots defined in the runcard
            result = results[ro_pulses[qubit].serial]
            r = result.to_dict()
            r.update(
                {
                    "amplitude[dimensionless]": qd_pulse_amplitude_range,
                    "qubit": len(qd_pulse_amplitude_range) * [qubit],
                    "iteration": len(qd_pulse_amplitude_range) * [iteration],
                    "resonator_type": len(qd_pulse_amplitude_range)
                    * [platform.resonator_type],
                }
            )
            data.add_data_from_dict(r)
    return data


def rabi(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   Period    T                  : 1/p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(2 * np.pi * x * p2 + p3) * np.exp(-x * p4)


def _fit(data: RabiAmplitudeData) -> RabiAmplitudeResults:
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()
    amplitudes = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby("amplitude", as_index=False)
            .mean()
        )

        amplitude = qubit_data["amplitude"].pint.to("dimensionless").pint.magnitude
        voltages = qubit_data["MSR"].pint.to("uV").pint.magnitude

        if resonator_type == "3D":
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / amplitude.values[np.argmin(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        else:
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / amplitude.values[np.argmax(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        try:
            popt, pcov = curve_fit(
                rabi, amplitude.values, voltages.values, p0=pguess, maxfev=10000
            )
            smooth_dataset = rabi(amplitude.values, *popt)
            pi_pulse_amplitude = np.abs((1.0 / popt[2]) / 2)
            # pi_pulse_peak_voltage = smooth_dataset.max()
            # t2 = 1.0 / popt[4]  # double check T1

        except:
            log.warning("rabi_fit: the fitting was not succesful")

        amplitudes[qubit] = pi_pulse_amplitude
        fitted_parameters[qubit] = popt

    return RabiAmplitudeResults(amplitudes, fitted_parameters)


def _plot(data: RabiAmplitudeData, fit: RabiAmplitudeResults, qubit):
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
    amplitudes = data.df["amplitude"].unique()
    data.df = data.df.drop(columns=["i", "q", "qubit"])

    if len(iterations) > 1:
        opacity = 0.3
    else:
        opacity = 1
    for iteration in iterations:
        amplitudes = (
            data.df["amplitude"].pint.to("dimensionless").pint.magnitude.unique()
        )
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data["amplitude"].pint.to("dimensionless").pint.magnitude,
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
                x=iteration_data["amplitude"].pint.to("dimensionless").pint.magnitude,
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
                x=amplitudes,
                y=data.df.groupby("amplitude")["MSR"]  # pylint: disable=E1101
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
                x=amplitudes,
                y=data.df.groupby("amplitude")["phase"]  # pylint: disable=E1101
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
        amplituderange = np.linspace(
            min(amplitudes),
            max(amplitudes),
            2 * len(data),
        )
        params = fit.fitted_parameters[qubit]

        fig.add_trace(
            go.Scatter(
                x=amplituderange,
                y=rabi(amplituderange, *params),
                name=f"q{qubit}/r{report_n} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color="rgb(255, 130, 67)",
            ),
            row=1,
            col=1,
        )

        fitting_report = fitting_report + (
            f"q{qubit}/r{report_n} | pi_pulse_amplitude: {fit.amplitude[qubit]:.3f}<br>"
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Amplitude (dimensionless)",
        yaxis_title="MSR (uV)",
        xaxis2_title="Amplitude (dimensionless)",
        yaxis2_title="Phase (rad)",
    )

    figures.append(fig)

    return figures, fitting_report


rabi_amplitude = Routine(_acquisition, _fit, _plot)
