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
class T1Parameters(Parameters):
    delay_before_readout_start: int
    delay_before_readout_end: list
    delay_before_readout_step: int
    software_averages: int = 1
    points: int = 10


@dataclass
class T1Results(Results):
    t1: Dict[List[Tuple], str] = field(metadata=dict(update="t1"))
    fitted_parameters: Dict[List[Tuple], List]


class T1Data(DataUnits):
    def __init__(self):
        super().__init__(
            name="data",
            quantities={"wait": "ns"},
            options=["qubit", "iteration", "resonator_type"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: T1Parameters
) -> T1Data:
    r"""
    In a T1 experiment, we measure an excited qubit after a delay. Due to decoherence processes
    (e.g. amplitude damping channel), it is possible that, at the time of measurement, after the delay,
    the qubit will not be excited anymore. The larger the delay time is, the more likely is the qubit to
    fall to the ground state. The goal of the experiment is to characterize the decay rate of the qubit
    towards the ground state.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (list): List of target qubits to perform the action
        delay_before_readout_start (int): Initial time delay before ReadOut
        delay_before_readout_end (list): Maximum time delay before ReadOut
        delay_before_readout_step (int): Scan range step for the delay before ReadOut
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Resonator signal voltage mesurement in volts
            - **i[V]**: Resonator signal voltage mesurement for the component I in volts
            - **q[V]**: Resonator signal voltage mesurement for the component Q in volts
            - **phase[rad]**: Resonator signal phase mesurement in radians
            - **wait[ns]**: Delay before ReadOut used in the current execution
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **labels[0]**: T1
            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **qubit**: The qubit being tested
    """

    # reload instrument settings from runcard
    platform.reload_settings()

    # create a sequence of pulses for the experiment
    # RX - wait t - MZ
    qd_pulses = {}
    ro_pulses = {}
    sequence = PulseSequence()
    for qubit in qubits:
        qd_pulses[qubit] = platform.create_RX_pulse(qubit, start=0)
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=qd_pulses[qubit].duration
        )
        sequence.add(qd_pulses[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    # wait time before readout
    ro_wait_range = np.arange(
        params.delay_before_readout_start,
        params.delay_before_readout_end,
        params.delay_before_readout_step,
    )

    # create a DataUnits object to store the MSR, phase, i, q and the delay time
    data = T1Data()

    # repeat the experiment as many times as defined by software_averages
    count = 0
    for iteration in range(params.software_averages):
        # sweep the parameter
        for wait in ro_wait_range:
            # # save data as often as defined by points
            # if count % params.points == 0 and count > 0:
            #     # save data
            #     yield data
            #     # calculate and save fit
            #     yield t1_fit(
            #         data,
            #         x="wait[ns]",
            #         y="MSR[uV]",
            #         qubits=qubits,
            #         resonator_type=platform.resonator_type,
            #         labels=["T1"],
            #     )

            for qubit in qubits:
                ro_pulses[qubit].start = qd_pulses[qubit].duration + wait

            # execute the pulse sequence
            results = platform.execute_pulse_sequence(sequence)

            for ro_pulse in ro_pulses.values():
                # average msr, phase, i and q over the number of shots defined in the runcard
                r = results[ro_pulse.serial].to_dict(average=True)
                r.update(
                    {
                        "wait[ns]": wait,
                        "qubit": ro_pulse.qubit,
                        "iteration": iteration,
                        "resonator_type": platform.resonator_type,
                    }
                )
                data.add(r)
            count += 1
    return data


def exp(x, *p):
    return p[0] - p[1] * np.exp(-1 * x * p[2])


def _fit(data: T1Data) -> T1Results:
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the T1 model
        y (str): name of the output values for the T1 model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **labels[0]**: T1.

    """
    qubits = data.df["qubit"].unique()
    resonator_type = data.df["resonator_type"].unique()
    t1s = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]
        voltages = qubit_data_df["MSR"].pint.to("uV").pint.magnitude
        times = qubit_data_df["wait"].pint.to("ns").pint.magnitude

        if resonator_type == "3D":
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
            t1 = abs(1 / popt[2])
        except:
            log.warning("t1_fit: the fitting was not succesful")
            t1 = 0
            popt = [0] * 3

        t1s[qubit] = t1
        fitted_parameters[qubit] = popt

    return T1Results(t1s, fitted_parameters)


def _plot(data: T1Data, fit: T1Results, qubit):
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

    data.df = data.df.drop(columns=["i", "q", "phase", "qubit"])
    iterations = data.df["iteration"].unique()
    waits = data.df["wait"].unique()

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
                x=waits,  # unique_waits,
                y=data.df.groupby("wait")["MSR"].mean() * 1e6,  # pylint: disable=E1101
                marker_color=get_color(report_n),
                name=f"q{qubit}/r{report_n}: Average",
                showlegend=True,
                legendgroup=f"q{qubit}/r{report_n}: Average",
            ),
            row=1,
            col=1,
        )

    # # add fitting trace
    if len(data) > 0:
        waitrange = np.linspace(
            min(data.df["wait"]),
            max(data.df["wait"]),
            2 * len(data),
        )
        # params = data_fit.df[data_fit.df["qubit"] == qubit].to_dict(
        #     orient="records"
        # )[0]

        fig.add_trace(
            go.Scatter(
                x=waitrange.magnitude,
                y=exp(
                    waitrange.magnitude,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                ),
                name=f"q{qubit}/r{report_n} Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(4 * report_n + 2),
            ),
            row=1,
            col=1,
        )
        fitting_report = fitting_report + (
            f"q{qubit}/r{report_n} | t1: {fit.t1[qubit]:,.0f} ns.<br><br>"
        )

    report_n += 1

    # last part
    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="MSR (uV)",
    )

    figures.append(fig)

    return figures, fitting_report


t1 = Routine(_acquisition, _fit, _plot)
