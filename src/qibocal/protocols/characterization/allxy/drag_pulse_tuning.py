from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from ....auto.operation import Qubits, Results, Routine
from ....config import log
from ....data import DataUnits
from ....fitting.utils import cos
from ....plots.utils import get_color
from .allxy_drag_pulse_tuning import AllXYDragParameters


@dataclass
class DragPulseTuningParameters(AllXYDragParameters):
    ...


@dataclass
class DragPulseTuningResults(Results):
    betas: Dict[List[Tuple], str] = field(metadata=dict(update="beta"))
    fitted_parameters: Dict[List[Tuple], List]


class DragPulseTuningData(DataUnits):
    def __init__(self):
        super().__init__(
            name="data",
            quantities={"beta_param": "dimensionless"},
            options=["qubit", "iteration"],
        )


def _acquisition(
    platform: AbstractPlatform, qubits: Qubits, params: DragPulseTuningParameters
) -> DragPulseTuningData:
    r"""
    In this experiment, we apply two sequences in a given qubit: Rx(pi/2) - Ry(pi) and Ry(pi) - Rx(pi/2) for a range
    of different beta parameter values. After fitting, we obtain the best coefficient value for a pi pulse with drag shape.

    Args:
        platform (AbstractPlatform): Qibolab platform object
        qubits (dict): Dict of target Qubit objects to perform the action
        beta_start (float): Initial drag pulse beta parameter
        beta_end (float): Maximum drag pulse beta parameter
        beta_step (float): Scan range step for the drag pulse beta parameter
        software_averages (int): Number of executions of the routine for averaging results
        points (int): Save data results in a file every number of points

    Returns:
        - A DataUnits object with the raw data obtained for the fast and precision sweeps with the following keys

            - **MSR[V]**: Difference between resonator signal voltage mesurement in volts from sequence 1 and 2
            - **i[V]**: Difference between resonator signal voltage mesurement for the component I in volts from sequence 1 and 2
            - **q[V]**: Difference between resonator signal voltage mesurement for the component Q in volts from sequence 1 and 2
            - **phase[rad]**: Difference between resonator signal phase mesurement in radians from sequence 1 and 2
            - **beta_param[dimensionless]**: Optimal drag coefficient
            - **qubit**: The qubit being tested
            - **iteration**: The iteration number of the many determined by software_averages

        - A DataUnits object with the fitted data obtained with the following keys

            - **optimal_beta_param**: Best drag pulse coefficent
            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: period
            - **popt3**: phase
            - **qubit**: The qubit being tested
    """
    # define the parameter to sweep and its range:
    # qubit drive DRAG pulse beta parameter
    beta_param_range = np.arange(
        params.beta_start, params.beta_end, params.beta_step
    ).round(4)

    # create a DataUnits object to store the MSR, phase, i, q and the beta parameter
    data = DragPulseTuningData()

    count = 0
    # repeat the experiment as many times as defined by software_averages
    for iteration in range(params.software_averages):
        for beta_param in beta_param_range:
            # create two sequences of pulses
            # seq1: RX(pi/2) - RY(pi) - MZ
            # seq1: RY(pi/2) - RX(pi) - MZ

            ro_pulses = {}
            seq1 = PulseSequence()
            seq2 = PulseSequence()
            for qubit in qubits:
                # drag pulse RX(pi/2)
                RX90_drag_pulse = platform.create_RX90_drag_pulse(
                    qubit, start=0, beta=beta_param
                )
                # drag pulse RY(pi)
                RY_drag_pulse = platform.create_RX_drag_pulse(
                    qubit,
                    start=RX90_drag_pulse.finish,
                    relative_phase=+np.pi / 2,
                    beta=beta_param,
                )
                # drag pulse RY(pi/2)
                RY90_drag_pulse = platform.create_RX90_drag_pulse(
                    qubit, start=0, relative_phase=np.pi / 2, beta=beta_param
                )
                # drag pulse RX(pi)
                RX_drag_pulse = platform.create_RX_drag_pulse(
                    qubit, start=RY90_drag_pulse.finish, beta=beta_param
                )

                # RO pulse
                ro_pulses[qubit] = platform.create_qubit_readout_pulse(
                    qubit,
                    start=2
                    * RX90_drag_pulse.duration,  # assumes all single-qubit gates have same duration
                )
                # RX(pi/2) - RY(pi) - RO
                seq1.add(RX90_drag_pulse)
                seq1.add(RY_drag_pulse)
                seq1.add(ro_pulses[qubit])

                # RX(pi/2) - RY(pi) - RO
                seq2.add(RY90_drag_pulse)
                seq2.add(RX_drag_pulse)
                seq2.add(ro_pulses[qubit])

            # execute the pulse sequences
            result1 = platform.execute_pulse_sequence(seq1)
            result2 = platform.execute_pulse_sequence(seq2)

            # retrieve the results for every qubit
            for ro_pulse in ro_pulses.values():
                r1 = result1[ro_pulse.serial]
                r2 = result2[ro_pulse.serial]
                # store the results
                r = {
                    "MSR[V]": r1.measurement.mean() - r2.measurement.mean(),
                    "i[V]": r1.i.mean() - r2.i.mean(),
                    "q[V]": r1.q.mean() - r2.q.mean(),
                    "phase[rad]": r1.phase.mean() - r2.phase.mean(),
                    "beta_param[dimensionless]": beta_param,
                    "qubit": ro_pulse.qubit,
                    "iteration": iteration,
                }
                data.add(r)
            count += 1

    return data


def _fit(data: DragPulseTuningData) -> DragPulseTuningResults:
    r"""
    Fitting routine for drag tunning. The used model is

        .. math::

            y = p_1 cos \Big(\frac{2 \pi x}{p_2} + p_3 \Big) + p_0.
    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the model
        y (str): name of the output values for the model
        qubit (int): ID qubit number
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: period
            - **popt3**: phase
            - **labels[0]**: optimal beta.


    """
    qubits = data.df["qubit"].unique()
    betas_optimal = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]
        voltages = qubit_data_df["MSR"].pint.to("uV").pint.magnitude
        beta_params = qubit_data_df["beta_param"].pint.magnitude

        pguess = [
            0,  # Offset:    p[0]
            beta_params.values[np.argmax(voltages)]
            - beta_params.values[np.argmin(voltages)],  # Amplitude: p[1]
            4,  # Period:    p[2]
            0.3,  # Phase:     p[3]
        ]

        try:
            popt, pcov = curve_fit(cos, beta_params.values, voltages.values)
            smooth_dataset = cos(beta_params.values, popt[0], popt[1], popt[2], popt[3])
            min_abs_index = np.abs(smooth_dataset).argmin()
            beta_optimal = beta_params.values[min_abs_index]    
        except:
            log.warning("drag_tuning_fit: the fitting was not succesful")
            popt = [0, 0, 1, 0]
            beta_optimal = 0

        fitted_parameters[qubit] = popt
        betas_optimal[qubit] = beta_optimal
    return DragPulseTuningResults(betas_optimal, fitted_parameters)


def _plot(data: DragPulseTuningData, fit: DragPulseTuningResults, qubit):
    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    report_n = 0
    iterations = data.df["iteration"].unique()
    beta_params = [i.magnitude for i in data.df["beta_param"].unique()]
    for iteration in iterations:
        iteration_data = data.df[data.df["iteration"] == iteration]
        fig.add_trace(
            go.Scatter(
                x=iteration_data["beta_param"].pint.magnitude,
                y=iteration_data["MSR"].pint.to("uV").pint.magnitude,
                marker_color=get_color(report_n),
                mode="markers",
                opacity=0.3,
                name=f"q{qubit}/r{report_n}: Probability",
                showlegend=not bool(iteration),
                legendgroup="group1",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=beta_params,
            y=data.df.groupby("beta_param", as_index=False)
            .mean()["MSR"]
            .pint.to("uV")
            .pint.magnitude,  # pylint: disable=E1101
            name=f"q{qubit}/r{report_n}: Average MSR",
            marker_color=get_color(report_n),
            mode="markers",
        ),
        row=1,
        col=1,
    )

    # add fitting traces

    if len(data) > 0:
        beta_range = np.linspace(
            min(beta_params),
            max(beta_params),
            20,
        )

        fig.add_trace(
            go.Scatter(
                x=beta_range,
                y=cos(
                    beta_range,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                ),
                name=f"q{qubit}/r{report_n}: Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(4 * report_n + 2),
            ),
            row=1,
            col=1,
        )
        fitting_report = fitting_report + (
            f"q{qubit}/r{report_n} | optimal_beta_param: {fit.betas[qubit]:.4f}<br><br>"
        )
        report_n += 1

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Beta parameter",
        yaxis_title="MSR[uV] [Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
    )

    figures.append(fig)

    return figures, fitting_report


drag_pulse_tuning = Routine(_acquisition, _fit, _plot)
