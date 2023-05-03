from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab.platforms.abstract import AbstractPlatform
from qibolab.pulses import PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import Qubits, Results, Routine
from qibocal.config import log
from qibocal.data import DataUnits
from qibocal.fitting.utils import cos
from qibocal.plots.utils import get_color

from . import allxy_drag_pulse_tuning


@dataclass
class DragPulseTuningParameters(allxy_drag_pulse_tuning.AllXYDragParameters):
    """DragPulseTuning runcard inputs."""


@dataclass
class DragPulseTuningResults(Results):
    """DragPulseTuning outputs."""

    betas: Dict[List[Tuple], str] = field(metadata=dict(update="beta"))
    """Optimal beta paramter for each qubit."""
    fitted_parameters: Dict[List[Tuple], List]
    """Raw fitting output."""


class DragPulseTuningData(DataUnits):
    """DragPulseTuning acquisition outputs."""

    def __init__(self):
        super().__init__(
            name="data",
            quantities={"beta_param": "dimensionless"},
            options=["qubit"],
        )


def _acquisition(
    params: DragPulseTuningParameters,
    platform: AbstractPlatform,
    qubits: Qubits,
) -> DragPulseTuningData:
    r"""
    Data acquisition for drag pulse tuning experiment.
    In this experiment, we apply two sequences in a given qubit: Rx(pi/2) - Ry(pi) and Ry(pi) - Rx(pi/2) for a range
    of different beta parameter values. After fitting, we obtain the best coefficient value for a pi pulse with drag shape.
    """
    # define the parameter to sweep and its range:
    # qubit drive DRAG pulse beta parameter
    beta_param_range = np.arange(
        params.beta_start, params.beta_end, params.beta_step
    ).round(4)

    # create a DataUnits object to store the MSR, phase, i, q and the beta parameter
    data = DragPulseTuningData()

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
            }
            data.add(r)

    return data


def _fit(data: DragPulseTuningData) -> DragPulseTuningResults:
    r"""
    Fitting routine for drag tunning. The used model is

        .. math::

            y = p_1 cos \Big(\frac{2 \pi x}{p_2} + p_3 \Big) + p_0.

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
    """Plotting function for DragPulseTuning."""

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    beta_params = [i.magnitude for i in data.df["beta_param"].unique()]
    qubit_data = data.df[data.df["qubit"] == qubit]
    fig.add_trace(
        go.Scatter(
            x=qubit_data["beta_param"].pint.magnitude,
            y=qubit_data["MSR"].pint.to("uV").pint.magnitude,
            marker_color=get_color(0),
            mode="markers",
            opacity=0.3,
            name="Probability",
            showlegend=True,
            legendgroup="group1",
        ),
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
                name="Fit",
                line=go.scatter.Line(dash="dot"),
                marker_color=get_color(1),
            ),
        )
        fitting_report = fitting_report + (
            f"{qubit} | Optimal Beta Param: {fit.betas[qubit]:.4f}<br><br>"
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Beta parameter",
        yaxis_title="MSR[uV] [Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
    )

    figures.append(fig)

    return figures, fitting_report


drag_pulse_tuning = Routine(_acquisition, _fit, _plot)
"""DragPulseTuning Routine object."""
