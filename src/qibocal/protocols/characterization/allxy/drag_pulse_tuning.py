from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from scipy.optimize import curve_fit

from qibocal import update
from qibocal.auto.operation import Data, Qubits, Results, Routine
from qibocal.config import log

from ..utils import V_TO_UV, table_dict, table_html
from . import allxy_drag_pulse_tuning


@dataclass
class DragPulseTuningParameters(allxy_drag_pulse_tuning.AllXYDragParameters):
    """DragPulseTuning runcard inputs."""

    beta_start: float
    """DRAG pulse beta start sweep parameter."""
    beta_end: float
    """DRAG pulse beta end sweep parameter."""
    beta_step: float
    """DRAG pulse beta sweep step parameter."""


@dataclass
class DragPulseTuningResults(Results):
    """DragPulseTuning outputs."""

    betas: dict[QubitId, float]
    """Optimal beta paramter for each qubit."""
    fitted_parameters: dict[QubitId, dict[str, float]]
    """Raw fitting output."""


DragPulseTuningType = np.dtype([("msr", np.float64), ("beta", np.float64)])


@dataclass
class DragPulseTuningData(Data):
    """DragPulseTuning acquisition outputs."""

    data: dict[QubitId, npt.NDArray[DragPulseTuningType]] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: DragPulseTuningParameters,
    platform: Platform,
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
        result1 = platform.execute_pulse_sequence(
            seq1,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )
        result2 = platform.execute_pulse_sequence(
            seq2,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.INTEGRATION,
                averaging_mode=AveragingMode.CYCLIC,
            ),
        )

        # retrieve the results for every qubit
        for qubit in qubits:
            r1 = result1[ro_pulses[qubit].serial]
            r2 = result2[ro_pulses[qubit].serial]
            # store the results
            data.register_qubit(
                DragPulseTuningType,
                (qubit),
                dict(
                    msr=r1.magnitude - r2.magnitude,
                    beta=beta_param,
                ),
            )

    return data


def drag_fit(x, p0, p1, p2, p3):
    # Offset                  : p[0]
    # Amplitude               : p[1]
    # Period                  : p[2]
    # Phase                   : p[3]
    return p0 + p1 * np.cos(2 * np.pi * x / p2 + p3)


def _fit(data: DragPulseTuningData) -> DragPulseTuningResults:
    r"""
    Fitting routine for drag tunning. The used model is

        .. math::

            y = p_1 cos \Big(\frac{2 \pi x}{p_2} + p_3 \Big) + p_0.

    """
    qubits = data.qubits
    betas_optimal = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data = data[qubit]
        voltages = qubit_data.msr * V_TO_UV
        beta_params = qubit_data.beta

        try:
            popt, pcov = curve_fit(drag_fit, beta_params, voltages)
            smooth_dataset = drag_fit(beta_params, popt[0], popt[1], popt[2], popt[3])
            min_abs_index = np.abs(smooth_dataset).argmin()
            beta_optimal = beta_params[min_abs_index]
        except:
            log.warning("drag_tuning_fit: the fitting was not succesful")
            popt = np.array([0, 0, 1, 0])
            beta_optimal = 0

        fitted_parameters[qubit] = popt.tolist()
        betas_optimal[qubit] = beta_optimal
    return DragPulseTuningResults(betas_optimal, fitted_parameters)


def _plot(data: DragPulseTuningData, qubit, fit: DragPulseTuningResults):
    """Plotting function for DragPulseTuning."""

    figures = []
    fitting_report = ""

    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )
    qubit_data = data[qubit]
    fig.add_trace(
        go.Scatter(
            x=qubit_data.beta,
            y=qubit_data.msr * V_TO_UV,
            mode="markers",
            name="Probability",
            showlegend=True,
            legendgroup="group1",
        ),
    )

    # add fitting traces
    if fit is not None:
        beta_range = np.linspace(
            min(qubit_data.beta),
            max(qubit_data.beta),
            20,
        )

        fig.add_trace(
            go.Scatter(
                x=beta_range,
                y=drag_fit(
                    beta_range,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )
        fitting_report = table_html(
            table_dict(qubit, "Optimal Beta Param", np.round(fit.betas[qubit], 4))
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Beta parameter",
        yaxis_title="MSR[uV] [Rx(pi/2) - Ry(pi)] - [Ry(pi/2) - Rx(pi)]",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: DragPulseTuningResults, platform: Platform, qubit: QubitId):
    update.drag_pulse_beta(results.betas[qubit], platform, qubit)


drag_pulse_tuning = Routine(_acquisition, _fit, _plot, _update)
"""DragPulseTuning Routine object."""
