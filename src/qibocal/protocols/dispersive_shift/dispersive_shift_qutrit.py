from dataclasses import asdict, dataclass

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import AcquisitionType, AveragingMode, Parameter, PulseSequence, Sweeper

from qibocal.auto.operation import QubitId, Results, Routine
from qibocal.calibration import CalibrationPlatform
from qibocal.protocols.utils import (
    GHZ_TO_HZ,
    HZ_TO_GHZ,
    lorentzian,
    lorentzian_fit,
    readout_frequency,
    table_dict,
    table_html,
)

from ...result import magnitude, phase
from ..resonator_spectroscopies.resonator_spectroscopy import ResSpecType
from .dispersive_shift import DispersiveShiftData, DispersiveShiftParameters

__all__ = ["dispersive_shift_qutrit"]


@dataclass
class DispersiveShiftQutritParameters(DispersiveShiftParameters):
    """Dispersive shift inputs."""


@dataclass
class DispersiveShiftQutritResults(Results):
    """Dispersive shift outputs."""

    frequency_state_zero: dict[QubitId, float]
    """State zero frequency."""
    frequency_state_one: dict[QubitId, float]
    """State one frequency."""
    frequency_state_two: dict[QubitId, float]
    """State two frequency."""
    fitted_parameters_state_zero: dict[QubitId, list[float]]
    """Fitted parameters state zero."""
    fitted_parameters_state_one: dict[QubitId, list[float]]
    """Fitted parameters state one."""
    fitted_parameters_state_two: dict[QubitId, list[float]]
    """Fitted parameters state one."""

    @property
    def state_zero(self):
        return {key: value for key, value in asdict(self).items() if "zero" in key}

    @property
    def state_one(self):
        return {key: value for key, value in asdict(self).items() if "one" in key}

    @property
    def state_two(self):
        return {key: value for key, value in asdict(self).items() if "two" in key}


"""Custom dtype for rabi amplitude."""


@dataclass
class DispersiveShiftQutritData(DispersiveShiftData):
    """Dipsersive shift acquisition outputs."""


def _acquisition(
    params: DispersiveShiftParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> DispersiveShiftQutritData:
    r"""
    Data acquisition for dispersive shift qutrit experiment.
    Perform spectroscopy on the readout resonator, with the qubit in ground, excited state and
    second excited state showing the resonator shift produced by the coupling between the resonator and the qubit.

    Args:
        params (DispersiveShiftParameters): experiment's parameters
        platform (CalibrationPlatform): Qibolab platform object
        targets (list): list of target qubits to perform the action

    """

    sequence_0 = PulseSequence()
    sequence_1 = PulseSequence()
    sequence_2 = PulseSequence()

    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        # prepare 0 and measure
        sequence_0 += natives.MZ()

        # prepare 1 and measure
        sequence_1 += natives.RX() | natives.MZ()

        # prepare 2 and measure
        assert natives.RX12 is not None, f"Missing RX12 calibration for qubit {qubit}"
        sequence_2 += (natives.RX() + natives.RX12()) | natives.MZ()

    # define the parameter to sweep and its range:
    delta_frequency_range = np.arange(
        -params.freq_width / 2, params.freq_width / 2, params.freq_step
    )

    data = DispersiveShiftQutritData(resonator_type=platform.resonator_type)

    sweepers = [
        Sweeper(
            parameter=Parameter.frequency,
            values=readout_frequency(q, platform, state=1) + delta_frequency_range,
            channels=[platform.qubits[q].probe],
        )
        for q in targets
    ]

    results = platform.execute(
        [sequence_0, sequence_1, sequence_2],
        [sweepers],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.INTEGRATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for qubit in targets:
        for state, sequence in enumerate([sequence_0, sequence_1, sequence_2]):
            ro_pulse = list(sequence.channel(platform.qubits[qubit].acquisition))[-1]
            result = results[ro_pulse.id]
            data.register_qubit(
                ResSpecType,
                (qubit, state),
                dict(
                    freq=readout_frequency(qubit, platform, state=1)
                    + delta_frequency_range,
                    signal=magnitude(result),
                    phase=phase(result),
                ),
            )

    return data


def _fit(data: DispersiveShiftQutritData) -> DispersiveShiftQutritResults:
    """Post-Processing for dispersive shift"""
    qubits = data.qubits

    frequency_0 = {}
    frequency_1 = {}
    frequency_2 = {}
    fitted_parameters_0 = {}
    fitted_parameters_1 = {}
    fitted_parameters_2 = {}

    for i in range(3):
        for qubit in qubits:
            data_i = data[qubit, i]
            fit_result = lorentzian_fit(
                data_i, resonator_type=data.resonator_type, fit="resonator"
            )
            if fit_result is not None:
                if i == 0:
                    frequency_0[qubit], fitted_parameters_0[qubit], _ = fit_result
                elif i == 1:
                    frequency_1[qubit], fitted_parameters_1[qubit], _ = fit_result
                else:
                    frequency_2[qubit], fitted_parameters_2[qubit], _ = fit_result

    return DispersiveShiftQutritResults(
        frequency_state_zero=frequency_0,
        frequency_state_one=frequency_1,
        frequency_state_two=frequency_2,
        fitted_parameters_state_one=fitted_parameters_1,
        fitted_parameters_state_zero=fitted_parameters_0,
        fitted_parameters_state_two=fitted_parameters_2,
    )


def _plot(
    data: DispersiveShiftQutritData, target: QubitId, fit: DispersiveShiftQutritResults
):
    """Plotting function for dispersive shift."""
    figures = []
    fig = make_subplots(
        rows=1,
        cols=2,
        horizontal_spacing=0.1,
        vertical_spacing=0.1,
        subplot_titles=(
            "Signal [a.u.]",
            "phase [rad]",
        ),
    )
    # iterate over multiple data folders

    fitting_report = ""

    data_0 = data[target, 0]
    data_1 = data[target, 1]
    data_2 = data[target, 2]
    fit_data_0 = fit.state_zero if fit is not None else None
    fit_data_1 = fit.state_one if fit is not None else None
    fit_data_2 = fit.state_two if fit is not None else None
    for i, label, q_data, data_fit in list(
        zip(
            (0, 1, 2),
            ("State 0", "State 1", "State 2"),
            (data_0, data_1, data_2),
            (fit_data_0, fit_data_1, fit_data_2),
        )
    ):
        opacity = 1
        frequencies = q_data.freq * HZ_TO_GHZ
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=q_data.signal,
                opacity=opacity,
                name=f"{label}",
                showlegend=True,
                legendgroup=f"{label}",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frequencies,
                y=q_data.phase,
                opacity=opacity,
                showlegend=False,
                legendgroup=f"{label}",
            ),
            row=1,
            col=2,
        )

        if fit is not None:
            freqrange = np.linspace(
                min(frequencies),
                max(frequencies),
                2 * len(q_data),
            )
            params = data_fit[
                (
                    "fitted_parameters_state_zero"
                    if i == 0
                    else (
                        "fitted_parameters_state_one"
                        if i == 1
                        else "fitted_parameters_state_two"
                    )
                )
            ][target]
            fig.add_trace(
                go.Scatter(
                    x=freqrange,
                    y=lorentzian(freqrange, *params),
                    name=f"{label} Fit",
                    line=go.scatter.Line(dash="dot"),
                ),
                row=1,
                col=1,
            )

    if fit is not None:
        fitting_report = table_html(
            table_dict(
                target,
                [
                    "State Zero Frequency [Hz]",
                    "State One Frequency [Hz]",
                    "State Two Frequency [Hz]",
                ],
                np.round(
                    [
                        fit_data_0["frequency_state_zero"][target] * GHZ_TO_HZ,
                        fit_data_1["frequency_state_one"][target] * GHZ_TO_HZ,
                        fit_data_2["frequency_state_two"][target] * GHZ_TO_HZ,
                    ]
                ),
            )
        )
    fig.update_layout(
        showlegend=True,
        xaxis_title="Frequency [GHz]",
        yaxis_title="Signal [a.u.]",
        xaxis2_title="Frequency [GHz]",
        yaxis2_title="Phase [rad]",
    )

    figures.append(fig)

    return figures, fitting_report


dispersive_shift_qutrit = Routine(_acquisition, fit=_fit, report=_plot)
