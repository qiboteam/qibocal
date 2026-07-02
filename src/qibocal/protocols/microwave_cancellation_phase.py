"""Calibration for microwave crosstalk mitigation that sweeps cancelllation pulse phase."""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qibolab import (
    AcquisitionType,
    AveragingMode,
    ParallelSweepers,
    Parameter,
    PulseSequence,
    Sweeper,
)
from scipy.optimize import curve_fit

from qibocal.auto.operation import (
    Data,
    Parameters,
    Protocol,
    QubitId,
    QubitPairId,
    Results,
)
from qibocal.calibration import CalibrationPlatform
from qibocal.config import log
from qibocal.protocols.utils import table_dict, table_html
from qibocal.update import replace

__all__ = ["microwave_cross_cancellation"]


@dataclass
class MWCancellationPhaseParameters(Parameters):
    """RabiAmplitudeFreq parameters."""

    phase_range: tuple[float, float, float]
    """Phase range to sweep over."""


MWCancPhaseType = np.dtype(
    [
        ("phase", np.float64),
        ("prob", np.float64),
        ("error", np.float64),
    ]
)
"""Custom dtype for rabi amplitude."""


@dataclass
class MWCancellationPhaseData(Data):
    """MWCancellationPhase data acquisition."""

    mw_pulse: float
    cross_amplitude: dict[QubitPairId, float] = field(default_factory=dict)
    """Cross amplitude for each qubit pair."""
    data: dict[QubitPairId, npt.NDArray[MWCancPhaseType]] = field(default_factory=dict)
    """Raw data acquired."""

    def phases(self, qubit: QubitId) -> npt.NDArray:
        """Unique qubit amplitudes."""
        return np.unique(self[qubit].phase)


@dataclass
class MWCancellationPhaseResults(Results):
    """MWCancellationPhase results."""

    cancellation_pulses: dict[QubitId, list[float]] = field(default_factory=dict)
    """Cancellation phase and amplitude for each qubit."""
    fitted_parameters: dict[QubitId, list[float]] = field(default_factory=dict)
    """Fitted parameters for each qubit."""


def _acquisition(
    params: MWCancellationPhaseParameters,
    platform: CalibrationPlatform,
    targets: list[QubitPairId],
) -> MWCancellationPhaseData:
    """Data acquisition for Rabi experiment sweeping amplitude."""

    experiment_sequence = PulseSequence()
    updates = {}
    canc_pulses = []
    cross_amplitudes = {}
    for qubit, drive_line in targets:
        single_q_seq = PulseSequence()
        qubit_natives = platform.parameters.native_gates.single_qubit[qubit]
        qd_channel, qd_pulse = qubit_natives.RX()[0]
        qro_channel, qro_pulse = qubit_natives.MZ()[0]

        drive_channel, drive_pulse = platform.parameters.native_gates.single_qubit[
            drive_line
        ].RX()[0]
        cross_ampl = platform.calibration.microwave_crosstalk_matrix[qubit, drive_line]
        cross_pulse = replace(
            drive_pulse.new(),
            duration=qd_pulse.duration,
            amplitude=cross_ampl,
        )
        cancellation_pulse = cross_pulse.new()
        cross_amplitudes[(qubit, drive_line)] = cross_ampl

        single_q_seq |= PulseSequence(
            [
                (drive_channel, cross_pulse),
                (qd_channel, cancellation_pulse),
            ]
        )

        qubit_frequency = platform.parameters.configs[qd_channel].frequency
        updates |= {drive_channel: {"frequency": qubit_frequency}}
        canc_pulses.append(cancellation_pulse)

        single_q_seq |= PulseSequence([(qro_channel, qro_pulse)])
        experiment_sequence += single_q_seq

    phase_sweeper = Sweeper(
        parameter=Parameter.relative_phase,
        range=params.phase_range,
        pulses=canc_pulses,
    )

    data = MWCancellationPhaseData(
        mw_pulse=qd_pulse.duration,
        cross_amplitude=cross_amplitudes,
    )

    results = platform.execute(
        [experiment_sequence],
        [ParallelSweepers([phase_sweeper])],
        updates=[updates],
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.CYCLIC,
    )

    for pair in targets:
        qubit, _ = pair
        ro_pulse = list(
            experiment_sequence.channel(platform.qubits[qubit].acquisition)
        )[-1]
        prob = results[ro_pulse.id]
        data.register_qubit(
            MWCancPhaseType,
            pair,
            dict(
                phase=phase_sweeper.values,
                prob=prob.tolist(),
                error=np.sqrt(prob * (1 - prob) / params.nshots).tolist(),
            ),
        )
    return data


def fitting_function(x, a, b, c, d, tau):
    return a * (1 - np.cos(np.sqrt(b**2 + c**2 + 2 * b * c * np.cos(x - d)) * tau))


def _fit(data: MWCancellationPhaseData) -> MWCancellationPhaseResults:
    """Do not perform any fitting procedure."""

    cancellation_pulses = {}
    qubit_params = {}
    for qubit in data.data:
        canc_ampl = data.cross_amplitude[qubit]
        phi = data.phases(qubit)
        probabilities = data[qubit].prob
        errors = data[qubit].error

        pguess = [0.5, canc_ampl, canc_ampl, 0, data.mw_pulse]
        try:
            popt, perr = curve_fit(
                fitting_function,
                phi,
                probabilities,
                p0=pguess,
                maxfev=100000,
                bounds=(
                    [0, 0, 0, -np.inf, data.mw_pulse - 1e-9],
                    [1, np.inf, np.inf, np.inf, data.mw_pulse + 1e-9],
                ),
                sigma=errors,
            )
            cancellation_pulses[qubit] = [popt[3], data.cross_amplitude[qubit]]
            qubit_params[qubit] = popt.tolist()
        except Exception as e:
            log.warning(f"Rabi fit failed for qubit {qubit} due to {e}.")

    return MWCancellationPhaseResults(
        cancellation_pulses=cancellation_pulses,
        fitted_parameters=qubit_params,
    )


def _plot(
    data: MWCancellationPhaseData,
    target: QubitPairId,
    fit: MWCancellationPhaseResults = None,
):
    """Plotting function for RabiAmplitudeFrequency."""
    figures = []
    fitting_report = ""
    fig = make_subplots(
        rows=1,
        cols=1,
        horizontal_spacing=0.1,
        vertical_spacing=0.2,
        subplot_titles=("Probability",),
    )
    qubit_data = data[target]
    phases = data.phases(target)

    fig.update_xaxes(title_text="Phase [rad.]", row=1, col=1)
    fig.update_yaxes(title_text="Excited State Probability", row=1, col=1)

    figures.append(fig)

    fig.add_trace(
        go.Scatter(
            x=phases,
            y=qubit_data.prob,
            mode="markers",
            error_y=dict(
                type="data",
                array=qubit_data.error,
                visible=True,
            ),
        ),
        row=1,
        col=1,
    )

    if fit is not None and target in fit.fitted_parameters:
        fit_phase = np.linspace(min(phases), max(phases), 50 * len(phases))
        fig.add_trace(
            go.Scatter(
                x=fit_phase,
                y=fitting_function(fit_phase, *fit.fitted_parameters[target]),
                mode="lines",
            ),
            row=1,
            col=1,
        )

        fitting_report = table_html(
            table_dict(
                [target] * 2,
                ["Cancellation phase", "Cancellation amplitude"],
                fit.cancellation_pulses[target],
            )
        )

    fig.update_layout(
        showlegend=False,
        legend={"orientation": "h"},
    )
    return figures, fitting_report


def _update(
    results: MWCancellationPhaseResults,
    platform: CalibrationPlatform,
    target: QubitPairId,
):
    return


microwave_cross_cancellation = Protocol(_acquisition, _fit, _plot, _update)
"""Rabi amplitude with frequency tuning."""
