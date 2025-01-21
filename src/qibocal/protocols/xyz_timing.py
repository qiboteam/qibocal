from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibo.models.error_mitigation import curve_fit
from qibolab import AcquisitionType, AveragingMode, Custom, Delay, Pulse, PulseSequence
from scipy import special

from qibocal.auto.operation import Data, Parameters, Results, Routine
from qibocal.calibration.calibration import QubitId
from qibocal.calibration.platform import CalibrationPlatform
from qibocal.result import probability

from .utils import table_dict, table_html

COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


@dataclass
class XYZTimingResults(Results):
    """Outputs of the xyz-timing protocol."""

    fitted_parameters: dict[QubitId, list[float]]
    """Output parameters of the fitted function."""
    fitted_errors: dict[QubitId, list[float]]
    """Errors of the fit parameters."""
    delays: dict[QubitId, float]
    """Flux-drive delays."""


@dataclass
class XYZTimingParameters(Parameters):
    """XYZ-timing runcard inputs."""

    flux_amplitude: float
    """Amplitude of the flux pulse."""
    delay_step: int
    """Sweeper step value of the relative starts."""
    delay_stop: int
    """Sweeper stop value of the relative starts."""


XYZTimingType = np.dtype(
    [("delay", np.float64), ("prob", np.float64), ("errors", np.float64)]
)


@dataclass
class XYZTimingData(Data):

    pulse_duration: dict[QubitId, int]
    """Duration of the drive and flux pulse"""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""


def _acquisition(
    params: XYZTimingParameters,
    platform: CalibrationPlatform,
    targets: list[QubitId],
) -> XYZTimingData:
    """Data acquisition for drive-flux channels timing"""
    natives = platform.natives.single_qubit
    durations = np.arange(
        1,
        params.delay_stop,
        params.delay_step,
    )
    sequences = []
    ro_pulses = []
    drive_durations = {}
    for qubit in targets:
        drive_pulse = natives[qubit].RX()[0]
        drive_durations[qubit] = int(drive_pulse[1].duration)

    data = XYZTimingData(pulse_duration=drive_durations)
    for duration in durations:
        ro_pulses.append([])
        for qubit in targets:
            sequence = PulseSequence()
            drive_channel = platform.qubits[qubit].drive
            flux_channel = platform.qubits[qubit].flux
            ro_channel = platform.qubits[qubit].acquisition
            drive_pulse = natives[qubit].RX()[0]
            readout_pulse = natives[qubit].MZ()[0]
            drive_duration = int(drive_pulse[1].duration)
            total_flux_duration = duration + drive_duration
            flux_pulse = Pulse(
                duration=total_flux_duration,
                amplitude=params.flux_amplitude,
                relative_phase=0,
                envelope=Custom(
                    i_=np.concatenate(
                        [
                            np.zeros(duration),
                            np.ones(drive_duration),
                        ]
                    ),
                    q_=np.zeros(total_flux_duration),
                ),
            )
            qd_delay = Delay(duration=drive_duration)

            sequence.extend(
                [
                    (drive_channel, qd_delay),
                    drive_pulse,
                    (flux_channel, flux_pulse),
                ]
            )
            sequence.align([drive_channel, flux_channel, ro_channel])
            sequence.append(readout_pulse)
            sequences.append(sequence)
            ro_pulses[-1].append(readout_pulse)

    results = platform.execute(
        sequences,
        nshots=params.nshots,
        relaxation_time=params.relaxation_time,
        acquisition_type=AcquisitionType.DISCRIMINATION,
        averaging_mode=AveragingMode.SINGLESHOT,
    )
    for i, duration in enumerate(durations):
        for j, qubit in enumerate(targets):
            ro_pulse = ro_pulses[i][j][1]
            prob = probability(results[ro_pulse.id], state=1)
            error = np.sqrt(prob * (1 - prob) / params.nshots)
            data.register_qubit(
                XYZTimingType,
                (qubit),
                dict(
                    delay=np.array([duration]),
                    prob=np.array([prob]),
                    errors=np.array([error]),
                ),
            )
    return data


def fit_function(x, a, b, c, d, e):
    """Fit function of the xyz-timing protocol"""
    return a + b * (special.erf(e * x - c) - special.erf(e * x + d))


def _fit(data: XYZTimingData) -> XYZTimingResults:
    """Post-processing for drive-flux channels timing"""
    params = {}
    errors = {}
    delays = {}
    for qubit in data.qubits:
        data_qubit = data.data[qubit]
        delay = data_qubit.delay
        prob = data_qubit.prob
        err = data_qubit.errors
        initial_pars = [
            1,
            0.5,
            data.pulse_duration[qubit] / 2,
            data.pulse_duration[qubit] / 2,
            1,
        ]
        fit_parameters, perr = curve_fit(
            fit_function,
            delay - data.pulse_duration[qubit],
            prob,
            p0=initial_pars,
            sigma=err,
        )

        err = np.sqrt(np.diag(perr)).tolist()
        params[qubit] = fit_parameters.tolist()
        errors[qubit] = err
        delays[qubit] = [
            (fit_parameters[2] - fit_parameters[3]) / (2 * fit_parameters[4]),
            float(np.linalg.norm([err[2], err[3]]) / 2) / fit_parameters[4] ** 2,
        ]
    return XYZTimingResults(
        fitted_parameters=params,
        fitted_errors=errors,
        delays=delays,
    )


def _plot(data: XYZTimingData, target: QubitId, fit: XYZTimingResults = None):
    """Plotting function for drive-flux channels timing"""
    figures = []
    qubit_data = data.data[target]
    delays = qubit_data.delay
    probs = qubit_data.prob
    error_bars = qubit_data.errors
    x = delays - data.pulse_duration[target]
    fitting_report = ""
    fig = go.Figure(
        [
            go.Scatter(
                x=x,
                y=probs,
                opacity=1,
                name="Probability of State 1",
                showlegend=True,
                legendgroup="Probability of State 1",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((x, x[::-1])),
                y=np.concatenate((probs + error_bars, (probs - error_bars)[::-1])),
                fill="toself",
                fillcolor=COLORBAND,
                line=dict(color=COLORBAND_LINE),
                showlegend=True,
                name="Errors",
            ),
        ]
    )
    if fit is not None:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=fit_function(x, *fit.fitted_parameters[target]),
                opacity=1,
                name="Fit",
                showlegend=True,
                legendgroup="Probability of State 0",
                mode="lines",
            ),
        )
        fig.add_vline(
            x=fit.delays[target][0],
            line=dict(color="grey", width=3, dash="dash"),
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["Flux pulse delay [ns]"],
                [fit.delays[target]],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Excited state probability",
    )

    figures.append(fig)

    return figures, fitting_report


xyz_timing = Routine(_acquisition, _fit, _plot)
