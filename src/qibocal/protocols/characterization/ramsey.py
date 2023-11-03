from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.platform import Platform
from qibolab.pulses import PulseSequence
from qibolab.qubits import QubitId
from qibolab.sweeper import Parameter, Sweeper, SweeperType
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from qibocal import update
from qibocal.auto.operation import Data, Parameters, Qubits, Results, Routine

from .utils import GHZ_TO_HZ, HZ_TO_GHZ, chi2_reduced, table_dict, table_html

POPT_EXCEPTION = [0, 0, 0, 0, 0]
"""Fit parameters output to handle exceptions"""
PERR_EXCEPTION = [1] * 5
"""Fit errors to handle exceptions; their choice has no physical meaning
and is meant to avoid breaking the code."""
COLORBAND = "rgba(0,100,80,0.2)"
COLORBAND_LINE = "rgba(255,255,255,0)"


@dataclass
class RamseyParameters(Parameters):
    """Ramsey runcard inputs."""

    delay_between_pulses_start: int
    """Initial delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_end: int
    """Final delay between RX(pi/2) pulses in ns."""
    delay_between_pulses_step: int
    """Step delay between RX(pi/2) pulses in ns."""
    n_osc: Optional[int] = 0
    """Number of oscillations to induce detuning (optional).
        If 0 standard Ramsey experiment is performed."""


@dataclass
class RamseyResults(Results):
    """Ramsey outputs."""

    frequency: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [GHz] for each qubit."""
    t2: dict[QubitId, tuple[float, Optional[float]]]
    """T2 for each qubit [ns]."""
    delta_phys: dict[QubitId, tuple[float, Optional[float]]]
    """Drive frequency [Hz] correction for each qubit."""
    fitted_parameters: dict[QubitId, list[float]]
    """Raw fitting output."""
    chi2: dict[QubitId, tuple[float, Optional[float]]]


RamseyType = np.dtype(
    [("wait", np.float64), ("prob", np.float64), ("errors", np.float64)]
)
"""Custom dtype for coherence routines."""


@dataclass
class RamseyData(Data):
    """Ramsey acquisition outputs."""

    n_osc: int
    """Number of oscillations for detuning."""
    t_max: int
    """Final delay between RX(pi/2) pulses in ns."""
    detuning_sign: int
    """Sign for induced detuning."""
    qubit_freqs: dict[QubitId, float] = field(default_factory=dict)
    """Qubit freqs for each qubit."""
    data: dict[QubitId, npt.NDArray] = field(default_factory=dict)
    """Raw data acquired."""

    @property
    def waits(self):
        """
        Return a list with the waiting times without repetitions.
        """
        qubit = next(iter(self.data))
        return np.unique(self.data[qubit].wait)


def _acquisition(
    params: RamseyParameters,
    platform: Platform,
    qubits: Qubits,
) -> RamseyData:
    """Data acquisition for Ramsey Experiment (detuned)."""
    # create a sequence of pulses for the experiment
    # RX90 - t - RX90 - MZ
    ro_pulses = {}
    RX90_pulses1 = {}
    RX90_pulses2 = {}
    freqs = {}
    sequence = PulseSequence()
    for qubit in qubits:
        RX90_pulses1[qubit] = platform.create_RX90_pulse(qubit, start=0)
        RX90_pulses2[qubit] = platform.create_RX90_pulse(
            qubit,
            start=RX90_pulses1[qubit].finish,
        )
        ro_pulses[qubit] = platform.create_qubit_readout_pulse(
            qubit, start=RX90_pulses2[qubit].finish
        )
        freqs[qubit] = qubits[qubit].drive_frequency
        sequence.add(RX90_pulses1[qubit])
        sequence.add(RX90_pulses2[qubit])
        sequence.add(ro_pulses[qubit])

    # define the parameter to sweep and its range:
    waits = np.arange(
        # wait time between RX90 pulses
        params.delay_between_pulses_start,
        params.delay_between_pulses_end,
        params.delay_between_pulses_step,
    )

    data = RamseyData(
        n_osc=params.n_osc,
        t_max=params.delay_between_pulses_end,
        detuning_sign=+1,
        qubit_freqs=freqs,
    )

    if params.n_osc == 0:
        sweeper = Sweeper(
            Parameter.start,
            waits,
            [RX90_pulses2[qubit] for qubit in qubits],
            type=SweeperType.ABSOLUTE,
        )

        # execute the sweep
        results = platform.sweep(
            sequence,
            ExecutionParameters(
                nshots=params.nshots,
                relaxation_time=params.relaxation_time,
                acquisition_type=AcquisitionType.DISCRIMINATION,
                averaging_mode=AveragingMode.SINGLESHOT,
            ),
            sweeper,
        )
        for qubit in qubits:
            probs = results[qubit].probability()
            # The probability errors are the standard errors of the binomial distribution
            errors = [np.sqrt(prob * (1 - prob) / params.nshots) for prob in probs]
            data.register_qubit(
                RamseyType,
                (qubit),
                dict(
                    wait=waits,
                    prob=probs,
                    errors=errors,
                ),
            )
    else:
        for wait in waits:
            for qubit in qubits:
                RX90_pulses2[qubit].start = RX90_pulses1[qubit].finish + wait
                ro_pulses[qubit].start = RX90_pulses2[qubit].finish
                if params.n_osc != 0:
                    RX90_pulses2[qubit].relative_phase = (
                        RX90_pulses2[qubit].start
                        * (-2 * np.pi)
                        * (params.n_osc)
                        / params.delay_between_pulses_end
                    )

            results = platform.execute_pulse_sequence(
                sequence,
                ExecutionParameters(
                    nshots=params.nshots,
                    relaxation_time=params.relaxation_time,
                    acquisition_type=AcquisitionType.DISCRIMINATION,
                    averaging_mode=(AveragingMode.SINGLESHOT),
                ),
            )

            for qubit in qubits:
                prob = results[qubit].probability()
                error = np.sqrt(prob * (1 - prob) / params.nshots)
                data.register_qubit(
                    RamseyType,
                    (qubit),
                    dict(
                        wait=np.array([wait]),
                        prob=np.array([prob]),
                        errors=np.array([error]),
                    ),
                )
    return data


def ramsey_fit(x, p0, p1, p2, p3, p4):
    # A fit to Superconducting Qubit Rabi Oscillation
    #   Offset                       : p[0]
    #   Oscillation amplitude        : p[1]
    #   DeltaFreq                    : p[2]
    #   Phase                        : p[3]
    #   Arbitrary parameter T_2      : 1/p[4]
    return p0 + p1 * np.sin(x * p2 + p3) * np.exp(-x * p4)


def _fit(data: RamseyData) -> RamseyResults:
    r"""
    Fitting routine for Ramsey experiment. The used model is
    .. math::
        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.
    """
    qubits = data.qubits
    waits = data.waits
    popts = {}
    freq_measure = {}
    t2_measure = {}
    delta_phys_measure = {}
    chi2 = {}
    for qubit in qubits:
        qubit_data = data[qubit]
        qubit_freq = data.qubit_freqs[qubit]
        probs = qubit_data["prob"]
        try:
            popt, perr = fitting(waits, probs, qubit_data.errors)
        except:
            popt = POPT_EXCEPTION
            perr = PERR_EXCEPTION

        delta_fitting = popt[2] / (2 * np.pi)
        delta_phys = data.detuning_sign * int(
            (delta_fitting - data.n_osc / data.t_max) * GHZ_TO_HZ
        )
        corrected_qubit_frequency = int(qubit_freq - delta_phys)
        t2 = popt[4]
        freq_measure[qubit] = (
            corrected_qubit_frequency,
            perr[2] * GHZ_TO_HZ / (2 * np.pi * data.t_max),
        )
        t2_measure[qubit] = (t2, perr[4])
        popts[qubit] = popt
        delta_phys_measure[qubit] = (
            delta_phys,
            popt[2] * GHZ_TO_HZ / (2 * np.pi * data.t_max),
        )
        chi2[qubit] = (
            chi2_reduced(
                probs,
                ramsey_fit(waits, *popts[qubit]),
                qubit_data.errors,
            ),
            np.sqrt(2 / len(probs)),
        )
    return RamseyResults(freq_measure, t2_measure, delta_phys_measure, popts, chi2)


def _plot(data: RamseyData, qubit, fit: RamseyResults = None):
    """Plotting function for Ramsey Experiment."""

    figures = []
    fig = go.Figure()
    fitting_report = ""

    qubit_data = data.data[qubit]
    waits = data.waits
    probs = qubit_data["prob"]
    error_bars = qubit_data["errors"]
    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Voltage",
                showlegend=True,
                legendgroup="Voltage",
                mode="lines",
            ),
            go.Scatter(
                x=np.concatenate((waits, waits[::-1])),
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
                x=waits,
                y=ramsey_fit(
                    waits,
                    float(fit.fitted_parameters[qubit][0]),
                    float(fit.fitted_parameters[qubit][1]),
                    float(fit.fitted_parameters[qubit][2]),
                    float(fit.fitted_parameters[qubit][3]),
                    float(fit.fitted_parameters[qubit][4]),
                ),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            )
        )
        fitting_report = table_html(
            table_dict(
                qubit,
                [
                    "Delta Frequency [Hz]",
                    "Drive Frequency [Hz]",
                    "T2* [ns]",
                    "chi2 reduced",
                ],
                [
                    fit.delta_phys[qubit],
                    fit.frequency[qubit],
                    fit.t2[qubit],
                    fit.chi2[qubit],
                ],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        uirevision="0",  # ``uirevision`` allows zooming while live plotting
        xaxis_title="Time (ns)",
        yaxis_title="Ground state probability",
    )

    figures.append(fig)

    return figures, fitting_report


def _update(results: RamseyResults, platform: Platform, qubit: QubitId):
    update.drive_frequency(results.frequency[qubit][0] * HZ_TO_GHZ, platform, qubit)


ramsey = Routine(_acquisition, _fit, _plot, _update)
"""Ramsey Routine object."""


def fitting(x: list, y: list, errors: list = None) -> list:
    """
    Given the inputs list `x` and outputs one `y`, this function fits the
    `ramsey_fit` function and returns a list with the fit parameters.
    """
    y_max = np.max(y)
    y_min = np.min(y)
    x_max = np.max(x)
    x_min = np.min(x)
    delta_y = y_max - y_min
    delta_x = x_max - x_min
    y = (y - y_min) / delta_y
    x = (x - x_min) / delta_x
    err = errors / delta_y if errors is not None else None
    ft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), x[1] - x[0])
    mags = abs(ft)
    local_maxima = find_peaks(mags, threshold=10)[0]
    index = local_maxima[0] if len(local_maxima) > 0 else None
    # 0.5 hardcoded guess for less than one oscillation
    f = freqs[index] * 2 * np.pi if index is not None else 0.5
    p0 = [
        0.5,
        0.5,
        f,
        0,
        1,
    ]
    popt, perr = curve_fit(
        ramsey_fit,
        x,
        y,
        p0=p0,
        maxfev=5000,
        bounds=(
            [0, 0, 0, -np.pi, 0],
            [1, 1, np.inf, np.pi, np.inf],
        ),
        sigma=err,
    )
    popt = [
        delta_y * popt[0] + y_min,
        delta_y * popt[1] * np.exp(x_min * popt[4] / delta_x),
        popt[2] / delta_x,
        popt[3] - x_min * popt[2] / delta_x,
        popt[4] / delta_x,
    ]
    perr = np.sqrt(np.diag(perr))
    perr = [
        delta_y * perr[0],
        delta_y
        * np.exp(x_min * popt[4] / delta_x)
        * np.sqrt(perr[1] ** 2 + (popt[1] * x_min * perr[4] / delta_x) ** 2),
        perr[2] / delta_x,
        np.sqrt(perr[3] ** 2 + (perr[2] * x_min / delta_x) ** 2),
        perr[4] / delta_x,
    ]
    return popt, perr
