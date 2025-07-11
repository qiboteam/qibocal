import numpy as np
import plotly.graph_objects as go
from qibolab import Delay, Platform, PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId
from qibocal.config import log

from ..utils import COLORBAND, COLORBAND_LINE, chi2_reduced, table_dict, table_html

CoherenceType = np.dtype(
    [("wait", np.float64), ("signal", np.float64), ("phase", np.float64)]
)
"""Custom dtype for coherence routines."""


def average_single_shots(data_type, single_shots):
    """Convert single shot acquisition results of signal routines to averaged.

    Args:
        data_type: Type of produced data object (eg. ``T1SignalData``, ``T2SignalData`` etc.).
        single_shots (dict): Dictionary containing acquired single shot data.
    """
    data = data_type()
    for qubit, values in single_shots.items():
        data.register_qubit(
            CoherenceType,
            (qubit),
            {name: values[name].mean(axis=0) for name in values.dtype.names},
        )
    return data


def dynamical_decoupling_sequence(
    platform: Platform,
    targets: list[QubitId],
    wait: int = 0,
    n: int = 1,
    kind: str = "CPMG",
) -> tuple[PulseSequence, list[Delay]]:
    """Create dynamical decoupling sequence.

    Two sequences are available:
    - CP: RX90 (wait RX wait )^N RX90
    - CPMG: RX90 (wait RY wait )^N RX90
    """

    assert kind in ["CPMG", "CP"], f"Unknown sequence {kind}, please use CP or CPMG"
    sequence = PulseSequence()
    all_delays = []
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel = platform.qubits[qubit].drive
        rx90_sequence = natives.R(theta=np.pi / 2)
        decoupling_sequence = (
            natives.R(phi=np.pi / 2) if kind == "CPMG" else natives.RX()
        )
        ro_channel, ro_pulse = natives.MZ()[0]

        drive_delays = 2 * n * [Delay(duration=wait)]
        ro_delays = 2 * n * [Delay(duration=wait)]

        sequence += rx90_sequence

        for i in range(n):
            sequence.append((qd_channel, drive_delays[2 * i]))
            sequence.append((ro_channel, ro_delays[2 * i]))
            sequence += decoupling_sequence
            sequence.append((qd_channel, drive_delays[2 * i + 1]))
            sequence.append((ro_channel, ro_delays[2 * i + 1]))

        sequence += rx90_sequence

        sequence.append(
            (
                ro_channel,
                Delay(
                    duration=2 * rx90_sequence.duration
                    + n * decoupling_sequence.duration
                ),
            )
        )

        sequence.append((ro_channel, ro_pulse))
        all_delays.extend(drive_delays)
        all_delays.extend(ro_delays)
    print(sequence)
    return sequence, all_delays


def exp_decay(x, *p):
    return p[0] - p[1] * np.exp(-1 * x / p[2])


def exponential_fit(data, zeno=False):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}
    pcovs = {}

    for qubit in qubits:
        voltages = data[qubit].signal
        times = data[qubit].wait

        try:
            y_max = np.max(voltages)
            y_min = np.min(voltages)
            y = (voltages - y_min) / (y_max - y_min)
            x_max = np.max(times)
            x_min = np.min(times)
            x = (times - x_min) / (x_max - x_min)

            p0 = [
                0.5,
                0.5,
                5,
            ]
            popt, pcov = curve_fit(
                exp_decay,
                x,
                y,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [-2, -2, 0],
                    [2, 2, np.inf],
                ),
            )
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min / popt[2] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
            ]
            fitted_parameters[qubit] = popt
            pcovs[qubit] = pcov.tolist()
            decay[qubit] = [popt[2], np.sqrt(pcov[2, 2]) * (x_max - x_min)]

        except Exception as e:
            log.warning(f"Exponential decay fit failed for qubit {qubit} due to {e}")

    return decay, fitted_parameters, pcovs


def single_exponential_fit(x, y, error, zeno=False):
    """Fitting for single exponential decay."""
    x_max = np.max(x)
    x_min = np.min(x)
    x_norm = (x - x_min) / (x_max - x_min)
    p0 = [
        0.5,
        0.5,
        5,
    ]

    popt, pcov = curve_fit(
        exp_decay,
        x_norm,
        y,
        p0=p0,
        maxfev=2000000,
        bounds=(
            [-2, -2, 0],
            [2, 2, np.inf],
        ),
        sigma=error,
    )
    popt = [
        popt[0],
        popt[1] * np.exp(x_min / (x_max - x_min) / popt[2]),
        popt[2] * (x_max - x_min),
    ]
    decay = [popt[2], np.sqrt(pcov[2, 2]) * (x_max - x_min)]
    chi2 = [
        chi2_reduced(
            y,
            exp_decay(x, *popt),
            error,
        ),
        np.sqrt(2 / len(y)),
    ]
    return decay, popt, pcov.tolist(), chi2


def exponential_fit_probability(data, zeno=False):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}
    chi2 = {}
    pcovs = {}

    for qubit in qubits:
        try:
            decay[qubit], fitted_parameters[qubit], pcovs[qubit], chi2[qubit] = (
                single_exponential_fit(
                    data[qubit].wait,
                    data[qubit].prob,
                    data[qubit].error,
                    zeno=zeno,
                )
            )

        except Exception as e:
            log.warning(f"Exponential decay fit failed for qubit {qubit} due to {e}")

    return decay, fitted_parameters, pcovs, chi2


def plot(data, target: QubitId, fit=None) -> tuple[list[go.Figure], str]:
    """Plotting function for spin-echo or CPMG protocol."""

    figures = []
    fitting_report = ""
    qubit_data = data[target]
    waits = qubit_data.wait
    probs = qubit_data.prob
    error_bars = qubit_data.error

    fig = go.Figure(
        [
            go.Scatter(
                x=waits,
                y=probs,
                opacity=1,
                name="Probability of 1",
                showlegend=True,
                legendgroup="Probability of 1",
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
        waitrange = np.linspace(
            min(waits),
            max(waits),
            2 * len(qubit_data),
        )
        params = fit.fitted_parameters[target]

        fig.add_trace(
            go.Scatter(
                x=waitrange,
                y=exp_decay(waitrange, *params),
                name="Fit",
                line=go.scatter.Line(dash="dot"),
            ),
        )
        fitting_report = table_html(
            table_dict(
                target,
                ["T2", "chi2 reduced"],
                [fit.t2[target], fit.chi2[target]],
                display_error=True,
            )
        )

    fig.update_layout(
        showlegend=True,
        xaxis_title="Time [ns]",
        yaxis_title="Probability of State 1",
    )

    figures.append(fig)

    return figures, fitting_report
