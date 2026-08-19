import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log

from ..utils import chi2_reduced


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
