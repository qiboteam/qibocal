import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log

from ..utils import chi2_reduced


def exp_decay(x, *p):
    return p[0] - p[1] * np.exp(-1 * x / p[2])


def exponential_fit(data, zeno=None):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}

    for qubit in qubits:
        voltages = data[qubit].signal
        if zeno:
            times = np.arange(1, len(data[qubit].signal) + 1)
        else:
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
            popt = curve_fit(
                exp_decay,
                x,
                y,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [-2, -2, 0],
                    [2, 2, np.inf],
                ),
            )[0]
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[2] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
            ]
            t2 = popt[2]
            fitted_parameters[qubit] = popt
            decay[qubit] = t2

        except Exception as e:
            log.warning(f"Exponential decay fit failed for qubit {qubit} due to {e}")

    return decay, fitted_parameters


def exponential_fit_probability(data):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}
    chi2 = {}

    for qubit in qubits:
        times = data[qubit].wait
        x_max = np.max(times)
        x_min = np.min(times)
        x = (times - x_min) / (x_max - x_min)
        probability = data[qubit].prob
        p0 = [
            0.5,
            0.5,
            5,
        ]

        try:
            popt, perr = curve_fit(
                exp_decay,
                x,
                probability,
                p0=p0,
                maxfev=2000000,
                bounds=(
                    [-2, -2, 0],
                    [2, 2, np.inf],
                ),
                sigma=data[qubit].error,
            )
            popt = [
                popt[0],
                popt[1] * np.exp(x_min * popt[2] / (x_max - x_min)),
                popt[2] * (x_max - x_min),
            ]
            perr = np.sqrt(np.diag(perr))
            fitted_parameters[qubit] = popt
            dec = popt[2]
            decay[qubit] = (dec, perr[2])
            chi2[qubit] = (
                chi2_reduced(
                    data[qubit].prob,
                    exp_decay(data[qubit].wait, *fitted_parameters[qubit]),
                    data[qubit].error,
                ),
                np.sqrt(2 / len(data[qubit].prob)),
            )

        except Exception as e:
            log.warning(f"Exponential decay fit failed for qubit {qubit} due to {e}")

    return decay, fitted_parameters, chi2
