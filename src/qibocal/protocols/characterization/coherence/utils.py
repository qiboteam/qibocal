import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log

from ..utils import V_TO_UV


def exp_decay(x, *p):
    return p[0] - p[1] * np.exp(-1 * x * p[2])


def exponential_fit(data, zeno=None):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}

    for qubit in qubits:
        voltages = data[qubit].msr * V_TO_UV
        if zeno:
            times = np.arange(1, len(data[qubit].msr) + 1)
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
                popt[2] / (x_max - x_min),
            ]
            t2 = 1.0 / popt[2]

        except Exception as e:
            log.warning(f"Exp decay fitting was not succesful. {e}")
            popt = [0] * 3
            t2 = 5.0

        fitted_parameters[qubit] = popt
        decay[qubit] = t2

    return decay, fitted_parameters
