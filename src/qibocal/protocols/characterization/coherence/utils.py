import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log


def exp_decay(x, *p):
    return p[0] - p[1] * np.exp(-1 * x * p[2])


def exponential_fit(data):
    qubits = data.df["qubit"].unique()

    decay = {}
    fitted_parameters = {}

    for qubit in qubits:
        qubit_data_df = data.df[data.df["qubit"] == qubit]
        voltages = qubit_data_df["MSR"].pint.to("uV").pint.magnitude
        times = qubit_data_df["wait"].pint.to("ns").pint.magnitude

        try:
            y_max = np.max(voltages.values)
            y_min = np.min(voltages.values)
            y = (voltages.values - y_min) / (y_max - y_min)
            x_max = np.max(times.values)
            x_min = np.min(times.values)
            x = (times.values - x_min) / (x_max - x_min)

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
