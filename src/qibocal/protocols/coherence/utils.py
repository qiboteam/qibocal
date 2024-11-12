import numpy as np
from qibolab import Delay, Platform, PulseSequence
from scipy.optimize import curve_fit

from qibocal.auto.operation import QubitId
from qibocal.config import log

from ..utils import chi2_reduced

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


def spin_echo_sequence(platform: Platform, targets: list[QubitId], wait: int = 0):
    """Create pulse sequence for spin-echo routine.

    Spin Echo 3 Pulses: RX(pi/2) - wait t(rotates z) - RX(pi) - wait t(rotates z) - RX(pi/2) - readout
    """
    sequence = PulseSequence()
    all_delays = []
    for qubit in targets:
        natives = platform.natives.single_qubit[qubit]
        qd_channel, rx90_pulse = natives.R(theta=np.pi / 2)[0]
        _, rx_pulse = natives.RX()[0]
        ro_channel, ro_pulse = natives.MZ()[0]

        delays = [
            Delay(duration=wait),
            Delay(duration=wait),
            Delay(duration=wait),
            Delay(duration=wait),
        ]

        sequence.extend(
            [
                (qd_channel, rx90_pulse),
                (qd_channel, delays[0]),
                (qd_channel, rx_pulse),
                (qd_channel, delays[1]),
                (qd_channel, rx90_pulse),
                (
                    ro_channel,
                    Delay(duration=2 * rx90_pulse.duration + rx_pulse.duration),
                ),
                (ro_channel, delays[2]),
                (ro_channel, delays[3]),
                (ro_channel, ro_pulse),
            ]
        )
        all_delays.extend(delays)

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


def exponential_fit_probability(data, zeno=False):
    qubits = data.qubits

    decay = {}
    fitted_parameters = {}
    chi2 = {}
    pcovs = {}

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
            popt, pcov = curve_fit(
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
                popt[1] * np.exp(x_min / (x_max - x_min) / popt[2]),
                popt[2] * (x_max - x_min),
            ]

            pcovs[qubit] = pcov.tolist()
            fitted_parameters[qubit] = popt
            dec = popt[2]
            decay[qubit] = [dec, np.sqrt(pcov[2, 2]) * (x_max - x_min)]
            chi2[qubit] = [
                chi2_reduced(
                    data[qubit].prob,
                    exp_decay(data[qubit].wait, *fitted_parameters[qubit]),
                    data[qubit].error,
                ),
                np.sqrt(2 / len(data[qubit].prob)),
            ]

        except Exception as e:
            log.warning(f"Exponential decay fit failed for qubit {qubit} due to {e}")

    return decay, fitted_parameters, pcovs, chi2
