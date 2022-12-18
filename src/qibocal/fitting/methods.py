"""Routine-specific method for post-processing data acquired."""
import lmfit
import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log
from qibocal.data import Data
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, parse, rabi, ramsey


def lorentzian_fit(data, x, y, qubit, nqubits, labels, fit_file_name=None, qrm_lo=None):
    """Fitting routine for resonator spectroscopy"""
    if fit_file_name == None:
        data_fit = Data(
            name=f"fit_q{qubit}",
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                labels[0],
                labels[1],
                labels[2],
            ],
        )
    else:
        data_fit = Data(
            name=fit_file_name + f"_q{qubit}",
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                labels[0],
                labels[1],
                labels[2],
            ],
        )

    frequencies = data.get_values(*parse(x))
    voltages = data.get_values(*parse(y))

    # Create a lmfit model for fitting equation defined in resonator_peak
    model_Q = lmfit.Model(lorenzian)

    # Guess parameters for Lorentzian max or min
    if (nqubits == 1 and labels[0] == "resonator_freq") or (
        nqubits != 1 and labels[0] == "qubit_freq"
    ):
        guess_center = frequencies[
            np.argmax(voltages)
        ]  # Argmax = Returns the indices of the maximum values along an axis.
        guess_offset = np.mean(
            voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
        )
        guess_sigma = abs(frequencies[np.argmin(voltages)] - guess_center)
        guess_amp = (np.max(voltages) - guess_offset) * guess_sigma * np.pi

    else:
        guess_center = frequencies[
            np.argmin(voltages)
        ]  # Argmin = Returns the indices of the minimum values along an axis.
        guess_offset = np.mean(
            voltages[np.abs(voltages - np.mean(voltages) < np.std(voltages))]
        )
        guess_sigma = abs(frequencies[np.argmax(voltages)] - guess_center)
        guess_amp = (np.min(voltages) - guess_offset) * guess_sigma * np.pi

    # Add guessed parameters to the model
    model_Q.set_param_hint("center", value=guess_center, vary=True)
    model_Q.set_param_hint("sigma", value=guess_sigma, vary=True)
    model_Q.set_param_hint("amplitude", value=guess_amp, vary=True)
    model_Q.set_param_hint("offset", value=guess_offset, vary=True)
    guess_parameters = model_Q.make_params()

    # fit the model with the data and guessed parameters
    try:
        fit_res = model_Q.fit(
            data=voltages, frequency=frequencies, params=guess_parameters
        )
    except:
        log.warning("The fitting was not successful")
        return data_fit

    # get the values for postprocessing and for legend.
    f0 = fit_res.best_values["center"]
    BW = fit_res.best_values["sigma"] * 2
    Q = abs(f0 / BW)
    peak_voltage = (
        fit_res.best_values["amplitude"] / (fit_res.best_values["sigma"] * np.pi)
        + fit_res.best_values["offset"]
    )

    freq = f0 * 1e9

    MZ_freq = 0
    if qrm_lo != None:
        MZ_freq = freq - qrm_lo

    data_fit.add(
        {
            labels[0]: freq,
            labels[1]: peak_voltage,
            labels[2]: MZ_freq,
            "popt0": fit_res.best_values["amplitude"],
            "popt1": fit_res.best_values["center"],
            "popt2": fit_res.best_values["sigma"],
            "popt3": fit_res.best_values["offset"],
        }
    )
    return data_fit


def rabi_fit(data, x, y, qubit, nqubits, labels):
    data_fit = Data(
        name=f"fit_q{qubit}",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            "popt4",
            labels[0],
            labels[1],
        ],
    )

    time = data.get_values(*parse(x))
    voltages = data.get_values(*parse(y))

    if nqubits == 1:
        pguess = [
            np.mean(voltages.values),
            np.max(voltages.values) - np.min(voltages.values),
            0.5 / time.values[np.argmin(voltages.values)],
            np.pi / 2,
            0.1e-6,
        ]
    else:
        pguess = [
            np.mean(voltages.values),
            np.max(voltages.values) - np.min(voltages.values),
            0.5 / time.values[np.argmax(voltages.values)],
            np.pi / 2,
            0.1e-6,
        ]
    try:
        popt, pcov = curve_fit(
            rabi, time.values, voltages.values, p0=pguess, maxfev=10000
        )
        smooth_dataset = rabi(time.values, *popt)
        pi_pulse_duration = np.abs((1.0 / popt[2]) / 2)
        pi_pulse_max_voltage = smooth_dataset.max()
        t2 = 1.0 / popt[4]  # double check T1
    except:
        log.warning("The fitting was not succesful")
        return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            "popt3": popt[3],
            "popt4": popt[4],
            labels[0]: pi_pulse_duration,
            labels[1]: pi_pulse_max_voltage,
        }
    )
    return data_fit


def ramsey_fit(data, x, y, qubit, qubit_freq, sampling_rate, offset_freq, labels):

    data_fit = Data(
        name=f"fit_q{qubit}",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            "popt4",
            labels[0],
            labels[1],
            labels[2],
        ],
    )

    time = data.get_values(*parse(x))
    voltages = data.get_values(*parse(y))

    pguess = [
        np.mean(voltages.values),
        np.max(voltages.values) - np.min(voltages.values),
        0.5 / time.values[np.argmin(voltages.values)],
        np.pi / 2,
        500e-9,
    ]

    try:
        popt, pcov = curve_fit(
            ramsey, time.values, voltages.values, p0=pguess, maxfev=2000000
        )
        delta_fitting = popt[2]
        delta_phys = int((delta_fitting * sampling_rate) - offset_freq)
        corrected_qubit_frequency = int(qubit_freq + delta_phys)
        t2 = 1.0 / popt[4]
    except:
        log.warning("The fitting was not succesful")
        return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            "popt3": popt[3],
            "popt4": popt[4],
            labels[0]: delta_phys,
            labels[1]: corrected_qubit_frequency,
            labels[2]: t2,
        }
    )
    return data_fit


def t1_fit(data, x, y, qubit, nqubits, labels):

    data_fit = Data(
        name=f"fit_q{qubit}",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            labels[0],
        ],
    )

    time = data.get_values(*parse(x))
    voltages = data.get_values(*parse(y))

    if nqubits == 1:
        pguess = [
            max(voltages.values),
            (max(voltages.values) - min(voltages.values)),
            1 / 250,
        ]
    else:
        pguess = [
            min(voltages.values),
            (max(voltages.values) - min(voltages.values)),
            1 / 250,
        ]

    try:
        popt, pcov = curve_fit(
            exp, time.values, voltages.values, p0=pguess, maxfev=2000000
        )
        t1 = abs(1 / popt[2])

    except:
        log.warning("The fitting was not succesful")
        return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            labels[0]: t1,
        }
    )
    return data_fit


def flipping_fit(data, x, y, qubit, nqubits, niter, pi_pulse_amplitude, labels):

    data_fit = Data(
        name=f"fit_q{qubit}",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            labels[0],
            labels[1],
        ],
    )

    flips = data.get_values(*parse(x))  # Check X data stores. N flips or i?
    voltages = data.get_values(*parse(y))

    if nqubits == 1:
        pguess = [0.0003, np.mean(voltages), -18, 0]  # epsilon guess parameter
    else:
        pguess = [0.0003, np.mean(voltages), 18, 0]  # epsilon guess parameter

    try:
        popt, pcov = curve_fit(flipping, flips, voltages, p0=pguess, maxfev=2000000)
        epsilon = -np.pi / popt[2]
        amplitude_delta = np.pi / (np.pi + epsilon)
        corrected_amplitude = amplitude_delta * pi_pulse_amplitude
        # angle = (niter * 2 * np.pi / popt[2] + popt[3]) / (1 + 4 * niter)
        # amplitude_delta = angle * 2 / np.pi * pi_pulse_amplitude
    except:
        log.warning("The fitting was not succesful")
        return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            "popt3": popt[3],
            labels[0]: amplitude_delta,
            labels[1]: corrected_amplitude,
        }
    )
    return data_fit


def drag_tuning_fit(data, x, y, qubit, nqubits, labels):

    data_fit = Data(
        name=f"fit_q{qubit}",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            labels[0],
        ],
    )

    beta_params_keys = parse(x)
    voltages_keys = parse(y)
    beta_params = data[beta_params_keys[0]].pint.to(beta_params_keys[1]).pint.magnitude
    voltages = data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

    pguess = [
        0,  # Offset:    p[0]
        beta_params.values[np.argmax(voltages)]
        - beta_params.values[np.argmin(voltages)],  # Amplitude: p[1]
        4,  # Period:    p[2]
        0.3,  # Phase:     p[3]
    ]

    try:
        popt, pcov = curve_fit(cos, beta_params.values, voltages.values)
        smooth_dataset = cos(beta_params.values, popt[0], popt[1], popt[2], popt[3])
        beta_optimal = beta_params.values[np.argmin(smooth_dataset)]

    except:
        log.warning("The fitting was not succesful")
        return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            "popt3": popt[3],
            labels[0]: beta_optimal,
        }
    )
    return data_fit


def spin_echo_fit(data, x, y, qubit, nqubits, labels):

    data_fit = Data(
        name=f"fit_q{qubit}",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            labels[0],
        ],
    )

    time = data.get_values(*parse(x))
    voltages = data.get_values(*parse(y))

    if nqubits == 1:
        pguess = [
            max(voltages.values),
            (max(voltages.values) - min(voltages.values)),
            1 / 250,
        ]
    else:
        pguess = [
            min(voltages.values),
            (max(voltages.values) - min(voltages.values)),
            1 / 250,
        ]

    try:
        popt, pcov = curve_fit(
            exp, time.values, voltages.values, p0=pguess, maxfev=2000000
        )
        t2 = abs(1 / popt[2])

    except:
        log.warning("The fitting was not succesful")
        return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            labels[0]: t2,
        }
    )
    return data_fit


def calibrate_qubit_states_fit(data, x, y, nshots, qubits):

    parameters = Data(
        name=f"parameters",
        quantities={
            "rotation_angle",  # in degrees
            "threshold",
            "fidelity",
            "assignment_fidelity",
            "average_state0",
            "average_state1",
            "qubit",
        },
    )

    i_keys = parse(x)
    q_keys = parse(y)

    for qubit in qubits:
        qubit_data = data.df[data.df["qubit"] == int(qubit)].reset_index(drop=True)

        iq_state0 = (
            qubit_data[qubit_data["state"] == 0][i_keys[0]]
            .pint.to(i_keys[1])
            .pint.magnitude
            + 1.0j
            * qubit_data[qubit_data["state"] == 0][q_keys[0]]
            .pint.to(q_keys[1])
            .pint.magnitude
        )
        iq_state1 = (
            qubit_data[qubit_data["state"] == 1][i_keys[0]]
            .pint.to(i_keys[1])
            .pint.magnitude
            + 1.0j
            * qubit_data[qubit_data["state"] == 1][q_keys[0]]
            .pint.to(q_keys[1])
            .pint.magnitude
        )

        iq_state1 = np.array(iq_state1)
        iq_state0 = np.array(iq_state0)

        iq_mean_state1 = np.mean(iq_state1)
        iq_mean_state0 = np.mean(iq_state0)
        origin = iq_mean_state0

        iq_state0_translated = iq_state0 - origin
        iq_state1_translated = iq_state1 - origin
        rotation_angle = np.angle(np.mean(iq_state1_translated))

        iq_state1_rotated = iq_state1 * np.exp(-1j * rotation_angle)
        iq_state0_rotated = iq_state0 * np.exp(-1j * rotation_angle)

        real_values_state1 = iq_state1_rotated.real
        real_values_state0 = iq_state0_rotated.real

        real_values_combined = np.concatenate((real_values_state1, real_values_state0))
        real_values_combined.sort()

        cum_distribution_state1 = [
            sum(map(lambda x: x.real >= real_value, real_values_state1))
            for real_value in real_values_combined
        ]
        cum_distribution_state0 = [
            sum(map(lambda x: x.real >= real_value, real_values_state0))
            for real_value in real_values_combined
        ]

        cum_distribution_diff = np.abs(
            np.array(cum_distribution_state1) - np.array(cum_distribution_state0)
        )
        argmax = np.argmax(cum_distribution_diff)
        threshold = real_values_combined[argmax]
        errors_state1 = nshots - cum_distribution_state1[argmax]
        errors_state0 = cum_distribution_state0[argmax]
        fidelity = cum_distribution_diff[argmax] / nshots
        assignment_fidelity = 1 - (errors_state1 + errors_state0) / nshots / 2
        # assignment_fidelity = 1/2 + (cum_distribution_state1[argmax] - cum_distribution_state0[argmax])/nshots/2

        results = {
            "rotation_angle": (-rotation_angle * 360 / (2 * np.pi)) % 360,  # in degrees
            "threshold": threshold,
            "fidelity": fidelity,
            "assignment_fidelity": assignment_fidelity,
            "average_state0": iq_mean_state0,
            "average_state1": iq_mean_state1,
            "qubit": qubit,
        }
        parameters.add(results)
    return parameters
