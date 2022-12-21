"""Routine-specific method for post-processing data acquired."""
from functools import partial

import lmfit
import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log
from qibocal.data import Data
from qibocal.fitting.utils import (
    cos,
    exp,
    flipping,
    freq_q_mathieu,
    freq_r_mathieu,
    freq_r_transmon,
    line,
    lorenzian,
    parse,
    rabi,
    ramsey,
)


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


def drag_tunning_fit(data, x, y, qubit, nqubits, labels):

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

    beta_params = data.get_values(*parse(x))
    voltages = data.get_values(*parse(y))

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


def res_spectroscopy_flux_fit(data, x, y, qubit, fluxline, params_fit):
    """Fit frequency as a function of current for the flux resonator spectroscopy
        Args:
        data (DataUnits): Data file with information on the feature response at each current point.
        x (str): Column of the data file associated to x-axis.
        y (str): Column of the data file associated to y-axis.
        qubit (int): qubit coupled to the resonator that we are probing.
        fluxline (int): id of the current line used for the experiment.
        params_fit (list): List of parameters for the fit. [freq_rh, g, Ec, Ej].
                          freq_rh is the resonator frequency at high power and g in the readout coupling.
                          If Ec and Ej are missing, the fit is valid in the transmon limit and if they are indicated,
                          contains the next-order correction.

    Returns:
        data_fit (Data): Data file with labels and fit parameters.

    """

    curr = np.array(data.get_values(*parse(x)))
    freq = np.array(data.get_values(*parse(y)))
    if qubit == fluxline:
        if len(params_fit) == 2:
            quantities = [
                "curr_sp",
                "xi",
                "d",
                "f_q/f_rh",
                "g",
                "f_rh",
                "f_qs",
                "f_rs",
                "f_offset",
                "C_ii",
            ]
        else:
            quantities = [
                "curr_sp",
                "xi",
                "d",
                "g",
                "Ec",
                "Ej",
                "f_rh",
                "f_qs",
                "f_rs",
                "f_offset",
                "C_ii",
            ]

        data_fit = Data(
            name=f"fit1_q{qubit}_f{fluxline}",
            quantities=quantities,
        )
        try:
            f_rh = params_fit[0]
            g = params_fit[1]
            max_c = curr[np.argmax(freq)]
            min_c = curr[np.argmin(freq)]
            xi = 1 / (2 * abs(max_c - min_c))
            if len(params_fit) == 2:
                f_r = np.max(freq)
                f_q_0 = f_rh - g**2 / (f_r - f_rh)
                popt = curve_fit(
                    freq_r_transmon,
                    curr,
                    freq,
                    p0=[max_c, xi, 0, f_q_0 / f_rh, g, f_rh],
                )[0]
                f_qs = popt[3] * popt[5]
                f_rs = freq_r_transmon(popt[0], *popt)
                f_offset = freq_r_transmon(0, *popt)
                C_ii = (f_rs - f_offset) / popt[0]
                data_fit.add(
                    {
                        "curr_sp": popt[0],
                        "xi": popt[1],
                        "d": abs(popt[2]),
                        "f_q/f_rh": popt[3],
                        "g": popt[4],
                        "f_rh": popt[5],
                        "f_qs": f_qs,
                        "f_rs": f_rs,
                        "f_offset": f_offset,
                        "C_ii": C_ii,
                    }
                )
            else:
                Ec = params_fit[2]
                Ej = params_fit[3]
                freq_r_mathieu1 = partial(freq_r_mathieu, p7=0.4999)
                popt = curve_fit(
                    freq_r_mathieu1,
                    curr,
                    freq,
                    p0=[f_rh, g, max_c, xi, 0, Ec, Ej],
                    method="dogbox",
                )[0]
                f_qs = freq_q_mathieu(popt[2], *popt[2::])
                f_rs = freq_r_mathieu(popt[2], *popt)
                f_offset = freq_r_mathieu(0, *popt)
                C_ii = (f_rs - f_offset) / popt[2]
                data_fit.add(
                    {
                        "curr_sp": popt[2],
                        "xi": popt[3],
                        "d": abs(popt[4]),
                        "g": popt[1],
                        "Ec": popt[5],
                        "Ej": popt[6],
                        "f_rh": popt[0],
                        "f_qs": f_qs,
                        "f_rs": f_rs,
                        "f_offset": f_offset,
                        "C_ii": C_ii,
                    }
                )
        except:
            log.warning("The fitting was not successful")
            return data_fit
    else:
        data_fit = Data(
            name=f"fit1_q{qubit}_f{fluxline}",
            quantities=[
                "popt0",
                "popt1",
            ],
        )
        try:
            freq_min = np.min(freq)
            freq_max = np.max(freq)
            freq_norm = (freq - freq_min) / (freq_max - freq_min)
            popt = curve_fit(line, curr, freq_norm)[0]
            popt[0] = popt[0] * (freq_max - freq_min)
            popt[1] = popt[1] * (freq_max - freq_min) + freq_min
        except:
            log.warning("The fitting was not successful")
            return data_fit

        data_fit.add(
            {
                "popt0": popt[0],  # C_ij
                "popt1": popt[1],
            }
        )
    return data_fit


def res_spectroscopy_flux_matrix(folder, fluxlines):
    """Calculation of the resonator flux matrix, Mf.
       curr = Mf*freq + offset_c.
       Mf = Mc^-1, offset_c = -Mc^-1 * offset_f
       freq = Mc*curr + offset_f
        Args:
        folder (str): Folder where the data files with the experimental and fit data are.
        fluxlines (list): ids of the current line used for the experiment.

    Returns:
        data (Data): Data file with len(fluxlines)+1 columns that contains the flux matrix (Mf) and
                     offset (offset_c) in the last column.

    """
    import os

    from pandas import DataFrame

    fits = []
    for q in fluxlines:
        for f in fluxlines:
            file = f"{folder}/data/resonator_flux_sample/fit1_q{q}_f{f}.csv"
            if os.path.exists(file):
                fits += [f]
    if len(fits) == len(fluxlines) ** 2:
        mat = np.zeros((len(fluxlines), len(fluxlines)))
        offset = np.zeros(len(fluxlines))
        for i, q in enumerate(fluxlines):
            for j, f in enumerate(fluxlines):
                data_fit = Data.load_data(
                    folder, "data", "resonator_flux_sample", "csv", f"fit1_q{q}_f{f}"
                )
                if q == f:
                    element = "C_ii"
                    offset[i] = data_fit.get_values("f_offset")[0]
                else:
                    element = "popt0"
                mat[i, j] = data_fit.get_values(element)[0]
        m = np.linalg.inv(mat)
        offset_c = -m @ offset
        data = Data(name=f"flux_matrix")
        data.df = DataFrame(m)
        data.df.insert(len(fluxlines), "offset_c", offset_c, True)
        # [m, offset_c] freq = M*curr + offset --> curr = m*freq + offset_c  m = M^-1, offset_c = -M^-1 * offset
        data.to_csv(f"{folder}/data/resonator_flux_sample/")
    else:
        data = Data(name=f"flux_matrix")
    return data


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


def calibrate_qubit_states_fit(data_gnd, data_exc, x, y, nshots, qubit):

    parameters = Data(
        name=f"parameters_q{qubit}",
        quantities=[
            "rotation_angle",  # in degrees
            "threshold",
            "fidelity",
            "assignment_fidelity",
        ],
    )

    iq_exc = data_exc.get_values(*parse(x)) + 1.0j * data_exc.get_values(*parse(y))
    iq_gnd = data_gnd.get_values(*parse(x)) + 1.0j * data_gnd.get_values(*parse(y))

    iq_exc = np.array(iq_exc)
    iq_gnd = np.array(iq_gnd)

    iq_mean_exc = np.mean(iq_exc)
    iq_mean_gnd = np.mean(iq_gnd)
    origin = iq_mean_gnd

    iq_gnd_translated = iq_gnd - origin
    iq_exc_translated = iq_exc - origin
    rotation_angle = np.angle(np.mean(iq_exc_translated))

    iq_exc_rotated = iq_exc * np.exp(-1j * rotation_angle)
    iq_gnd_rotated = iq_gnd * np.exp(-1j * rotation_angle)

    real_values_exc = iq_exc_rotated.real
    real_values_gnd = iq_gnd_rotated.real

    real_values_combined = np.concatenate((real_values_exc, real_values_gnd))
    real_values_combined.sort()

    cum_distribution_exc = [
        sum(map(lambda x: x.real >= real_value, real_values_exc))
        for real_value in real_values_combined
    ]
    cum_distribution_gnd = [
        sum(map(lambda x: x.real >= real_value, real_values_gnd))
        for real_value in real_values_combined
    ]

    cum_distribution_diff = np.abs(
        np.array(cum_distribution_exc) - np.array(cum_distribution_gnd)
    )
    argmax = np.argmax(cum_distribution_diff)
    threshold = real_values_combined[argmax]
    errors_exc = nshots - cum_distribution_exc[argmax]
    errors_gnd = cum_distribution_gnd[argmax]
    fidelity = cum_distribution_diff[argmax] / nshots
    assignment_fidelity = 1 - (errors_exc + errors_gnd) / nshots / 2
    # assignment_fidelity = 1/2 + (cum_distribution_exc[argmax] - cum_distribution_gnd[argmax])/nshots/2

    results = {
        "rotation_angle": (-rotation_angle * 360 / (2 * np.pi)) % 360,  # in degrees
        "threshold": threshold,
        "fidelity": fidelity,
        "assignment_fidelity": assignment_fidelity,
    }
    parameters.add(results)
    return parameters
