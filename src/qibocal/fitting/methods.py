# -*- coding: utf-8 -*-
"""Routine-specific method for post-processing data acquired."""
import lmfit
import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log
from qibocal.data import Data
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, parse, rabi, ramsey, line


def lorentzian_fit(data, x, y, qubit, nqubits, labels, fit_file_name=None):
    """Fitting routine for resonator spectroscopy"""
    if fit_file_name == None:
        data_fit = Data(
            name=f"fit_q{qubit}",
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                labels[1],
                labels[0],
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
                labels[1],
                labels[0],
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

    data_fit.add(
        {
            labels[1]: peak_voltage,
            labels[0]: freq,
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


def res_spectrocopy_flux_fit(data, x, y, qubit, fluxline, labels):
    """ Fit frequency as a funcition of current for the flux resonator spectroscopy
        Args:
        data (DataUnits): Data file with information on the feature response at each current point.
        x (str): Column of the data file associated to x-axis.
        y (str): Column of the data file associated to y-axis.
        qubit (int): qubit coupled to the resonator that we are probing.
        fluxline (int): id of the current line used for the experiment.
        labels (list): Names of the data computed from the fit.

    Returns:
        data_fit (Data): Data file with labels and fit parameters.

    """

    curr=np.array(data.get_values(*parse(x)))
    freq=np.array(data.get_values(*parse(y)))/10**9
    freq_min1=np.min(freq)
    freq_max1=np.max(freq)
    freq_norm=(freq-freq_min1)/(freq_max1-freq_min1)

    I_array=np.linspace(curr[0],curr[-1],200)
    small_span=100000/10**9
    resolution=11
    freq_err=small_span/(resolution-1)
    freq_error_arr=np.zeros(len(freq))
    for j in range(len(freq_error_arr)):
        freq_error_arr[j]=freq_err

    if qubit==fluxline:
        data_fit = Data(
        name=f"fit_q{qubit}_f{fluxline}",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            labels[0],
            labels[1],
            labels[2],
            labels[3],
            labels[4],
            labels[5],
            labels[6],
            labels[7],
        ],
        )
        try:
            popt, pcov = curve_fit(cos, curr,freq_norm, sigma=(freq_error_arr-freq_min1)/(freq_max1-freq_min1))
            popt[1]=popt[1]*(freq_max1-freq_min1)
            popt[0]=popt[0]*(freq_max1-freq_min1)+freq_min1
            n=int(np.round(popt[3]/np.pi))
            curr_max=(n*np.pi-popt[3])*popt[2]/(2*np.pi)
            curr_max_err=np.abs(popt[2]/(2*np.pi))*np.sqrt(np.abs(pcov[3,3]))+np.abs((n*np.pi-popt[3])/(2*np.pi))*np.sqrt(np.abs(pcov[2,2]))
            freq_max=cos(curr_max, *popt)
            freq_max_err=(np.sqrt(np.abs(pcov[1,1]))+np.sqrt(np.abs(pcov[0,0])))*(freq_max1-freq_min1)
            freq_offset=cos(0, *popt)
            freq_offset_err=np.sqrt(np.abs(pcov[0,0]))*(freq_max1-freq_min1)+np.abs(np.cos(popt[3]))*np.sqrt(np.abs(pcov[1,1]))*(freq_max1-freq_min1)+np.abs(np.sin(popt[3])*popt[1])*np.sqrt(np.abs(pcov[3,3]))
            #freq_offset=freq_max-freq_zero
            #freq_offset_err=freq_max_err+freq_zero_err
            C_ii=(freq_max-freq_offset)/curr_max
            C_ii_err=freq_max_err/np.abs(curr_max)+freq_offset_err/np.abs(curr_max)+np.abs((freq_max-freq_offset)/curr_max**2)*curr_max_err
        except:
            log.warning("The fitting was not succesful")
            return data_fit

        data_fit.add(
        {
        "popt0": popt[0],
        "popt1": popt[1],
        "popt2": popt[2],
        "popt3": popt[3],
        labels[0]: curr_max,
        labels[1]: curr_max_err,
        labels[2]: freq_max,
        labels[3]: freq_max_err,
        labels[4]: C_ii,
        labels[5]: C_ii_err,
        labels[6]: freq_offset,
        labels[7]: freq_offset_err,
        }
        )   
    else:
        data_fit = Data(
        name=f"fit_q{qubit}_f{fluxline}",
        quantities=[
            "popt0",
            "popt1",
            labels[0],
            labels[1],
        ],
        )
        try:
            popt, pcov = curve_fit(line, curr,freq_norm, sigma=(freq_error_arr-freq_min1)/(freq_max1-freq_min1))
            popt[0]=popt[0]*(freq_max1-freq_min1)
            popt[1]=popt[1]*(freq_max1-freq_min1)+freq_min1
            C_ij=popt[0]
            C_ij_err=np.sqrt(np.abs(pcov[0,0]))*(freq_max1-freq_min1)
        except:
            log.warning("The fitting was not succesful")
            return data_fit

        data_fit.add(
        {
        "popt0": popt[0],
        "popt1": popt[1],
        labels[0]: C_ij,
        labels[1]: C_ij_err,
        }
        )  
    return data_fit
