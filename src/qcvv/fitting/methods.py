# -*- coding: utf-8 -*-
"""Routine-specific method for post-processing data acquired."""

from curses import qiflush

import lmfit
import numpy as np
from scipy.optimize import curve_fit

from qcvv.config import log
from qcvv.data import Data
from qcvv.fitting.utils import (
    cos,
    exp,
    flipping,
    lorenzian,
    lorenzian_diff,
    parse,
    rabi,
    ramsey,
)


def lorentzian_fit(x, y, qubits, name="fit"):

    data_fit = Data(
        name=name,
        quantities=[
            "qubit",
            "peak_value",
            "fit_amplitude",
            "fit_center",
            "fit_sigma",
            "fit_offset",
        ],
    )
    # Create a lmfit model for fitting equation defined in resonator_peak
    model_Q = lmfit.Model(lorenzian)

    for qubit in np.unique(qubits).tolist():
        xq = x[qubit == qubits]

        yq = y[qubit == qubits]

        y_peak = yq[np.argmax(np.diff(yq))]
        guess_center = xq[yq == y_peak][0]
        xsigma = []
        for xi in xq[xq >= guess_center]:
            if yq[xi == xq] > np.std(yq):
                xsigma += [xi]
            else:
                break
        if len(xsigma) == 0:
            xsigma = x
        guess_sigma = max(xsigma) - min(xsigma)
        guess_offset = np.mean(yq[np.abs(yq - np.mean(yq) < np.std(yq))])
        guess_amp = (y_peak - guess_offset) * guess_sigma * np.pi

        # Add guessed parameters to the model
        model_Q.set_param_hint("center", value=guess_center, vary=True)
        model_Q.set_param_hint("sigma", value=guess_sigma, vary=True)
        model_Q.set_param_hint("amplitude", value=guess_amp, vary=True)
        model_Q.set_param_hint("offset", value=guess_offset, vary=True)
        guess_parameters = model_Q.make_params()

        # fit the model with the data and guessed parameters
        try:
            fit_res = model_Q.fit(data=yq, frequency=xq, params=guess_parameters)
        except:
            r = {
                "qubit": qubit,
                "peak_value": 0,
                "fit_amplitude": guess_amp,
                "fit_center": guess_center,
                "fit_sigma": guess_sigma,
                "fit_offset": guess_offset,
            }
            data_fit.add(r)
            return data_fit

        # get the values for postprocessing and for legend.
        f0 = fit_res.best_values["center"]
        BW = fit_res.best_values["sigma"] * 2
        Q = abs(f0 / BW)
        peak_voltage = (
            fit_res.best_values["amplitude"] / (fit_res.best_values["sigma"] * np.pi)
            + fit_res.best_values["offset"]
        )

        r = {
            "qubit": qubit,
            "peak_value": peak_voltage,
            "fit_amplitude": fit_res.best_values["amplitude"],
            "fit_center": fit_res.best_values["center"],
            "fit_sigma": fit_res.best_values["sigma"],
            "fit_offset": fit_res.best_values["offset"],
        }
        # print(r)
        data_fit.add(r)
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
            labels[2],
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
        rabi_oscillations_pi_pulse_max_voltage = smooth_dataset.max() * 1e6
        t1 = 1.0 / popt[4]  # double check T1
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
            labels[1]: rabi_oscillations_pi_pulse_max_voltage,
            labels[2]: t1,
        }
    )
    return data_fit


def ramsey_fit(
    data, xtag: str, ytags: list, qubit, qubit_freq, sampling_rate, offset_freq, labels
):

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

    if not isinstance(ytags, list):
        ytags = [ytags]

    for ytag in ytags:
        time = data.get_values(*parse(xtag))
        voltages = data.get_values(*parse(ytag))

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
            corrected_qubit_frequency = int(qubit_freq - delta_phys)
            t2 = 1.0 / popt[4]
        except:
            log.warning("The fitting was not succesful")
            popt = np.array([0, 0, 0, 0, 0])
            delta_phys = np.array(0)
            corrected_qubit_frequency = np.array(0)
            t2 = np.array(0)

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


def flipping_fit(
    data, xtag: str, ytags: list, qubit, nqubits, niter, pi_pulse_amplitude, labels
):

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
    if not isinstance(ytags, list):
        ytags = [ytags]

    for ytag in ytags:
        x = data.get_values(*parse(xtag))  # Check X data stores. N flips or i?
        y = data.get_values(*parse(ytag))

        if nqubits == 1:
            pguess = [
                max(y - np.mean(y)),
                np.mean(y),
                -18,
                0,
            ]  # epsilon guess parameter
        else:
            pguess = [max(y - np.mean(y)), np.mean(y), 18, 0]  # epsilon guess parameter

        try:
            popt, pcov = curve_fit(flipping, x, y, p0=pguess, maxfev=2000000)
            epsilon = -np.pi / popt[2]
            amplitude_delta = np.pi / (np.pi + epsilon)
            corrected_amplitude = amplitude_delta * pi_pulse_amplitude
            # angle = (niter * 2 * np.pi / popt[2] + popt[3]) / (1 + 4 * niter)
            # amplitude_delta = angle * 2 / np.pi * pi_pulse_amplitude
        except:
            log.warning("The fitting was not succesful")
            popt = np.array([0, 0, 0, 0])
            amplitude_delta = np.array(0)
            corrected_amplitude = np.array(0)

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
