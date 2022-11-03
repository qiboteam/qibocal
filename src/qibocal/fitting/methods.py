# -*- coding: utf-8 -*-
import lmfit
import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log
from qibocal.data import Data
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, parse, rabi, ramsey


def lorentzian_fit(data, x, y, qubit, nqubits, labels, fit_file_name=None):
    r"""
    Fitting routine for resonator/qubit spectroscopy.
    The used model is

    .. math::

        y = \frac{A}{\pi} \Big[ \frac{\sigma}{(f-f_0)^2 + \sigma^2} \Big] + y_0.

    Args:

    Args:
        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the Lorentzian model
        y (str): name of the output values for the Lorentzian model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

            -   When using ``resonator_spectroscopy`` the expected labels are [`resonator_freq`, `peak voltage`], where `resonator_freq` is the estimated frequency of the resonator, and peak voltage ...

            -   when using ``qubit_spectroscopy`` the expected labels are [`qubit_freq`, `peak voltage`], where `qubit_freq` is the estimated frequency of the qubit

        fit_file_name (str): file name, ``None`` is the default value.

    Returns:

        A ``Data`` object with the following keys

            - *labels[0]*: peak voltage
            - *labels[1]*: frequency
            - *popt0*: Lorentzian's amplitude
            - *popt1*: Lorentzian's center
            - *popt2*: Lorentzian's sigma
            - *popt3*: Lorentzian's offset
    """
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
    r"""
    Fitting routine for Rabi experiment. The used model is

    .. math::

        y = p_0 + p_1 sin(2 \pi p_2 x + p_3) e^{-x p_4}.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the Rabi model
        y (str): name of the output values for the Rabi model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - *popt0*: offset
            - *popt1*: oscillation amplitude
            - *popt2*: frequency
            - *popt3*: phase
            - *popt4*: T2
            - *labels[0]*: pulse duration
            - *labels[1]*: pulse maximum voltage
    """

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
    r"""
    Fitting routine for Ramsey experiment. The used model is

    .. math::

        y = p_0 + p_1 sin \Big(2 \pi p_2 x + p_3 \Big) e^{-x p_4}.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the Ramsey model
        y (str): name of the output values for the Ramsey model
        qubit (int): ID qubit number
        qubits_freq (float):
        sampling_rate (float):
        offset_freq (float):
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - *popt0*: offset
            - *popt1*: oscillation amplitude
            - *popt2*: frequency
            - *popt3*: phase
            - *popt4*: T2
            - *labels[0]*: physical delta
            - *labels[1]*: corrected qubit frequency
            - *labels[2]*: T2
    """
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

    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            p_0-p_1 e^{-x p_2}.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the T1 model
        y (str): name of the output values for the T1 model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - *popt0*: p0
            - *popt1*: p1
            - *popt2*: p2
            - *labels[0]*: T1.



    """

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
    r"""
    Fitting routine for T1 experiment. The used model is

    .. math::

        p_0 sin\Big(\frac{2 \pi x}{p_2} + p_3\Big).

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the flipping model
        y (str): name of the output values for the flipping model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        niter(int): ...
        pi_pulse_amplitude(float): ...
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - *popt0*: p0
            - *popt1*: p1
            - *popt2*: p2
            - *popt3*: p3
            - *labels[0]*: delta amplitude
            - *labels[1]*: corrected amplitude


    """

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
    r"""
    Fitting routine for drag tunning. The used model is

        .. math::

            p_1 cos \Big(\frac{2 \pi x}{p_2} + p_3 \Big) + p_0.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the model
        y (str): name of the output values for the model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - *popt0*: offset
            - *popt1*: oscillation amplitude
            - *popt2*: period
            - *popt3*: phase
            - *labels[0]*: optimal beta.


    """

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
