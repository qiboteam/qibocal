"""Routine-specific method for post-processing data acquired."""
from functools import partial

import lmfit
import numpy as np
import pandas as pd
import pint
from scipy.optimize import curve_fit

from qibocal.config import log
from qibocal.data import Data
from qibocal.fitting.utils import (
    cos,
    cumulative,
    exp,
    flipping,
    freq_q_mathieu,
    freq_r_mathieu,
    freq_r_transmon,
    line,
    lorenzian,
    parse,
    pint_to_float,
    rabi,
    ramsey,
)


def lorentzian_fit(data, x, y, qubits, resonator_type, labels, fit_file_name=None):
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
        qubits (list): A list with the IDs of the qubits
        resonator_type (str): the type of readout resonator ['3D', '2D']
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

            -   When using ``resonator_spectroscopy`` the expected labels are [`readout_frequency`, `peak voltage`], where `readout_frequency` is the estimated frequency of the resonator, and `peak_voltage` the peak of the Lorentzian

            -   when using ``qubit_spectroscopy`` the expected labels are [`drive_frequency`, `peak voltage`], where `drive_frequency` is the estimated frequency of the qubit

        fit_file_name (str): file name, ``None`` is the default value.

    Returns:

        A ``Data`` object with the following keys

            - **labels[0]**: peak voltage
            - **labels[1]**: frequency
            - **popt0**: Lorentzian's amplitude
            - **popt1**: Lorentzian's center
            - **popt2**: Lorentzian's sigma
            - **popt3**: Lorentzian's offset
            - **qubit**: The qubit being tested

    Example:

        In the code below, a noisy Lorentzian dataset is implemented and then the ``lorentzian_fit`` method is applied.

            .. testcode::

                import numpy as np
                from qibocal.data import DataUnits
                from qibocal.fitting.methods import lorentzian_fit
                from qibocal.fitting.utils import lorenzian
                import matplotlib.pyplot as plt

                name = "test"
                nqubits = 1
                label = "drive_frequency"
                amplitude = -1
                center = 2
                sigma = 3
                offset = 4

                # generate noisy Lorentzian

                x = np.linspace(center - 10, center + 10, 100)
                noisy_lorentzian = (
                    lorenzian(x, amplitude, center, sigma, offset)
                    + amplitude * np.random.randn(100) * 0.5e-2
                )

                # Initialize data and evaluate the fit

                data = DataUnits(quantities={"frequency": "Hz"}, options=["qubit", "iteration"])

                mydict = {"frequency[Hz]": x, "MSR[V]": noisy_lorentzian, "qubit": 0, "iteration" : 0}

                data.load_data_from_dict(mydict)

                fit = lorentzian_fit(
                    data,
                    "frequency[Hz]",
                    "MSR[V]",
                    qubits = [0],
                    resonator_type='3D',
                    labels=[label, "peak_voltage", "intermediate_freq"],
                    fit_file_name=name,
                )

                fit_params = [fit.get_values(f"popt{i}") for i in range(4)]
                fit_data = lorenzian(x,*fit_params)

                # Plot

                #fig = plt.figure(figsize = (10,5))
                #plt.scatter(x,noisy_lorentzian,label="data",s=10,color = 'darkblue',alpha = 0.9)
                #plt.plot(x,fit_data, label = "fit", color = 'violet', linewidth = 3, alpha = 0.4)
                #plt.xlabel('frequency (Hz)')
                #plt.ylabel('MSR (Volt)')
                #plt.legend()
                #plt.title("Data fit")
                #plt.grid()
                #plt.show()

            The following plot shows the resulting output:

            .. image:: lorentzian_fit_result.png
                :align: center

    """
    if fit_file_name == None:
        data_fit = Data(
            name=f"fits",
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                labels[0],
                labels[1],
                "qubit",
            ],
        )
    else:
        data_fit = Data(
            name=fit_file_name,
            quantities=[
                "popt0",
                "popt1",
                "popt2",
                "popt3",
                labels[0],
                labels[1],
                "qubit",
            ],
        )
    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby("frequency", as_index=False)
            .mean()
        )
        frequencies_keys = parse(x)
        voltages_keys = parse(y)
        frequencies = (
            qubit_data[frequencies_keys[0]].pint.to(frequencies_keys[1]).pint.magnitude
        )  # convert frequencies to GHz for better fitting
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

        # Create a lmfit model for fitting equation defined in resonator_peak
        model_Q = lmfit.Model(lorenzian)

        # Guess parameters for Lorentzian max or min
        if (resonator_type == "3D" and "readout_frequency" in labels[0]) or (
            resonator_type == "2D" and "drive_frequency" in labels[0]
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
            # get the values for postprocessing and for legend.
            f0 = fit_res.best_values["center"]
            BW = fit_res.best_values["sigma"] * 2
            Q = abs(f0 / BW)
            peak_voltage = (
                fit_res.best_values["amplitude"]
                / (fit_res.best_values["sigma"] * np.pi)
                + fit_res.best_values["offset"]
            )

            freq = f0 * 1e9

            data_fit.add(
                {
                    labels[0]: freq,
                    labels[1]: peak_voltage,
                    "popt0": fit_res.best_values["amplitude"],
                    "popt1": fit_res.best_values["center"],
                    "popt2": fit_res.best_values["sigma"],
                    "popt3": fit_res.best_values["offset"],
                    "qubit": qubit,
                }
            )
        except:
            log.warning("lorentzian_fit: the fitting was not successful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
            )

    return data_fit


def rabi_fit(data, x, y, qubits, resonator_type, labels):
    r"""
    Fitting routine for Rabi experiment. The used model is

    .. math::

        y = p_0 + p_1 sin(2 \pi p_2 x + p_3) e^{-x p_4}.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the Rabi model
        y (str): name of the output values for the Rabi model
        qubits (list): A list with the IDs of the qubits
        resonator_type (str): the type of readout resonator ['3D', '2D']
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **labels[0]**: pulse parameter
            - **labels[1]**: pulse's maximum voltage
            - **qubit**: The qubit being tested
    """

    data_fit = Data(
        name="fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            "popt4",
            labels[0],
            labels[1],
            "qubit",
        ],
    )

    parameter_keys = parse(x)
    voltages_keys = parse(y)
    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby(parameter_keys[0], as_index=False)
            .mean()
        )
        parameter = (
            qubit_data[parameter_keys[0]].pint.to(parameter_keys[1]).pint.magnitude
        )
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

        if resonator_type == "3D":
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / parameter.values[np.argmin(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        else:
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / parameter.values[np.argmax(voltages.values)],
                np.pi / 2,
                0.1e-6,
            ]
        try:
            popt, pcov = curve_fit(
                rabi, parameter.values, voltages.values, p0=pguess, maxfev=10000
            )
            smooth_dataset = rabi(parameter.values, *popt)
            pi_pulse_parameter = np.abs((1.0 / popt[2]) / 2)
            pi_pulse_peak_voltage = smooth_dataset.max()
            t2 = 1.0 / popt[4]  # double check T1
            data_fit.add(
                {
                    "popt0": popt[0],
                    "popt1": popt[1],
                    "popt2": popt[2],
                    "popt3": popt[3],
                    "popt4": popt[4],
                    labels[0]: pi_pulse_parameter,
                    labels[1]: pi_pulse_peak_voltage,
                    "qubit": qubit,
                }
            )
        except:
            log.warning("rabi_fit: the fitting was not succesful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
            )

    return data_fit


def ramsey_fit(
    data, x, y, qubits, resonator_type, qubit_freqs, sampling_rate, offset_freq, labels
):
    r"""
    Fitting routine for Ramsey experiment. The used model is

    .. math::

        y = p_0 + p_1 sin \Big(p_2 x + p_3 \Big) e^{-x p_4}.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the Ramsey model
        y (str): name of the output values for the Ramsey model
        qubits (list): A list with the IDs of the qubits
        qubits_freq (float): frequency of the qubit
        sampling_rate (float): Platform sampling rate
        offset_freq (float): Total qubit frequency offset. It contains the artificial detunning applied
                             by the experimentalist + the inherent offset in the actual qubit frequency stored in the runcard.
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: frequency
            - **popt3**: phase
            - **popt4**: T2
            - **labels[0]**: Physical detunning of the actual qubit frequency
            - **labels[1]**: New qubit frequency after correcting the actual qubit frequency with the detunning calculated (labels[0])
            - **labels[2]**: T2
            - **qubit**: The qubit being tested
    """
    data_fit = Data(
        name="fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            "popt4",
            labels[0],
            labels[1],
            labels[2],
            "qubit",
        ],
    )

    parameter_keys = parse(x)
    voltages_keys = parse(y)
    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby(parameter_keys[0], as_index=False)
            .mean()
        )
        times = qubit_data[parameter_keys[0]].pint.to(parameter_keys[1]).pint.magnitude
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

        try:
            y_max = np.max(voltages.values)
            y_min = np.min(voltages.values)
            y = (voltages.values - y_min) / (y_max - y_min)
            x_max = np.max(times.values)
            x_min = np.min(times.values)
            x = (times.values - x_min) / (x_max - x_min)
            if resonator_type == "3D":
                index = np.argmin(y)
            else:
                index = np.argmax(y)

            p0 = [
                np.mean(y),
                y_max - y_min,
                0.5 / x[index],
                np.pi / 2,
                0,
            ]
            popt = curve_fit(ramsey, x, y, method="lm", p0=p0)[0]
            popt = [
                (y_max - y_min) * popt[0] + y_min,
                (y_max - y_min) * popt[1] * np.exp(x_min * popt[4] / (x_max - x_min)),
                popt[2] / (x_max - x_min),
                popt[3] - x_min * popt[2] / (x_max - x_min),
                popt[4] / (x_max - x_min),
            ]
            delta_fitting = popt[2] / 2 * np.pi
            delta_phys = int((delta_fitting * sampling_rate) - offset_freq)
            corrected_qubit_frequency = int(qubit_freqs[qubit] + delta_phys)
            t2 = 1.0 / popt[4]

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
                    "qubit": qubit,
                }
            )
        except:
            log.warning("ramsey_fit: the fitting was not succesful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
            )

    return data_fit


def t1_fit(data, x, y, qubits, resonator_type, labels):
    """
    Fitting routine for T1 experiment. The used model is

        .. math::

            y = p_0-p_1 e^{-x p_2}.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the T1 model
        y (str): name of the output values for the T1 model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **labels[0]**: T1.

    """

    data_fit = Data(
        name=f"fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            labels[0],
        ],
    )

    parameter_keys = parse(x)
    voltages_keys = parse(y)
    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby(parameter_keys[0], as_index=False)
            .mean()
        )
        times = qubit_data[parameter_keys[0]].pint.to(parameter_keys[1]).pint.magnitude
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

        if resonator_type == "3D":
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
                exp, times.values, voltages.values, p0=pguess, maxfev=2000000
            )
            t1 = abs(1 / popt[2])
            data_fit.add(
                {
                    "popt0": popt[0],
                    "popt1": popt[1],
                    "popt2": popt[2],
                    labels[0]: t1,
                    "qubit": qubit,
                }
            )

        except:
            log.warning("t1_fit: the fitting was not succesful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
            )

    return data_fit


def flipping_fit(data, x, y, qubits, resonator_type, pi_pulse_amplitudes, labels):
    r"""
    Fitting routine for T1 experiment. The used model is

    .. math::

        y = p_0 sin\Big(\frac{2 \pi x}{p_2} + p_3\Big).

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the flipping model
        y (str): name of the output values for the flipping model
        qubit (int): ID qubit number
        nqubits (int): total number of qubits
        niter(int): Number of times of the flipping sequence applied to the qubit
        pi_pulse_amplitudes(list): list of corrected pi pulse amplitude
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: p0
            - **popt1**: p1
            - **popt2**: p2
            - **popt3**: p3
            - **labels[0]**: delta amplitude
            - **labels[1]**: corrected amplitude


    """
    data_fit = Data(
        name="fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            labels[0],
            labels[1],
            "qubit",
        ],
    )

    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby("flips", as_index=False)
            .mean()
        )
        flips_keys = parse(x)
        voltages_keys = parse(y)
        flips = qubit_data[flips_keys[0]].pint.to(flips_keys[1]).pint.magnitude
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

        if resonator_type == "3D":
            pguess = [0.0003, np.mean(voltages), -18, 0]  # epsilon guess parameter
        else:
            pguess = [0.0003, np.mean(voltages), 18, 0]  # epsilon guess parameter

        try:
            popt, pcov = curve_fit(flipping, flips, voltages, p0=pguess, maxfev=2000000)
            epsilon = -np.pi / popt[2]
            amplitude_correction_factor = np.pi / (np.pi + epsilon)
            corrected_amplitude = (
                amplitude_correction_factor * pi_pulse_amplitudes[qubit]
            )
            data_fit.add(
                {
                    "popt0": popt[0],
                    "popt1": popt[1],
                    "popt2": popt[2],
                    "popt3": popt[3],
                    labels[0]: amplitude_correction_factor,
                    labels[1]: corrected_amplitude,
                    "qubit": qubit,
                }
            )
        except:
            log.warning("flipping_fit: the fitting was not succesful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
            )

    return data_fit


def drag_tuning_fit(data: Data, x, y, qubits, labels):
    r"""
    Fitting routine for drag tunning. The used model is

        .. math::

            y = p_1 cos \Big(\frac{2 \pi x}{p_2} + p_3 \Big) + p_0.

    Args:

        data (`DataUnits`): dataset for the fit
        x (str): name of the input values for the model
        y (str): name of the output values for the model
        qubit (int): ID qubit number
        labels (list of str): list containing the lables of the quantities computed by this fitting method.

    Returns:

        A ``Data`` object with the following keys

            - **popt0**: offset
            - **popt1**: oscillation amplitude
            - **popt2**: period
            - **popt3**: phase
            - **labels[0]**: optimal beta.


    """

    data_fit = Data(
        name=f"fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            "popt3",
            labels[0],
            "qubit",
        ],
    )

    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby("beta_param", as_index=False)
            .mean()
        )
        beta_params_keys = parse(x)
        voltages_keys = parse(y)
        beta_params = (
            qubit_data[beta_params_keys[0]].pint.to(beta_params_keys[1]).pint.magnitude
        )
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

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
            data_fit.add(
                {
                    "popt0": popt[0],
                    "popt1": popt[1],
                    "popt2": popt[2],
                    "popt3": popt[3],
                    labels[0]: beta_optimal,
                    "qubit": qubit,
                }
            )
        except:
            log.warning("drag_tuning_fit: the fitting was not succesful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
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
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
            )

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
            data_fit.add(
                {
                    "popt0": popt[0],  # C_ij
                    "popt1": popt[1],
                }
            )
        except:
            log.warning("The fitting was not successful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
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


def spin_echo_fit(data, x, y, qubits, resonator_type, labels):
    data_fit = Data(
        name=f"fits",
        quantities=[
            "popt0",
            "popt1",
            "popt2",
            labels[0],
        ],
    )

    parameter_keys = parse(x)
    voltages_keys = parse(y)
    for qubit in qubits:
        qubit_data = (
            data.df[data.df["qubit"] == qubit]
            .drop(columns=["qubit", "iteration"])
            .groupby(parameter_keys[0], as_index=False)
            .mean()
        )
        times = qubit_data[parameter_keys[0]].pint.to(parameter_keys[1]).pint.magnitude
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

        if resonator_type == "3D":
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
                exp, times.values, voltages.values, p0=pguess, maxfev=2000000
            )
            t2 = abs(1 / popt[2])

            data_fit.add(
                {
                    "popt0": popt[0],
                    "popt1": popt[1],
                    "popt2": popt[2],
                    labels[0]: t2,
                    "qubit": qubit,
                }
            )
        except:
            log.warning("spin_echo_fit: the fitting was not succesful")
            data_fit.add(
                {key: 0 if key != "qubit" else qubit for key in data_fit.df.columns}
            )

    return data_fit


def calibrate_qubit_states_fit(data, x, y, nshots, qubits, degree=True):
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
        qubit_data = data.df[data.df["qubit"] == qubit]

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

        rotation_angle = np.angle(iq_mean_state1 - iq_mean_state0)

        iq_state1_rotated = iq_state1 * np.exp(-1j * rotation_angle)
        iq_state0_rotated = iq_state0 * np.exp(-1j * rotation_angle)

        real_values_state1 = iq_state1_rotated.real
        real_values_state0 = iq_state0_rotated.real

        real_values_combined = np.concatenate((real_values_state1, real_values_state0))
        real_values_combined.sort()

        cum_distribution_state1 = cumulative(real_values_combined, real_values_state1)
        cum_distribution_state0 = cumulative(real_values_combined, real_values_state0)

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
        if degree:
            rotation_angle = (-rotation_angle * 360 / (2 * np.pi)) % 360

        results = {
            "rotation_angle": rotation_angle,
            "threshold": threshold,
            "fidelity": fidelity,
            "assignment_fidelity": assignment_fidelity,
            "average_state0": iq_mean_state0,
            "average_state1": iq_mean_state1,
            "qubit": qubit,
        }
        parameters.add(results)
    return parameters


def ro_optimization_fit(data, *labels, debug=False):
    """
    Fit the fidelities from parameters swept as labels, and extract rotation angle and threshold

    Args:
        data (Data): data to fit
        labels (str): variable used in the routine with format "variable_name"

    Returns:
        Data: data with the fit results
    """
    quantities = [
        *labels,
        "rotation_angle",
        "threshold",
        "fidelity",
        "assignment_fidelity",
        "average_state0",
        "average_state1",
    ]
    data_fit = Data(
        name="fit",
        quantities=quantities,
    )

    # Create a ndarray for i and q shots for all labels
    # shape=(i + j*q, qubit, state, label1, label2, ...)

    shape = (*[len(data.df[label].unique()) for label in labels],)
    nb_shots = len(data.df["iteration"].unique())

    iq_complex = data.df["i"].pint.magnitude.to_numpy().reshape(shape) + 1j * data.df[
        "q"
    ].pint.magnitude.to_numpy().reshape(shape)

    # Move state to 0, and iteration to -1
    labels = list(labels)
    iq_complex = np.moveaxis(
        iq_complex, [labels.index("state"), labels.index("iteration")], [0, -1]
    )
    labels.remove("state")
    labels.remove("iteration")
    labels = ["state"] + labels + ["iteration"]

    # Take the mean ground state
    mean_gnd_state = np.mean(iq_complex[0, ...], axis=-1, keepdims=True)
    mean_exc_state = np.mean(iq_complex[1, ...], axis=-1, keepdims=True)
    angle = np.angle(mean_exc_state - mean_gnd_state)

    # Rotate the data
    iq_complex = iq_complex * np.exp(-1j * angle)

    # Take the cumulative distribution of the real part of the data
    iq_complex_sorted = np.sort(iq_complex.real, axis=-1)

    def cum_dist(complex_row):
        state0 = complex_row.real
        state1 = complex_row.imag
        combined = np.sort(np.concatenate((state0, state1)))

        # Compute the indices where elements in state0 and state1 would be inserted in combined
        idx_state0 = np.searchsorted(combined, state0, side="left")
        idx_state1 = np.searchsorted(combined, state1, side="left")

        # Create a combined histogram for state0 and state1
        hist_combined = np.bincount(
            idx_state0, minlength=len(combined)
        ) + 1j * np.bincount(idx_state1, minlength=len(combined))

        return hist_combined.cumsum()

    cum_dist = (
        np.apply_along_axis(
            func1d=cum_dist,
            axis=-1,
            arr=iq_complex_sorted[0, ...] + 1j * iq_complex_sorted[1, ...],
        )
        / nb_shots
    )

    # Find the threshold for which the difference between the cumulative distribution of the two states is maximum
    argmax = np.argmax(np.abs(cum_dist.real - cum_dist.imag), axis=-1, keepdims=True)

    # Use np.take_along_axis to get the correct indices for the threshold calculation
    threshold = np.take_along_axis(
        np.concatenate((iq_complex_sorted[0, ...], iq_complex_sorted[1, ...]), axis=-1),
        argmax,
        axis=-1,
    )

    # Calculate the fidelity
    fidelity = np.take_along_axis(
        np.abs(cum_dist.real - cum_dist.imag), argmax, axis=-1
    )
    assignment_fidelity = (
        1
        - (
            1
            - np.take_along_axis(cum_dist.real, argmax, axis=-1)
            + np.take_along_axis(cum_dist.imag, argmax, axis=-1)
        )
        / 2
    )

    # Add all the results to the data with labels as subnet without "state", "iteration"
    data_fit.df = (
        data.df.drop_duplicates(
            subset=[i for i in labels if i not in ["state", "iteration"]]
        )
        .reset_index(drop=True)
        .apply(pint_to_float)
    )
    data_fit.df["rotation_angle"] = angle.flatten()
    data_fit.df["threshold"] = threshold.flatten()
    data_fit.df["fidelity"] = fidelity.flatten()
    data_fit.df["assignment_fidelity"] = assignment_fidelity.flatten()
    data_fit.df["average_state0"] = mean_gnd_state.flatten()
    data_fit.df["average_state1"] = mean_exc_state.flatten()

    if debug:
        return data_fit, cum_dist, iq_complex
    else:
        return data_fit
