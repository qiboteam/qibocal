"""Routine-specific method for post-processing data acquired."""
import lmfit
import numpy as np
from scipy.optimize import curve_fit

from qibocal.config import log
from qibocal.data import Data
from qibocal.fitting.utils import cos, exp, flipping, lorenzian, parse, rabi, ramsey


def lorentzian_fit(
    data, x, y, qubits, resonator_type, labels, fit_file_name=None, lo_freqs=None
):
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

            -   When using ``resonator_spectroscopy`` the expected labels are [`resonator_freq`, `peak voltage`], where `resonator_freq` is the estimated frequency of the resonator, and `peak_voltage` the peak of the Lorentzian

            -   when using ``qubit_spectroscopy`` the expected labels are [`qubit_freq`, `peak voltage`], where `qubit_freq` is the estimated frequency of the qubit

        fit_file_name (str): file name, ``None`` is the default value.

    Returns:

        A ``Data`` object with the following keys

            - **labels[0]**: peak voltage
            - **labels[1]**: frequency
            - **labels[2]**: readout frequency
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
                label = "qubit_freq"
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

                data = DataUnits(quantities={"frequency": "Hz"})

                mydict = {"frequency[Hz]": x, "MSR[V]": noisy_lorentzian}

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
                labels[2],
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
                labels[2],
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
        )
        voltages = qubit_data[voltages_keys[0]].pint.to(voltages_keys[1]).pint.magnitude

        # Create a lmfit model for fitting equation defined in resonator_peak
        model_Q = lmfit.Model(lorenzian)

        # Guess parameters for Lorentzian max or min
        if (resonator_type == "3D" and labels[0] == "resonator_freq") or (
            resonator_type != "3D" and labels[0] == "qubit_freq"
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
            log.warning("lorentzian_fit: the fitting was not successful")
            return data_fit

        # get the values for postprocessing and for legend.
        f0 = fit_res.best_values["center"]
        BW = fit_res.best_values["sigma"] * 2
        Q = abs(f0 / BW)
        peak_voltage = (
            fit_res.best_values["amplitude"] / (fit_res.best_values["sigma"] * np.pi)
            + fit_res.best_values["offset"]
        )

        freq = f0

        intermediate_freq = 0
        if lo_freqs != None:
            intermediate_freq = freq - lo_freqs[qubit]

        data_fit.add(
            {
                labels[0]: freq,
                labels[1]: peak_voltage,
                labels[2]: intermediate_freq,
                "popt0": fit_res.best_values["amplitude"],
                "popt1": fit_res.best_values["center"],
                "popt2": fit_res.best_values["sigma"],
                "popt3": fit_res.best_values["offset"],
                "qubit": qubit,
            }
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
        name=f"fits",
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
        except:
            log.warning("rabi_fit: the fitting was not succesful")
            return data_fit

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
    return data_fit


def ramsey_fit(
    data, x, y, qubits, resonator_type, qubit_freqs, sampling_rate, offset_freq, labels
):
    r"""
    Fitting routine for Ramsey experiment. The used model is

    .. math::

        y = p_0 + p_1 sin \Big(2 \pi p_2 x + p_3 \Big) e^{-x p_4}.

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
        name=f"fits",
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

        if resonator_type == "3D":
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / times.values[np.argmin(voltages.values)],
                np.pi / 2,
                500e-9,
            ]
        else:
            pguess = [
                np.mean(voltages.values),
                np.max(voltages.values) - np.min(voltages.values),
                0.5 / times.values[np.argmax(voltages.values)],
                np.pi / 2,
                500e-9,
            ]
        try:
            popt, pcov = curve_fit(
                ramsey, times.values, voltages.values, p0=pguess, maxfev=2000000
            )
            delta_fitting = popt[2]
            delta_phys = int((delta_fitting * sampling_rate) - offset_freq)
            corrected_qubit_frequency = int(qubit_freqs[qubit] + delta_phys)
            t2 = 1.0 / popt[4]
        except:
            log.warning("ramsey_fit: the fitting was not succesful")
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
                "qubit": qubit,
            }
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

        except:
            log.warning("t1_fit: the fitting was not succesful")
            return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            labels[0]: t1,
            "qubit": qubit,
        }
    )
    return data_fit


def flipping_fit(data, x, y, qubits, resonator_type, pi_pulse_amplitude, labels):
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
        pi_pulse_amplitude(float): corrected pi pulse amplitude
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
            corrected_amplitude = amplitude_correction_factor * pi_pulse_amplitude
            # angle = (niter * 2 * np.pi / popt[2] + popt[3]) / (1 + 4 * niter)
            # amplitude_delta = angle * 2 / np.pi * pi_pulse_amplitude
        except:
            log.warning("flipping_fit: the fitting was not succesful")
            return data_fit

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
        nqubits (int): total number of qubits
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

        except:
            log.warning("drag_tuning_fit: the fitting was not succesful")
            return data_fit

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
    return data_fit


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

        except:
            log.warning("spin_echo_fit: the fitting was not succesful")
            return data_fit

    data_fit.add(
        {
            "popt0": popt[0],
            "popt1": popt[1],
            "popt2": popt[2],
            labels[0]: t2,
            "qubit": qubit,
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
