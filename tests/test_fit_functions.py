import json
import math

import numpy as np
from conftest import TEST_FILE_DIR

from qibocal.protocols.rabi.utils import (
    fit_amplitude_function as rabi_fit_amplitude_function,
)
from qibocal.protocols.rabi.utils import fit_length_function as rabi_fit_length_function
from qibocal.protocols.rabi.utils import (
    rabi_initial_guess,
)
from qibocal.protocols.ramsey.processing import fitting as ramsey_fitting
from qibocal.protocols.ramsey.processing import process_fit as ramsey_process_fit


def test_ramsey_fit():
    test_folder = TEST_FILE_DIR / "ramsey_fit_data"

    subfolders = [p for p in test_folder.iterdir() if p.is_dir()]
    for sub in subfolders:
        data_file = sub / "data.npz"
        results_file = str(sub / "results.json")
        json_file = str(sub / "data.json")

        numpy_data = np.load(data_file)
        for f in numpy_data.files:
            dataset = numpy_data[f]
            times, signal = zip(*dataset)

            with open(results_file) as file1:
                results = json.load(file1)
            with open(json_file) as file2:
                data = json.load(file2)

            fit_params, fit_err = ramsey_fitting(times, signal)
            new_freq, t2, delta_signal, delta_fit, _ = ramsey_process_fit(
                fit_params, fit_err, data['"qubit_freqs"'][f], data['"detuning"']
            )

            assert math.isclose(
                results['"frequency"'][f][0], new_freq[0], rel_tol=2.5e-2
            )
            assert math.isclose(results['"t2"'][f][0], t2[0], rel_tol=2.5e-2)
            assert math.isclose(
                results['"delta_phys"'][f][0], delta_signal[0], rel_tol=2.5e-2
            )
            assert math.isclose(
                results['"delta_fitting"'][f][0], delta_fit[0], rel_tol=2.5e-2
            )


def test_rabi_fit():
    test_folder = TEST_FILE_DIR / "rabi_fit_data"

    subfolders = [p for p in test_folder.iterdir() if p.is_dir()]
    for sub in subfolders:
        data_file = sub / "data.npz"
        results_file = str(sub / "results.json")

        str_sub = str(sub)

        numpy_data = np.load(data_file)
        for f in numpy_data.files:
            dataset = numpy_data[f]
            if len(dataset[0]) == 3:
                raw_x, raw_signal, errors = zip(*dataset)
            else:
                raw_x, raw_signal = zip(*dataset)
                errors = None

            with open(results_file) as file1:
                results = json.load(file1)

            signal_flag = "signal" in str_sub

            if "freq" in str_sub:
                sig_min = np.min(raw_signal)
                sig_max = np.max(raw_signal)
                x_min = np.min(raw_x)
                x_max = np.max(raw_x)
                x = (raw_x - x_min) / (x_max - x_min)
                signal = (raw_signal - sig_min) / (sig_max - sig_min)
            else:
                signal = raw_signal
                x = raw_x

            rabi_flag = "amp" if "amp" in str_sub else "length"
            fit_param = '"amplitude"' if "amp" in str_sub else '"length"'
            fit_func = (
                rabi_fit_amplitude_function
                if rabi_flag == "amp"
                else rabi_fit_length_function
            )

            pguess = rabi_initial_guess(x, signal, rabi_flag, signal_flag)

            _fit_params, _, pi_pulse_parameter = fit_func(
                x,
                signal,
                pguess,
                sigma=errors,
            )

            if isinstance(pi_pulse_parameter, list):
                new_param = pi_pulse_parameter[0]
                true_param = results[fit_param][f][0]
            else:
                new_param = pi_pulse_parameter
                true_param = results[fit_param][f]

            assert math.isclose(new_param, true_param, rel_tol=2.5e-2)
