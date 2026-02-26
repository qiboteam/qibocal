import json
import math
from pathlib import Path

import numpy as np

from qibocal.protocols.rabi.utils import (
    fit_amplitude_function as rabi_fit_amplitude_function,
)
from qibocal.protocols.rabi.utils import fit_length_function as rabi_fit_length_function
from qibocal.protocols.ramsey.utils import fitting as ramsey_fitting
from qibocal.protocols.ramsey.utils import process_fit as ramsey_process_fit
from qibocal.protocols.utils import fallback_period, guess_period

TEST_FILE_DIR = Path(__file__).resolve().parent


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

            if any([f in str_sub for f in ["freq", "signal"]]):
                sig_min = np.min(raw_signal)
                sig_max = np.max(raw_signal)
                x_min = np.min(raw_x)
                x_max = np.max(raw_x)
                x = (raw_x - x_min) / (x_max - x_min)
                signal = (raw_signal - sig_min) / (sig_max - sig_min)
                x_lims = (x_min, x_max)
                signal_lims = (sig_min, sig_max)

            else:
                signal = raw_signal
                x = raw_x
                x_lims = (None, None)
                signal_lims = (None, None)

            period = fallback_period(guess_period(x, signal))
            median_sig = np.median(signal)
            q80 = np.quantile(signal, 0.8)
            q20 = np.quantile(signal, 0.2)
            amplitude_guess = abs(q80 - q20)

            if "amp" in str_sub:
                signal_flag = "signal" in str_sub
                pguess = [
                    median_sig,
                    amplitude_guess,
                    period,
                    np.pi / 2 if signal_flag else np.pi,
                ]
                _, _, pi_pulse_parameter = rabi_fit_amplitude_function(
                    x,
                    signal,
                    pguess,
                    sigma=errors,
                    signal=signal_flag,
                    x_limits=x_lims,
                    y_limits=signal_lims,
                )

                if isinstance(pi_pulse_parameter, list):
                    new_amplitude = pi_pulse_parameter[0]
                    true_amplitude = results['"amplitude"'][f][0]
                else:
                    new_amplitude = pi_pulse_parameter
                    true_amplitude = results['"amplitude"'][f]

                assert math.isclose(true_amplitude, new_amplitude, rel_tol=2.5e-2)

            if "length" in str_sub:
                signal_flag = "signal" in str_sub
                pguess = [
                    median_sig,
                    amplitude_guess,
                    period,
                    np.pi / 2 if signal_flag else np.pi,
                    0,
                ]
                _, _, pi_pulse_parameter = rabi_fit_length_function(
                    x,
                    signal,
                    pguess,
                    sigma=errors,
                    signal=signal_flag,
                    x_limits=x_lims,
                    y_limits=signal_lims,
                )

                if isinstance(pi_pulse_parameter, list):
                    new_duration = pi_pulse_parameter[0]
                    true_duration = results['"duration"'][f][0]
                else:
                    new_amplitude = pi_pulse_parameter
                    true_duration = results['"duration"'][f]

                assert math.isclose(true_duration, new_duration, rel_tol=2.5e-2)
