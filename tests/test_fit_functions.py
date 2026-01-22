import json
import math
from pathlib import Path

import numpy as np

from qibocal.protocols.ramsey.utils import fitting as ramsey_fitting
from qibocal.protocols.ramsey.utils import process_fit as ramsey_process_fit

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
