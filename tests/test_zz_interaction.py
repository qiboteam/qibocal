import json

import numpy as np
from conftest import TEST_FILE_DIR

from qibocal.protocols.ramsey.processing import fitting as ramsey_fitting
from qibocal.protocols.utils import GHZ_TO_HZ
from qibocal.protocols.zz_interaction.jazz import _jazz_fitting_process


def test_zz_coupling_fit():
    test_folder = TEST_FILE_DIR / "zz_fit_data"

    subfolders = [p for p in test_folder.iterdir() if p.is_dir()]
    for sub in subfolders:
        data_file = sub / "data.npz"
        results_file = str(sub / "results.json")
        numpy_data = np.load(data_file)

        with open(results_file) as file1:
            results = json.load(file1)

        if "jazz" in str(sub):
            for f in numpy_data.files:
                tq, cq = json.loads(f)
                dataset = numpy_data[f]

                times, signal, error, _, _ = zip(*dataset)
                times = np.asarray(times)
                signal = np.asarray(signal)
                error = np.asarray(error)

                popt, zz = _jazz_fitting_process(
                    probs=signal,
                    delays=times,
                    err=error,
                )

                assert np.allclose(
                    results['"fitted_parameters"'][f],
                    popt,
                    rtol=2.5e-2,
                )

                assert np.isclose(results['"zz"'][f][0], zz[0], rtol=2.5e-2)

        else:
            signals_freqs: dict[tuple[int, int, str], float] = {}
            popts_dict: dict[tuple[int, int, str], list[float]] = {}
            targets, controls = set(), set()
            for f in numpy_data.files:
                dataset = numpy_data[f]
                times, signal, error, _, _ = zip(*dataset)
                times = np.asarray(times)
                signal = np.asarray(signal)
                error = np.asarray(error)

                targ_q, ctrl_q, control_status = json.loads(f)

                popt, perr = ramsey_fitting(times, signal, error)
                delta_fit = [
                    -popt[2] / (2 * np.pi) * GHZ_TO_HZ,
                    perr[2] * GHZ_TO_HZ / (2 * np.pi),
                ]

                signals_freqs[targ_q, ctrl_q, control_status] = delta_fit[0]
                popts_dict[targ_q, ctrl_q, control_status] = popt
                targets |= {targ_q}
                controls |= {ctrl_q}

            for tq, cq in zip(targets, controls):
                assert np.allclose(
                    results['"fitted_parameters"'][str([tq, cq])]['"I"'],
                    popts_dict[tq, cq, "I"],
                    rtol=2.5e-2,
                )
                assert np.allclose(
                    results['"fitted_parameters"'][str([tq, cq])]['"X"'],
                    popts_dict[tq, cq, "X"],
                    rtol=2.5e-2,
                )

                zz = signals_freqs[tq, cq, "X"] - signals_freqs[tq, cq, "I"]

                assert np.isclose(
                    results['"zz"'][str([tq, cq])][0],
                    zz,
                    rtol=2.5e-2,
                )
