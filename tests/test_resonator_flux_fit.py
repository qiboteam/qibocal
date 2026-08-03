from pathlib import Path

import numpy as np
import pytest

from qibocal.protocols.flux_dependence.resonator_flux_dependence import (
    ResonatorFluxData,
    ResonatorFluxResults,
    _fit_function,
)
from qibocal.protocols.flux_dependence.resonator_flux_dependence import (
    _fit as resonator_flux_fit,
)
from qibocal.protocols.utils import GHZ_TO_HZ

TEST_FILE_DIR = Path(__file__).resolve().parent
PATH_TESTING_DATA = TEST_FILE_DIR / "tests_data"

RESULT_FOLDERS = sorted(PATH_TESTING_DATA.glob("resonator_flux-*"))


@pytest.mark.parametrize("results_folder", [p for p in RESULT_FOLDERS])
def test_resonator_flux_fit(results_folder):
    data = ResonatorFluxData.load(results_folder)
    expected = ResonatorFluxResults.load(results_folder)

    assert data is not None and expected is not None

    fitted = resonator_flux_fit(data)

    for qubit in expected.frequency:
        assert (
            pytest.approx(expected.frequency[qubit], abs=1e3) == fitted.frequency[qubit]
        )
        assert (
            pytest.approx(expected.sweetspot[qubit], abs=1e-3)
            == fitted.sweetspot[qubit]
        )


if __name__ == "__main__":
    """Run all fits and generate plots to help assess whether a fit is good or not.

    This is intended for use while working on the resonator flux protocol. The
    PATH_TESTING_DATA directory contains many examples with very different features, but
    for all it is clear where the sweetspot should be, and thus a good algorithm should
    be able to find the sweetspot for all of them.
    """
    from matplotlib import pyplot as plt  # not a test dependency

    output_base = Path(__file__).parent / "regression_fit_plots/resonator_flux"
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Generating comparison plots in: {output_base.absolute()}")

    for results_path in RESULT_FOLDERS:
        print(f"\n=== {results_path.name} ===")
        data = ResonatorFluxData.load(results_path)
        expected = ResonatorFluxResults.load(results_path)
        assert data is not None
        assert expected is not None

        fitted = resonator_flux_fit(data)

        # overwrite the existing results.json files, such that if we are happy with the
        # new fit, the regression test can easily be updated by committing the output
        fitted.save(results_path)

        for qubit in expected.frequency:
            freq, freq_idx = np.unique(data.data[qubit].freq, return_inverse=True)
            bias, bias_idx = np.unique(data.data[qubit].bias, return_inverse=True)
            signal = np.full((len(bias), len(freq)), np.nan)
            signal[bias_idx, freq_idx] = data.data[qubit].signal

            fit_function = _fit_function(data, qubit)

            def _filtered(params):
                """Some parameters are stored under fitted_parameters, even though they
                are not fitted."""
                return {
                    k: v
                    for k, v in params.items()
                    if k not in {"w_max", "xj", "crosstalk_element"}
                }

            fitted_params = _filtered(fitted.fitted_parameters[qubit])
            expected_params = _filtered(expected.fitted_parameters[qubit])

            fitted_frequencies = fit_function(bias, **fitted_params) * GHZ_TO_HZ
            expected_frequencies = fit_function(bias, **expected_params) * GHZ_TO_HZ

            plt.figure(figsize=(10, 6))
            plt.pcolormesh(freq, bias, signal, cmap="viridis")
            plt.xlabel("Frequency [GHz]")
            plt.ylabel("Bias [a.u.]")
            plt.colorbar(label="Signal [a.u.]")
            plt.plot(
                fitted_frequencies,
                bias,
                color="white",
                linestyle="--",
                linewidth=3.5,
                label="New fit",
            )
            plt.scatter(
                fitted.frequency[qubit],
                fitted.sweetspot[qubit],
                color="white",
                marker=".",
                s=200,
                zorder=15,
                label="New sweetspot",
            )
            plt.plot(
                expected_frequencies,
                bias,
                color="red",
                linestyle="--",
                linewidth=2,
                label="Old fit",
            )
            plt.scatter(
                expected.frequency[qubit],
                expected.sweetspot[qubit],
                color="red",
                marker=".",
                s=60,
                zorder=15,
                label="Old sweetspot",
            )
            plt.xlim(freq.min(), freq.max())
            plt.legend()
            plt.tight_layout()
            plt.savefig(output_base / f"{results_path.name}_{qubit}.png")
            plt.close()
