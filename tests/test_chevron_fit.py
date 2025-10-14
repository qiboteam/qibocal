from pathlib import Path

import numpy as np
import pytest

from qibocal.protocols import chevron, chevron_signal

CHEVRON_DATA = Path(__file__).parent / "chevron_data"


@pytest.mark.parametrize("chevron_type", ["prob", "signal"])
def test_chevron_fit(chevron_type):
    """Testing chevron fit on real data."""
    protocol = chevron_signal if chevron_type == "signal" else chevron
    data = protocol.data_type.load(CHEVRON_DATA / chevron_type)
    target_results = protocol.results_type.load(CHEVRON_DATA / chevron_type)

    results, _ = protocol.fit(data)

    assert results.native == target_results.native

    for pair in data.sorted_pairs:
        np.testing.assert_allclose(
            results.duration[pair], target_results.duration[pair]
        )
        np.testing.assert_allclose(
            results.half_duration[pair], target_results.half_duration[pair]
        )
        np.testing.assert_allclose(
            results.amplitude[pair], target_results.amplitude[pair]
        )
        np.testing.assert_allclose(
            results.fitted_parameters[pair],
            target_results.fitted_parameters[pair],
            rtol=1e-2,
        )
