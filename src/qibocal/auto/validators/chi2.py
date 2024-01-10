"""Chi2 validation"""
from typing import Optional, Union

import numpy as np
from qibolab.qubits import QubitId, QubitPairId

from qibocal.config import log

from ..operation import Results

CHI2_MAX = 0.05
"""Max value for accepting fit result."""


def check_chi2(
    results: Results,
    target: Union[QubitId, QubitPairId, list[QubitId]],
    thresholds: Optional[list] = None,
) -> Optional[float]:
    """Performs validation of results using chi2.

    Find the threshold of the chi2 among thresholds.
    """

    if thresholds is None:
        thresholds = [CHI2_MAX]

    try:
        chi2 = results.chi2[target][0]
        idx = np.searchsorted(thresholds, chi2)
        return idx - 1

    except AttributeError:
        log.error(f"Chi2 validation not available for {type(results)}")
        return None
