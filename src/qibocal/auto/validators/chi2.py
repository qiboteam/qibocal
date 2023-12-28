"""Chi2 validation"""
from typing import Union

from qibolab.qubits import QubitId, QubitPairId

from qibocal.config import log

from ..operation import Results

CHI2_MAX = 0.05
"""Max value for accepting fit result."""


def check_chi2(
    results: Results,
    target: Union[QubitId, QubitPairId, list[QubitId]],
    thresholds: [CHI2_MAX],
):
    """Performs validation of results using chi2.

    It assesses that chi2 is below the chi2_max_value threshold.
    """

    try:
        chi2 = getattr(results, "chi2")[target][0]
        for threshold in sorted(thresholds):
            if chi2 < threshold:
                return thresholds.index(threshold)

        return None
    except AttributeError:
        log.warning(f"Chi2 validation not available for {type(results)}")
        return None
