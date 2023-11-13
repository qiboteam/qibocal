"""Chi2 validation"""
from typing import Union

from qibolab.qubits import QubitId, QubitPairId

from qibocal.config import log

from ..operation import Results
from ..status import Broken, Normal

CHI2_MAX = 0.05
"""Max value for accepting fit result."""


def chi2(
    results: Results,
    qubit: Union[QubitId, QubitPairId, list[QubitId]],
    chi2_max_value=None,
):
    """Performs validation of results using chi2.

    It assesses that chi2 is below the chi2_max_value threshold.

    """

    if chi2_max_value is None:
        chi2_max_value = CHI2_MAX
    try:
        chi2 = getattr(results, "chi2")[qubit][0]
        if chi2 < chi2_max_value:
            return Normal()
        else:
            return Broken()
    except AttributeError:
        log.warning(f"Chi2 attribute not present in {type(results)}")
        return Broken()
