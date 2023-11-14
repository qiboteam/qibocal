"""Chi2 validation"""
from typing import Union

from qibolab.qubits import QubitId, QubitPairId

from qibocal.config import raise_error

from ..operation import Results
from ..status import Failure, Normal

CHI2_MAX = 0.05
"""Max value for accepting fit result."""


def check_chi2(
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
            return Failure()
    except AttributeError:
        raise_error(
            NotImplementedError, f"Chi2 validation not available for {type(results)}"
        )
