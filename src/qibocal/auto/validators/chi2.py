"""Chi2 validation"""

from typing import List, Optional, Union

import numpy as np
from pydantic import Field, validator
from pydantic.dataclasses import dataclass
from qibolab.qubits import QubitId, QubitPairId

from qibocal.config import log

from ..operation import Results

CHI2_MAX = 0.05
"""Max value for accepting fit result."""


@dataclass
class Chi2Validator:
    thresholds: Optional[List[float]] = Field(default_factory=lambda: [CHI2_MAX])
    """List of chi2 thresholds (must be ordered)."""

    @validator("thresholds")
    @classmethod
    def validate_thresholds(cls, value):
        # Custom validator for thresholds
        if sorted(value) != value:
            raise ValueError("Thresholds must be ordered.")
        return value

    def __call__(
        self,
        results: Results,
        target: Union[QubitId, QubitPairId, list[QubitId]],
    ) -> Optional[float]:
        """Performs validation of results using chi2.

        Find the threshold of the chi2 among thresholds.
        """

        try:
            chi2 = results.chi2[target][0]
            idx = np.searchsorted(self.thresholds, chi2)
            return idx - 1

        except AttributeError:
            log.error(f"Chi2 validation not available for {type(results)}")
            return None
