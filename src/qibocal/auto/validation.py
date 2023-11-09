"""Validation module."""
from dataclasses import dataclass
from enum import Enum
from typing import NewType, Optional, Union

from qibolab.qubits import QubitId, QubitPairId

from qibocal.config import log, raise_error

from .operation import Results
from .status import Broken, Normal, Status

CHI2_MAX = 0.05
"""Max value for accepting fit result."""
ValidatorId = NewType("ValidatorId", str)
"""Identifier for validator object."""


@dataclass
class Validator:
    """Generic validator object."""

    scheme: ValidatorId

    def __call__(
        self, results: Results, qubit: Union[QubitId, QubitPairId, list[QubitId]]
    ) -> Status:
        """Perform validation and returns qibocal.status.Status object."""
        raise_error(NotImplementedError)

    @classmethod
    def load(cls, params: dict):
        """Load validator from dict."""
        return cls(**params)


@dataclass
class Chi2Validator(Validator):
    """Chi2 validator."""

    chi2_max_value: Optional[float] = CHI2_MAX
    """Custom maximum chi2 value."""

    def __call__(
        self, results: Results, qubit: Union[QubitId, QubitPairId, list[QubitId]]
    ) -> Status:
        try:
            chi2 = getattr(results, "chi2")[qubit][0]
            if chi2 < self.chi2_max_value:
                return Normal()
            else:
                return Broken()
        except AttributeError:
            log.warning(f"Chi2 attribute not present in {type(results)}")
            return Broken()


class ValidationSchemes(Enum):
    """Validation schemes."""

    chi2 = Chi2Validator
