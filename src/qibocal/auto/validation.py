"""Validation module."""
from dataclasses import dataclass
from typing import NewType, Optional, Union

from qibolab.qubits import QubitId, QubitPairId

from . import validators
from .operation import Results
from .status import Status

ValidatorId = NewType("ValidatorId", str)
"""Identifier for validator object."""


@dataclass
class Validator:
    """Generic validator object."""

    scheme: ValidatorId
    parameters: Optional[dict] = None

    def __call__(
        self, results: Results, qubit: Union[QubitId, QubitPairId, list[QubitId]]
    ) -> Status:
        validator = getattr(validators, self.scheme)
        return validator(results, qubit, **self.parameters)
