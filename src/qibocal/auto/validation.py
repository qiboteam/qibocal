"""Validation module."""
from dataclasses import dataclass
from typing import NewType, Optional, Union

from qibolab.qubits import QubitId, QubitPairId

from ..config import log
from .operation import Results
from .status import Status
from .validators import VALIDATORS

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
        log.info(
            f"Performing validation in qubit {qubit} of {results.__class__.__name__} using {self.scheme} scheme."
        )
        validator = VALIDATORS[self.scheme]
        return validator(results, qubit, **self.parameters)
