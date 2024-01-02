"""Validation module."""
from dataclasses import dataclass, field
from typing import Callable, NewType, Optional, Union

from qibolab.qubits import QubitId, QubitPairId

from ..config import raise_error
from .operation import Results
from .status import Failure, Normal, Status
from .validators import VALIDATORS

ValidatorId = NewType("ValidatorId", str)
"""Identifier for validator object."""

Target = Union[QubitId, QubitPairId, list[QubitId]]
"""Protocol target."""


@dataclass
class Validator:
    """Generic validator object."""

    scheme: ValidatorId
    """Validator present in validators module."""
    parameters: Optional[dict] = field(default_factory=dict)
    """"Validator parameters."""
    outcomes: Optional[list[tuple[str, dict]]] = field(default_factory=list)
    """Depending on the validation we jump into one of the possible nodes."""

    # TODO: think of a better name
    @property
    def method(self) -> Callable[[Results, Target], Union[Status, str]]:
        """Validation function."""
        try:
            return VALIDATORS[self.scheme]
        except AttributeError:
            raise_error(AttributeError, f"Validator {self.scheme} not available.")

    def validate(self, results: Results, target: Target):
        index = self.method(results=results, target=target, **self.parameters)
        # If index is None -> status is Failure
        # if index is 0 -> Normal Status
        # else: jump to corresponding outcomes
        if index == None:
            raise_error(ValueError, "Stopping execution due to error in validation.")
            return Failure()
        elif index == 0:
            return Normal()
        return self.outcomes[index - 1][0]
