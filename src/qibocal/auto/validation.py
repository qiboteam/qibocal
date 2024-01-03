"""Validation module."""
from dataclasses import dataclass, field
from typing import Callable, NewType, Optional, Union

from qibolab.qubits import QubitId, QubitPairId

from ..config import log, raise_error
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
        """Validation function in validators module."""
        try:
            return VALIDATORS[self.scheme]
        except KeyError:
            raise_error(KeyError, f"Validator {self.scheme} not available.")

    def validate(
        self, results: Results, target: Target
    ) -> tuple[Union[Status, str], Optional[dict]]:
        """Perform validation of target in results.

        Possible Returns are:
            - (Failure, None) which stops the execution.
            - (Normal, None) which corresponds to the normal flow
            - (task, dict) which moves the head to task using parameters in dict.
        """
        index = self.method(results=results, target=target, **self.parameters)
        # If index is None -> status is Failure
        # if index is 0 -> Normal Status
        # else: jump to corresponding outcomes
        if index == None:
            log.error("Stopping execution due to error in validation.")
            return Failure(), None
        elif index == 0:
            # for chi2 (to be generalized for other validators):
            # if chi2 is less than first threshold the status is normal
            return Normal(), None

        # else we return outcomes [index-1] since outcomes outcomes[i] is
        # the output of thresholds[index+1], given that for the first threshold
        # the status is Normal.
        return self.outcomes[index - 1]
