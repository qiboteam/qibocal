"""Specify runcard layout, handles (de)serialization."""

import os
from functools import cached_property
from typing import Any, NewType, Optional, Union

from pydantic.dataclasses import dataclass
from qibo.backends import Backend, GlobalBackend
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from .operation import OperationId
from .validation import Validator

Id = NewType("Id", str)
"""Action identifiers type."""

Targets = Union[list[QubitId], list[QubitPairId], list[tuple[QubitId, ...]]]
"""Elements to be calibrated by a single protocol."""

MAX_ITERATIONS = 5
"""Default max iterations."""


@dataclass(config=dict(smart_union=True))
class Action:
    """Action specification in the runcard."""

    id: Id
    """Action unique identifier."""
    operation: Optional[OperationId] = None
    """Operation to be performed by the executor."""
    main: Optional[Id] = None
    """Main subsequent for action in normal flow."""
    next: Optional[Union[list[Id], Id]] = None
    """Alternative subsequent actions, branching from the current one."""
    priority: Optional[int] = None
    """Priority level, determining the execution order."""
    targets: Optional[Targets] = None
    """Local qubits (optional)."""
    update: bool = True
    """Runcard update mechanism."""
    validator: Optional[Validator] = None
    """Define validation scheme and parameters."""
    parameters: Optional[dict[str, Any]] = None
    """Input parameters, either values or provider reference."""

    def __hash__(self) -> int:
        """Each action is uniquely identified by its id."""
        return hash(self.id)


@dataclass(config=dict(smart_union=True))
class Runcard:
    """Structure of an execution runcard."""

    actions: list[Action]
    """List of action to be executed."""
    targets: Optional[Targets] = None
    """Qubits to be calibrated."""
    backend: str = "qibolab"
    """Qibo backend."""
    platform: str = os.environ.get("QIBO_PLATFORM", "dummy")
    """Qibolab platform."""
    max_iterations: int = MAX_ITERATIONS
    """Maximum number of iterations."""

    @cached_property
    def backend_obj(self) -> Backend:
        """Allocate backend."""
        GlobalBackend.set_backend(self.backend, self.platform)
        return GlobalBackend()

    @property
    def platform_obj(self) -> Platform:
        """Allocate platform."""
        return self.backend_obj.platform

    @classmethod
    def load(cls, params: dict):
        """Load a runcard (dict)."""
        return cls(**params)
