"""Specify runcard layout, handles (de)serialization."""

import os
from typing import Any, NewType, Optional, Union

from pydantic.dataclasses import dataclass
from qibo.backends import Backend, GlobalBackend
from qibo.transpiler.pipeline import Passes
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from .operation import OperationId

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
    targets: Optional[Targets] = None
    """Local qubits (optional)."""
    update: bool = True
    """Runcard update mechanism."""
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
    """Qubits to be calibrated.
       If `None` the protocols will be executed on all qubits
       available in the platform."""
    backend: str = "qibolab"
    """Qibo backend."""
    platform: str = os.environ.get("QIBO_PLATFORM", "dummy")
    """Qibolab platform."""
    max_iterations: int = MAX_ITERATIONS
    """Maximum number of iterations."""

    def __post_init__(self):
        if self.targets is None and self.platform_obj is not None:
            self.targets = list(self.platform_obj.qubits)

    @property
    def backend_obj(self) -> Backend:
        """Allocate backend."""
        GlobalBackend.set_backend(self.backend, platform=self.platform)
        backend = GlobalBackend()
        # FIXME: remove this latin abomination
        backend.transpiler = Passes(connectivity=backend.platform.topology)
        backend.transpiler.passes = backend.transpiler.passes[-1:]
        return backend

    @property
    def platform_obj(self) -> Platform:
        """Allocate platform."""
        return self.backend_obj.platform

    @classmethod
    def load(cls, params: dict):
        """Load a runcard (dict)."""
        return cls(**params)
