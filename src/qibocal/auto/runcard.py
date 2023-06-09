"""Specify runcard layout, handles (de)serialization."""
from functools import cached_property
from typing import Any, Dict, List, NewType, Optional, Union

from pydantic import Field
from pydantic.dataclasses import dataclass
from qibo.backends import Backend, construct_backend
from qibolab.platform import Platform
from qibolab.qubits import QubitId

from .operation import OperationId

Id = NewType("Id", str)
"""Action identifiers type."""


@dataclass(config=dict(smart_union=True))
class Action:
    """Action specification in the runcard."""

    id: Id
    """Action unique identifier."""
    operation: Optional[OperationId] = None
    """Operation to be performed by the executor."""
    main: Optional[Id] = None
    """Main subsequent for action in normal flow."""
    next: Optional[Union[List[Id], Id]] = None
    """Alternative subsequent actions, branching from the current one."""
    priority: Optional[int] = None
    """Priority level, determining the execution order."""
    qubits: list[QubitId] = Field(default_factory=list)
    """Local qubits (optional)."""
    update: bool = True
    """Runcard update mechanism."""
    parameters: Optional[Dict[str, Any]] = None
    """Input parameters, either values or provider reference."""

    def __hash__(self) -> int:
        """Each action is uniquely identified by its id."""
        return hash(self.id)


@dataclass(config=dict(smart_union=True))
class Runcard:
    """Structure of an execution runcard."""

    actions: List[Action]
    qubits: List[QubitId] = Field(default_facotry=list)
    backend: str = "qibolab"
    platform: str = "dummy"
    # TODO: pass custom runcard (?)

    @cached_property
    def backend_obj(self) -> Backend:
        """Allocate backend."""
        return construct_backend(self.backend, self.platform)

    @property
    def platform_obj(self) -> Platform:
        """Allocate platform."""
        return self.backend_obj.platform

    @classmethod
    def load(cls, params: dict):
        """Load a runcard (dict)."""
        return cls(**params)
