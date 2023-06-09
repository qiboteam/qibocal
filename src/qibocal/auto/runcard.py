"""Specify runcard layout, handles (de)serialization."""
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, NewType, Optional, Union

import yaml
from pydantic import Field
from pydantic.dataclasses import dataclass
from qibo.backends import Backend, construct_backend
from qibolab.platform import Platform
from qibolab.qubits import Qubit, QubitId

from qibocal.utils import allocate_qubits

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


@dataclass(config=dict(smart_union=True, arbitrary_types_allowed=True))
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
    def load(cls, card: Union[dict, Path]):
        """Load a runcard.

        It accepts both a dictionary, or a path to a YAML file, to be first
        deserialized in a dictionary, and further loaded into an instance.

        """
        content = (
            yaml.safe_load(card.read_text(encoding="utf-8"))
            if isinstance(card, Path)
            else card
        )

        return cls(**content)
