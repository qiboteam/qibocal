"""Specify runcard layout, handles (de)serialization."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, NewType, Optional, Union

import yaml
from pydantic.dataclasses import dataclass
from qibo.backends import GlobalBackend
from qibo.backends.abstract import Backend
from qibolab.platform import Platform
from qibolab.qubits import QubitId, QubitPairId

from .operation import OperationId

Id = NewType("Id", str)
"""Action identifiers type."""

Targets = Union[list[QubitId], list[QubitPairId], list[tuple[QubitId, ...]]]
"""Elements to be calibrated by a single protocol."""

RUNCARD = "runcard.yml"
"""Runcard filename."""

SINGLE_ACTION = "action.yml"


@dataclass(config=dict(smart_union=True))
class Action:
    """Action specification in the runcard."""

    id: Id
    """Action unique identifier."""
    operation: OperationId
    """Operation to be performed by the executor."""
    targets: Optional[Targets] = None
    """Local qubits (optional)."""
    update: bool = True
    """Runcard update mechanism."""
    parameters: Optional[dict[str, Any]] = None
    """Input parameters, either values or provider reference."""

    def dump(self, path: Path):
        """Dump single action to yaml"""
        (path / SINGLE_ACTION).write_text(yaml.safe_dump(asdict(self)))

    @classmethod
    def load(cls, path):
        """Load action from yaml."""
        return cls(**yaml.safe_load((path / SINGLE_ACTION).read_text(encoding="utf-8")))


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
    update: bool = True

    @property
    def backend_obj(self) -> Backend:
        """Allocate backend."""
        return GlobalBackend()

    @property
    def platform_obj(self) -> Platform:
        """Allocate platform."""
        return self.backend_obj.platform

    @classmethod
    def load(cls, runcard: Union[dict, os.PathLike]):
        """Load a runcard dict or path."""
        if not isinstance(runcard, dict):
            runcard = cls.load(
                yaml.safe_load((runcard / RUNCARD).read_text(encoding="utf-8"))
            )
        return cls(**runcard)

    def dump(self, path):
        """Dump runcard object to yaml."""
        (path / RUNCARD).write_text(yaml.safe_dump(asdict(self)))
