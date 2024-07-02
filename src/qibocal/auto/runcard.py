"""Specify runcard layout, handles (de)serialization."""

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic.dataclasses import dataclass
from qibolab.platform import Platform

from .. import protocols
from .execute import Executor
from .history import History
from .mode import ExecutionMode
from .task import Action, Targets

RUNCARD = "runcard.yml"
"""Runcard filename."""


@dataclass
class Runcard:
    """Structure of an execution runcard."""

    actions: list[Action]
    """List of action to be executed."""
    targets: Optional[Targets] = None
    """Qubits to be calibrated.

    If `None` the protocols will be executed on all qubits
    available in the platform.
    """
    backend: str = "qibolab"
    """Qibo backend."""
    platform: str = os.environ.get("QIBO_PLATFORM", "dummy")
    """Qibolab platform."""
    update: bool = True

    @classmethod
    def load(cls, runcard: Union[dict[str, Any], Path]):
        """Load a runcard dict or path."""
        if not isinstance(runcard, dict):
            return cls(yaml.safe_load((runcard / RUNCARD).read_text(encoding="utf-8")))
        return cls(**runcard)

    def dump(self, path):
        """Dump runcard object to yaml."""
        (path / RUNCARD).write_text(yaml.safe_dump(asdict(self)))

    def run(
        self, output: Path, platform: Platform, mode: ExecutionMode, update: bool = True
    ) -> History:
        """Run runcard and dump to output."""
        targets = self.targets if self.targets is not None else list(platform.qubits)
        history = History.load(output)
        update = update and self.update
        instance = Executor(
            history=history, platform=platform, targets=targets, update=update
        )

        for action in self.actions:
            instance.run_protocol(
                protocol=getattr(protocols, action.operation),
                parameters=action,
                mode=mode,
            )
            instance.history.flush(output)
        return instance.history
