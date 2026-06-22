"""Specify runcard layout, handles (de)serialization."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from pydantic.dataclasses import dataclass
from qibo.backends import construct_backend

from qibocal.calibration.platform import CalibrationPlatform

from .. import protocols
from .execute import Executor
from .history import History
from .mode import ExecutionMode
from .output import Metadata
from .task import Action, Targets

RUNCARD = "runcard.yml"
"""Runcard filename."""


@dataclass
class Runcard:
    """Structure of an execution runcard."""

    actions: list[Action]
    """List of action to be executed."""
    targets: Targets | None = None
    """Qubits to be calibrated.

    If `None` the protocols will be executed on all qubits
    available in the platform.
    """
    backend: str = "qibolab"
    """Qibo backend."""
    platform: str = "mock"
    """Qibolab platform."""
    update: bool = True

    @classmethod
    def load(cls, runcard: dict[str, Any] | Path):
        """Load a runcard dict or path."""
        if not isinstance(runcard, dict):
            return cls(yaml.safe_load((runcard / RUNCARD).read_text(encoding="utf-8")))
        return cls(**runcard)

    def dump(self, path):
        """Dump runcard object to yaml."""
        (path / RUNCARD).write_text(yaml.safe_dump(asdict(self)), encoding="utf-8")

    def run(
        self,
        output: Path,
        platform: CalibrationPlatform,
        mode: ExecutionMode,
        update: bool = True,
    ) -> History:
        """Run runcard and dump to output."""
        targets = self.targets if self.targets is not None else list(platform.qubits)
        history = History.load(output)
        update = update and self.update
        backend = construct_backend(backend="qibolab", platform=platform)
        instance = Executor(
            history=history,
            platform=platform,
            targets=targets,
            update=update,
            path=output,
            meta=Metadata.generate(backend),
        )

        for action in self.actions:
            instance.run_protocol(
                protocol=getattr(protocols, action.operation),
                parameters=action,
                mode=mode,
                output=output,
            )
        instance.history.dump(output)
        return instance.history
