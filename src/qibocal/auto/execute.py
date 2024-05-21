"""Tasks execution."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qibolab.platform import Platform

from qibocal.config import log

from .history import History
from .mode import ExecutionMode
from .runcard import Action, Runcard, Targets
from .task import Task


@dataclass
class Executor:
    """Execute a tasks' graph and tracks its history."""

    actions: list[Action]
    """List of actions."""
    history: Optional[History] = None
    """The execution history, with results and exit states."""
    output: Optional[Path] = None
    """Output path."""
    targets: Optional[Targets] = None
    """Qubits/Qubit Pairs to be calibrated."""
    platform: Optional[Platform] = None
    """Qubits' platform."""
    update: bool = True
    """Runcard update mechanism."""

    def __post_init__(self):
        # create default history
        if self.history is None:
            self.history = History({})

    # TODO: find a more elegant way to pass everything
    @classmethod
    def load(
        cls,
        card: Runcard,
        output: Path = None,
        platform: Platform = None,
        targets: Targets = None,
        update: bool = True,
    ):
        """Load execution graph and associated executor from a runcard."""

        return cls(
            actions=card.actions,
            history=History({}),
            output=output,
            platform=platform,
            targets=targets,
            update=update,
        )

    def run_protocol(self, mode: ExecutionMode, protocol: Action):
        """Run single protocol in ExecutionMode mode."""
        task = Task(protocol)
        log.info(f"Executing mode {mode.name} on {task.id}.")
        completed = task.run(
            platform=self.platform,
            targets=self.targets,
            folder=self.output,
            mode=mode,
        )

        if mode.name in ["autocalibration", "fit"] and self.platform is not None:
            completed.update_platform(platform=self.platform, update=self.update)

        self.history.push(completed)

        return completed

    def run_protocols(self, mode: ExecutionMode):
        """Actual execution.

        The platform's update method is called if:
        - self.update is True and task.update is None
        - task.update is True
        """
        for action in self.actions:
            completed = self.run_protocol(mode=mode, protocol=action)
            yield completed.task.id
