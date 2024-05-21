"""Tasks execution."""

from dataclasses import dataclass
from pathlib import Path

from qibolab.platform import Platform

from qibocal.config import log

from ..protocols import Operation
from .history import History
from .operation import Routine
from .runcard import Action, Runcard, Targets
from .task import Task


@dataclass
class Executor:
    """Execute a tasks' graph and tracks its history."""

    actions: list[Action]
    """List of actions."""
    history: History
    """The execution history, with results and exit states."""
    output: Path
    """Output path."""
    targets: Targets
    """Qubits/Qubit Pairs to be calibrated."""
    platform: Platform
    """Qubits' platform."""
    update: bool = True
    """Runcard update mechanism."""

    # TODO: find a more elegant way to pass everything
    @classmethod
    def load(
        cls,
        card: Runcard,
        output: Path,
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

    def run(self, mode):
        """Actual execution.

        The platform's update method is called if:
        - self.update is True and task.update is None
        - task.update is True
        """
        for action in self.actions:
            task = Task(action, self._operation(action.operation))
            log.info(f"Executing mode {mode.name} on {task.id}.")
            completed = task.run(
                platform=self.platform,
                targets=self.targets,
                folder=self.output,
                mode=mode,
            )
            self.history.push(completed)

            if mode.name in ["autocalibration", "fit"] and self.platform is not None:
                completed.update_platform(platform=self.platform, update=self.update)

            yield completed.task.id

    def _operation(self, name: str) -> Routine:
        """Retrieve routine."""
        # probe plugins first, to allow shadowing builtins
        ...

        try:
            # then builtins
            return Operation[name].value
        except KeyError:
            # eventually failed, if not found
            raise RuntimeError(f"Operation '{name}' not found.")
