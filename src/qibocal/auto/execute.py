"""Tasks execution."""

from dataclasses import dataclass
from pathlib import Path

from qibolab.platform import Platform

from qibocal.config import log

from .history import History
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
    max_iterations: int
    """Maximum number of iterations."""
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
            max_iterations=card.max_iterations,
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
            task = Task(action)
            task.iteration = self.history.iterations(task.id)
            log.info(
                f"Executing mode {mode.name} on {task.id} iteration {task.iteration}."
            )
            completed = task.run(
                max_iterations=self.max_iterations,
                platform=self.platform,
                targets=self.targets,
                folder=self.output,
                mode=mode,
            )
            self.history.push(completed)

            if mode.name in ["autocalibration", "fit"] and self.platform is not None:
                completed.update_platform(platform=self.platform, update=self.update)

            yield completed.task.uid
