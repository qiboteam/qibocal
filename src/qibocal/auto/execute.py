"""Tasks execution."""

from dataclasses import dataclass
from pathlib import Path

from qibolab.platform import Platform

from qibocal.config import log

from .experiment import Experiment
from .history import History
from .mode import ExecutionMode
from .runcard import Runcard, Targets


@dataclass
class Executor:
    """Execute a tasks' graph and tracks its history."""

    actions: list[Experiment]
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
        for experiment in self.actions:

            experiment.iteration = self.history.iterations(experiment.id)
            log.info(
                f"Executing mode {mode.name} on {experiment.id} iteration {experiment.iteration}."
            )

            if mode in [ExecutionMode.acquire, ExecutionMode.autocalibration]:
                experiment.acquire(self.platform, self.targets)
                experiment.dump(self.output)

            if mode in [ExecutionMode.fit, ExecutionMode.autocalibration]:
                experiment.fit()
                experiment.dump(self.output)

                if experiment.update:
                    experiment.update_platform(self.platform)
                    experiment.dump(self.output)

            self.history.push(experiment)

            yield experiment.uid
