"""Tasks execution."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set

from qibolab.platform import Platform

from qibocal.config import log

from .graph import Graph
from .history import History
from .runcard import Id, Runcard
from .task import Qubits, Task


@dataclass
class Executor:
    """Execute a tasks' graph and tracks its history."""

    graph: Graph
    """The graph to be executed."""
    history: History
    """The execution history, with results and exit states."""
    output: Path
    """Output path."""
    qubits: Qubits
    """Qubits to be calibrated."""
    platform: Platform
    """Qubits' platform."""
    update: bool = True
    """Runcard update mechanism."""
    head: Optional[Id] = None
    """The current position."""
    pending: Set[Id] = field(default_factory=set)
    """The branched off tasks, not yet executed."""

    # TODO: find a more elegant way to pass everything
    @classmethod
    def load(
        cls,
        card: Runcard,
        output: Path,
        platform: Platform = None,
        qubits: Qubits = None,
        update: bool = True,
    ):
        """Load execution graph and associated executor from a runcard."""

        return cls(
            graph=Graph.from_actions(card.actions),
            history=History({}),
            output=output,
            platform=platform,
            qubits=qubits,
            update=update,
        )

    def available(self, task: Task):
        """Check if a task has all dependencies satisfied."""
        for pred in self.graph.predecessors(task.id):
            ptask = self.graph.task(pred)

            if ptask.uid not in self.history:
                return False

        return True

    def successors(self, task: Task):
        """Retrieve successors of a specified task."""
        succs: list[Task] = []

        if task.main is not None:
            # main task has always more priority on its own, with respect to
            # same with the same level
            succs.append(self.graph.task(task.main))
        # add all possible successors to the list of successors
        succs.extend([self.graph.task(id) for id in task.next])

        return succs

    def next(self) -> Optional[Id]:
        """Resolve the next task to be executed.

        Returns `None` if the execution is completed.

        .. todo::

            consider transforming this into an iterator, and this could be its
            `__next__` method, raising a `StopIteration` instead of returning
            `None`.
            it would be definitely more Pythonic...

        """
        candidates = self.successors(self.current)

        if len(candidates) == 0:
            candidates.extend([])

        candidates = list(filter(lambda t: self.available(t), candidates))

        # sort accord to priority
        candidates.sort(key=lambda t: t.priority)
        if len(candidates) != 0:
            self.pending.update([t.id for t in candidates[1:]])
            return candidates[0].id

        availables = list(
            filter(lambda t: self.available(self.graph.task(t)), self.pending)
        )
        if len(availables) == 0:
            if len(self.pending) == 0:
                return None
            raise RuntimeError("")

        selected = min(availables, key=lambda t: self.graph.task(t).priority)
        self.pending.remove(selected)
        return selected

    @property
    def current(self):
        """Retrieve current task, associated to the `head` pointer."""
        assert self.head is not None
        return self.graph.task(self.head)

    def run(self, mode):
        """Actual execution.

        The platform's update method is called if:
        - self.update is True and task.update is None
        - task.update is True
        """
        self.head = self.graph.start
        while self.head is not None:
            task = self.current
            log.info(f"Executing mode {mode.name} on {task.id}.")
            completed = task.run(
                platform=self.platform,
                qubits=self.qubits,
                folder=self.output,
                mode=mode,
            )
            self.history.push(completed)
            self.head = self.next()
            update = self.update and task.update
            if (
                mode.name in ["autocalibration", "fit"]
                and self.platform is not None
                and update
            ):
                task.update_platform(results=completed.results, platform=self.platform)
            yield task.uid
