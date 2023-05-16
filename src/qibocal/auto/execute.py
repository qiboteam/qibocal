"""Tasks execution."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Union

from qibolab.platforms.abstract import AbstractPlatform

from .graph import Graph
from .history import Completed, History
from .runcard import Id, Runcard
from .status import Normal
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
    qubits: Optional[Qubits] = None
    """Qubits to be calibrated."""
    platform: Optional[AbstractPlatform] = None
    """Qubits' platform."""
    head: Optional[Id] = None
    """The current position."""
    pending: Set[Id] = field(default_factory=set)
    """The branched off tasks, not yet executed."""

    # TODO: find a more elegant way to pass everything
    @classmethod
    def load(
        cls,
        card: Union[dict, Path],
        output: Path,
        platform: AbstractPlatform = None,
        qubits: Qubits = None,
    ):
        """Load execution graph and associated executor from a runcard."""
        runcard = Runcard.load(card)

        return cls(
            graph=Graph.from_actions(runcard.actions),
            history=History({}),
            output=output,
            platform=platform,
            qubits=qubits,
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
        succs: List[Task] = []

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

    def run(self):
        """Actual execution."""
        self.head = self.graph.start

        while self.head is not None:
            task = self.current
            output = task.run(self.output, platform=self.platform, qubits=self.qubits)
            completed = Completed(task, output, Normal())
            self.history.push(completed)
            self.head = self.next()
            if self.platform is not None:
                self.platform.update(completed.res.update)
