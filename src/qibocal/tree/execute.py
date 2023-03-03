from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from .graph import Graph
from .history import Completed, History
from .runcard import Id, Runcard
from .task import Task


@dataclass
class Executor:
    graph: Graph
    history: History
    pointer: Optional[Id] = None

    @classmethod
    def load(cls, card: Union[dict, Path]):
        runcard = Runcard.load(card)

        return cls(graph=Graph.from_actions(runcard.actions), history=History({}))

    def available(self, task: Task):
        for pred in self.graph.predecessors(task.id):
            ptask = self.graph.task(pred)

            if ptask.uid not in self.history:
                return False

        return True

    def next(self) -> Optional[Id]:
        task: Task = self.current

        candidates = []

        if task.main is not None:
            candidates.append(self.graph.task(task.main))
        candidates.extend(task.next)

        candidates = list(filter(lambda t: self.available(t), candidates))

        if len(candidates) == 0:
            return None

        candidates.sort(key=lambda t: -t.priority)
        return candidates[0].id

    @property
    def current(self):
        assert self.pointer is not None
        return self.graph.task(self.pointer)

    def run(self):
        self.pointer = self.graph.start

        while self.pointer is not None:
            task = self.current

            output = task.run()
            completed = Completed(task, output)
            self.history.push(completed)

            self.pointer = self.next()
