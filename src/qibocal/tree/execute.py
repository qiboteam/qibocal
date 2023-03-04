from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

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

    def successors(self):
        task: Task = self.current

        candidates: List[Task] = []
        if task.main is not None:
            # main task has always more priority on its own, with respect to
            # same with the same level
            candidates.append(self.graph.task(task.main))
        # add all possible successors to the list of candidates
        candidates.extend([self.graph.task(id) for id in task.next])

        return candidates

    def next(self) -> Optional[Id]:
        candidates = self.successors()

        if len(candidates) == 0:
            candidates.extend([])

        candidates = list(filter(lambda t: self.available(t), candidates))

        if len(candidates) == 0:
            return None

        # sort accord to priority
        candidates.sort(key=lambda t: -t.priority)
        # TODO: append missing successors to the list of
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
