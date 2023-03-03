from typing import List, Optional

import networkx as nx

from .runcard import Action, Id
from .task import Task


class Graph(nx.DiGraph):
    @classmethod
    def load(cls, actions: List[dict]):
        return cls.from_actions([Action(**d) for d in actions])

    @classmethod
    def from_actions(cls, actions: List[Action]):
        dig = cls()

        for action in actions:
            dig.add_node(action.id, task=Task(action))

        for task in dig.tasks():
            if task.main is not None:
                dig.add_edge(task.id, task.main, main=True)

            for succ in task.next:
                dig.add_edge(task.id, succ)

        return dig

    @property
    def start(self) -> Id:
        for task in self.tasks():
            if task.priority == 0:
                return task.id

        raise RuntimeError()

    def task(self, id: Id) -> Task:
        return self.nodes[id]["task"]

    def tasks(self):
        for node, data in self.nodes.items():
            task: Task = data["task"]
            yield task
