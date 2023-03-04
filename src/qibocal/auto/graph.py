from typing import Dict, List, Set, Tuple

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

    def draw(self, ax=None):
        from networkx.drawing.nx_pydot import graphviz_layout

        rawpos = graphviz_layout(self, prog="dot")
        assert rawpos is not None

        priorities = sorted([t.priority for t in self.tasks()])
        xs = [p[1] for p in rawpos.values()]
        length = max(xs) - min(xs)

        ys: Dict[float, Set[float]] = {}
        pos: Dict[Id, Tuple[float, float]] = {}
        for id, (x, y) in rawpos.items():
            depth = -priorities.index(self.task(id).priority)
            if x in ys:
                if depth in ys[x]:
                    depth = min(ys[x]) - 0.2
                ys[x].add(depth)
            else:
                ys[x] = {depth}

            newx = x + len(ys[x]) * length / 10
            pos[id] = (newx, depth)

        nx.draw(self, pos=pos, with_labels=True, ax=ax)
