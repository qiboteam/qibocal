"""Execution graph and navigation utilities."""
import networkx as nx

from .runcard import Action, Id
from .task import Task


class Graph(nx.DiGraph):
    """Execution graph."""

    @classmethod
    def load(cls, actions: list[dict]):
        """Load graph from list of actions dump.

        Useful to load the graph from its description in a runcard.

        """
        return cls.from_actions([Action(**d) for d in actions])

    @classmethod
    def from_actions(cls, actions: list[Action]):
        """Load graph from list of actions.

        One node is added to the graph for each action, and the edges are
        created to represent the execution normal flow, according to the
        `action.main` and `action.next` attributes.

        """
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
        """Retrieve the graph starting point.

        Note that this method is potentially unsafe, since it is not checking
        for the existence of multiple starting points (defined by a
        `node.priority == 0` condition), and trust the graph to be a valid one.

        To validate a graph for a single starting point check
        :func:`qibocal.auto.validate.starting_point`.

        """
        for task in self.tasks():
            if task.priority == 0:
                return task.id

        raise RuntimeError()

    def task(self, id: Id) -> Task:
        """Retrieve a task from its identifier."""
        return self.nodes[id]["task"]

    def tasks(self):
        """Iterate over all tasks in the graph."""
        for node, data in self.nodes.items():
            task: Task = data["task"]
            yield task
