from pydantic.dataclasses import dataclass

from .graph import Graph


@dataclass
class Validator:
    graph: Graph

    def starting_point(self):
        candidates = []

        for task in self.graph.tasks():
            if task.action.priority == 0:
                candidates.append(task)

        if len(candidates) != 1:
            return False

        return True
