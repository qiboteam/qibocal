from dataclasses import dataclass

from .history import History
from .graph import Graph


@dataclass
class Executor:
    graph: Graph
    history: History

    @classmethod
    def load(cls, card):
        return cls(
            graph=Graph.load(card["actions"]),
            history=History([]),
        )

    def run(self):
        return
