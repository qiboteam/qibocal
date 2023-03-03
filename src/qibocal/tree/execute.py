from dataclasses import dataclass
from pathlib import Path
from typing import Union

from .graph import Graph
from .history import History
from .runcard import Runcard


@dataclass
class Executor:
    graph: Graph
    history: History

    @classmethod
    def load(cls, card: Union[dict, Path]):
        runcard = Runcard.load(card)

        return cls(graph=Graph.from_actions(runcard.actions), history=History([]))

    def run(self):
        return
