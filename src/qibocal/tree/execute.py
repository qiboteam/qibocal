from dataclasses import dataclass
from pathlib import Path
from typing import Union

import yaml

from .graph import Graph
from .history import History


@dataclass
class Executor:
    graph: Graph
    history: History

    @classmethod
    def load(cls, card: Union[dict, Path]):
        content = (
            yaml.safe_load(card.read_text(encoding="utf-8"))
            if isinstance(card, Path)
            else card
        )

        return cls(
            graph=Graph.load(content),
            history=History([]),
        )

    def run(self):
        return
