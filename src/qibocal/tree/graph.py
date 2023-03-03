from typing import List

import networkx as nx

from qibocal.tree.runcard import Action


class Graph(nx.DiGraph):
    @classmethod
    def load(cls, actions: List[dict]):
        return cls.from_actions([Action(**d) for d in actions])

    @classmethod
    def from_actions(cls, actions: List[Action]):
        dig = cls()

        for action in actions:
            dig.add_node(action.id, action=action)

        for node in dig.nodes:
            __import__("pdb").set_trace()

        return dig
