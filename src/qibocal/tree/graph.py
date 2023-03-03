import networkx as nx


class Graph(nx.DiGraph):
    @classmethod
    def load(cls, actions: dict):
        return cls()
