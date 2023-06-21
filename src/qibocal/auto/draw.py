"""Drawing utilities for execution graphs."""
from typing import Set

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout

from .graph import Graph
from .runcard import Id


def draw(graph: Graph, ax=None):
    """Draw a graph according to its priority."""

    rawpos = graphviz_layout(graph, prog="dot")
    assert rawpos is not None

    priorities = sorted([t.priority for t in graph.tasks()])
    xs = [p[1] for p in rawpos.values()]
    length = max(xs) - min(xs)

    ys: dict[float, Set[float]] = {}
    pos: dict[Id, tuple[float, float]] = {}
    for id, (x, y) in rawpos.items():
        depth = -priorities.index(graph.task(id).priority)
        if x in ys:
            if depth in ys[x]:
                depth = min(ys[x]) - 0.2
            ys[x].add(depth)
        else:
            ys[x] = {depth}

        newx = x + len(ys[x]) * length / 10
        pos[id] = (newx, depth)

    nx.draw(graph, pos=pos, with_labels=True, ax=ax)
