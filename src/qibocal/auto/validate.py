"""Extra tools to validate an execution graph."""
from .graph import Graph


def starting_point(graph: Graph):
    """Check graph starting point.

    Since a graph is designed to work with a unique starting point, the user
    has to make sure to specify a single one, since the starting point is
    identified only by a 0 priority (i.e. top-priority).

    The execution of a graph with multiple starting points has to be considered
    undefined behavior.

    """
    candidates = []

    for task in graph.tasks():
        if task.action.priority == 0:
            candidates.append(task)

    if len(candidates) != 1:
        return False

    return True
