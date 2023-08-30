"""Track execution history."""

from .runcard import Id
from .task import Completed


class History(dict[tuple[Id, int], Completed]):
    """Execution history.

    This is not only used for logging and debugging, but it is an actual part
    of the execution itself, since later routines can retrieve the output of
    former ones from here.

    """

    def push(self, completed: Completed):
        """Adding completed task to history."""
        self[(completed.task.id, completed.task.iteration)] = completed

        # FIXME: I'm not too sure why but the code doesn't work anymore
        # with this line. We can postpone it when we will have the
        # ExceptionalFlow.
        # completed.task.iteration += 1

    # TODO: implemet time_travel()
