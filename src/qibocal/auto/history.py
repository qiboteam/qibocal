"""Track execution history."""

from .runcard import Id
from .task import Completed


def add_timings_to_meta(meta, history):
    for task_id, iteration in history:
        completed = history[(task_id, iteration)]
        if task_id not in meta:
            meta[task_id] = {}

        if "acquisition" not in meta[task_id] and completed.data_time > 0:
            meta[task_id]["acquisition"] = completed.data_time
        if "fit" not in meta[task_id] and completed.results_time > 0:
            meta[task_id]["fit"] = completed.results_time
        if "acquisition" in meta[task_id]:
            meta[task_id]["tot"] = meta[task_id]["acquisition"]
        if "fit" in meta[task_id]:
            meta[task_id]["tot"] += meta[task_id]["fit"]

    return meta


class History(dict[tuple[Id, int], Completed]):
    """Execution history.

    This is not only used for logging and debugging, but it is an actual part
    of the execution itself, since later routines can retrieve the output of
    former ones from here.

    """

    def push(self, completed: Completed):
        """Adding completed task to history."""
        self[(completed.task.id, completed.task.iteration)] = completed

        # TODO: re-implement this
        # completed.task.iteration += 1

    # TODO: implemet time_travel()
