"""Track execution history."""

from pathlib import Path

from qibocal.config import raise_error

from .runcard import Id
from .task import Completed


def add_timings_to_meta(meta, history):
    for task_id in history:
        completed = history[task_id]
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


class History(dict[Id, Completed]):
    """Execution history.

    This is not only used for logging and debugging, but it is an actual part
    of the execution itself, since later routines can retrieve the output of
    former ones from here.

    """

    @classmethod
    def load(cls, path: Path):
        """To be defined"""
        instance = cls()
        for protocol in (path / "data").glob("*"):
            instance.push(Completed.load(protocol))
        return instance

    def push(self, completed: Completed):
        """Adding completed task to history."""
        if completed.task.id in self:
            # patch to make sure that calling fit after acquire works
            if self[completed.task.id]._results is not None:
                raise_error(KeyError, f"{completed.task.id} already in History.")
        self[completed.task.id] = completed

    # TODO: implemet time_travel()
