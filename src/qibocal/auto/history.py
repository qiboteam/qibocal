"""Track execution history."""
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .operation import DATAFILE, RB_DATAFILE, RESULTSFILE, Data, Results
from .runcard import Id
from .status import Status
from .task import Task


@dataclass
class Completed:
    """A completed task."""

    task: Task
    """A snapshot of the task when it was completed.

    .. todo::

        once tasks will be immutable, a separate `iteration` attribute should
        be added

    """
    status: Status
    """Protocol status."""
    folder: Path
    """Folder with data and results."""
    _data: Optional[Data] = None
    """Protocol data."""
    _results: Optional[Results] = None
    """Fitting output."""
    data_time: Optional[float] = 0
    """Protocol data."""
    results_time: Optional[float] = 0
    """Fitting output."""

    @property
    def datapath(self):
        """Path contaning data and results file for task."""
        path = self.folder / "data" / f"{self.task.id}_{self.task.iteration}"
        if not path.is_dir():
            path.mkdir(parents=True)
        return path

    @property
    def results(self):
        """Access task's results."""
        if not (self.datapath / RESULTSFILE).is_file():
            return None

        if self._results is None:
            Results = self.task.operation.results_type
            self._results = Results.load(self.datapath)
        return self._results

    @results.setter
    def results(self, results: Results):
        """Set and store results."""
        self._results = results
        self._results.save(self.datapath)

    @property
    def data(self):
        """Access task's data."""
        if (
            not (self.datapath / DATAFILE).is_file()
            and not (self.datapath / RB_DATAFILE).is_file()
        ):
            return None
        if self._data is None:
            Data = self.task.operation.data_type
            self._data = Data.load(self.datapath)
        return self._data

    @data.setter
    def data(self, data: Data):
        """Set and store data."""
        self._data = data
        self._data.save(self.datapath)

    def __post_init__(self):
        self.task = copy.deepcopy(self.task)


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
