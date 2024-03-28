import json
import tempfile
from functools import cached_property
from pathlib import Path

import yaml
from qibo.backends import GlobalBackend
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.mode import ExecutionMode
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import TaskId
from qibocal.config import log

META = "meta.json"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"
PLATFORM = "platform.yml"


def generate_figures_and_report(node, target):
    """Returns figures and table for report."""
    if node.results is None:
        # plot acquisition data
        return node.task.operation.report(data=node.data, fit=None, target=target)
    if target not in node.results:
        # plot acquisition data and message for unsuccessful fit
        figures = node.task.operation.report(data=node.data, fit=None, target=target)[0]
        return figures, "An error occurred when performing the fit."
    # plot acquisition and fit
    return node.task.operation.report(data=node.data, fit=node.results, target=target)


def report(path):
    """Report generation

    Arguments:

    - FOLDER: input folder.

    """
    if path.exists():
        log.warning(f"Regenerating {path}/index.html")
    # load meta
    meta = json.loads((path / META).read_text())
    # load runcard
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    # set backend, platform and qubits
    GlobalBackend.set_backend(backend=meta["backend"], platform=meta["platform"])
    backend = GlobalBackend()
    platform = backend.platform

    # load executor
    executor = Executor.load(runcard, path, targets=runcard.targets)
    # produce html
    builder = ReportBuilder(path, runcard.targets, executor, meta)
    builder.run(path)


class ReportBuilder:
    """Builder to produce html report."""

    def __init__(self, path: Path, targets, executor: Executor, metadata, history=None):
        self.path = self.title = path
        self.targets = targets
        self.executor = executor
        self.metadata = metadata
        self._history = history

    @cached_property
    def history(self):
        if self._history is None:
            list(self.executor.run(mode=ExecutionMode.report))
            return self.executor.history
        else:
            return self._history

    def routine_name(self, routine, iteration):
        """Prettify routine's name for report headers."""
        name = routine.replace("_", " ").title()
        return f"{name} - {iteration}"

    def routine_targets(self, task_id: TaskId):
        """Get local targets parameter from Task if available otherwise use global one."""
        local_targets = self.history[task_id].task.targets
        return local_targets if len(local_targets) > 0 else self.targets

    def single_qubit_plot(self, task_id: TaskId, qubit: QubitId):
        """Generate single qubit plot."""
        node = self.history[task_id]

        figures, fitting_report = generate_figures_and_report(node, qubit)
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            html_list = []
            for figure in figures:
                figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
                temp.seek(0)
                fightml = temp.read().decode("utf-8")
                html_list.append(fightml)

        all_html = "".join(html_list)
        return all_html, fitting_report

    def run(self, path):
        """Generation of html report."""
        from qibocal.web.report import create_report

        create_report(path, self)
