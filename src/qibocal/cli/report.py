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

from .utils import create_qubits_dict

META = "meta.json"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"
PLATFORM = "platform.yml"


def report(path):
    """Report generation

    Arguments:

    - FOLDER: input folder.

    """
    # load meta
    meta = json.loads((path / META).read_text())
    # load runcard
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    # set backend, platform and qubits
    GlobalBackend.set_backend(backend=meta["backend"], platform=meta["platform"])
    backend = GlobalBackend()
    platform = backend.platform
    qubits = create_qubits_dict(qubits=runcard.qubits, platform=platform)

    # load executor
    executor = Executor.load(runcard, path, qubits=qubits)

    # produce html
    builder = ReportBuilder(path, qubits, executor, meta)
    builder.run(path)


class ReportBuilder:
    """Builder to produce html report."""

    def __init__(self, path: Path, qubits, executor: Executor, metadata, history=None):
        self.path = self.title = path
        self.qubits = qubits
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

    def routine_qubits(self, task_id: TaskId):
        """Get local qubits parameter from Task if available otherwise use global one."""
        local_qubits = self.history[task_id].task.qubits
        return local_qubits if len(local_qubits) > 0 else self.qubits

    def single_qubit_plot(self, task_id: TaskId, qubit: QubitId):
        """Generate single qubit plot."""
        node = self.history[task_id]
        figures, fitting_report = node.task.operation.report(
            data=node.data, fit=node.results, qubit=qubit
        )
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            html_list = []
            for figure in figures:
                figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
                temp.seek(0)
                fightml = temp.read().decode("utf-8")
                html_list.append(fightml)

        all_html = "".join(html_list)
        return all_html, fitting_report

    def plot(self, task_id: TaskId):
        """Generate plot when only acquisition data are provided."""
        node = self.history[task_id]
        data = node.task.data
        figures, fitting_report = node.task.operation.report(data)
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
