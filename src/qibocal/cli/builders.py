import datetime
import tempfile
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path

import yaml
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import TaskId
from qibocal.cli.utils import generate_output_folder
from qibocal.utils import allocate_qubits_pairs, allocate_single_qubits

META = "meta.yml"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"
PLATFORM = "platform.yml"


@dataclass
class ExecutionMode(Enum):
    acquire = "acquire"
    fit = "fit"
    autocalibration = "auto"
    report = "report"


class PostProcessingBuilder:
    """Generic builder for performing post-processing operations."""

    def __init__(self, path: Path):
        self.path = path
        self.metadata = yaml.safe_load((path / META).read_text())
        self.runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))
        self.executor = Executor.load(
            self.runcard,
            self.path,
        )
        self.qubits = self.runcard.qubits

    def run(self, mode: ExecutionMode = None):
        """PostProcessing task."""
        self.executor.run(mode=mode)


class FitBuilder(PostProcessingBuilder):
    """Builder to run fitting on output folder"""

    def run(self, mode: ExecutionMode = ExecutionMode.fit):
        super().run(mode=mode)

        # update time in meta.yml
        e = datetime.datetime.now(datetime.timezone.utc)
        self.metadata["end-time"] = e.strftime("%H:%M:%S")
        with open(self.path / META, "w") as file:
            yaml.dump(self.metadata, file)


class ReportBuilder(PostProcessingBuilder):
    """Builder to produce html report."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.qubits = self.runcard.qubits

    @cached_property
    def history(self):
        self.executor.run(mode=ExecutionMode.report)
        return self.executor.history

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
            node.data, qubit, node.results
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

    def run(self):
        """Generation of html report."""
        from qibocal.web.report import create_report

        create_report(self.path)


class Builder:
    """Generic builder accepting a runcard."""

    def __init__(self, runcard, folder, force):
        self.folder = generate_output_folder(folder, force)
        self.runcard = Runcard.load(runcard)
        self._prepare_output(runcard)
        self.update = None
        self.executor = Executor.load(
            self.runcard, self.folder, self.platform, self.qubits, self.update
        )

    def run(self, mode: ExecutionMode):
        """Execute protocols in runcard."""

        if self.platform is not None:
            self.platform.connect()
            self.platform.setup()
            self.platform.start()

        self.executor.run(mode=mode)

        if self.platform is not None:
            self.platform.stop()
            self.platform.disconnect()

    @property
    def platform(self):
        """Qibolab's platform object."""
        return self.runcard.platform_obj

    @property
    def backend(self):
        """ "Qibo's backend object."""
        return self.runcard.backend_obj

    @property
    def qubits(self):
        """Qubits dictionary."""
        if self.platform is not None:
            if any(isinstance(i, list) for i in self.runcard.qubits):
                return allocate_qubits_pairs(self.platform, self.runcard.qubits)

            return allocate_single_qubits(self.platform, self.runcard.qubits)

        return self.runcard.qubits

    def _prepare_output(self, runcard):
        """Methods that takes care of:
        - dumping original platform
        - storing qq runcard
        - generating meta.yml
        """
        if self.backend.name == "qibolab":
            self.platform.dump(self.folder / PLATFORM)

        (self.folder / RUNCARD).write_text(yaml.dump(runcard))

        import qibocal

        e = datetime.datetime.now(datetime.timezone.utc)
        meta = {}
        meta["title"] = self.folder.name
        meta["backend"] = str(self.backend)
        meta["platform"] = str(self.backend.platform)
        meta["date"] = e.strftime("%Y-%m-%d")
        meta["start-time"] = e.strftime("%H:%M:%S")
        meta["end-time"] = e.strftime("%H:%M:%S")
        meta["versions"] = self.backend.versions  # pylint: disable=E1101
        meta["versions"]["qibocal"] = qibocal.__version__

        (self.folder / META).write_text(yaml.dump(meta))


class AcquisitionBuilder(Builder):
    """Builder for perfoming only data acquisition."""


class ActionBuilder(Builder):
    """Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
        update (bool): option to update platform after each routine.
    """

    def __init__(self, runcard, folder, force, update):
        super().__init__(runcard, folder, force)
        self.update = update

    def dump_report(self):
        report = ReportBuilder(self.folder)
        report.run()

    def dump_platform_runcard(self):
        """Dump platform runcard."""
        if self.platform is not None:
            self.platform.dump(self.folder / UPDATED_PLATFORM)
