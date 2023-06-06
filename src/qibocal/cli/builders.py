import datetime
import shutil
import tempfile
from pathlib import Path

import yaml
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor, History
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import TaskId
from qibocal.cli.utils import generate_output_folder, load_yaml
from qibocal.config import raise_error
from qibocal.utils import allocate_qubits

META = "meta.yml"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"
PLATFORM = "platform.yml"


class ActionBuilder:
    """Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
        update (bool): option to
    """

    def __init__(self, runcard, folder, force, update):
        # setting output folder
        self.folder = generate_output_folder(folder, force)
        # parse runcard
        self.runcard = Runcard.load(Path(runcard))
        self.update = update
        self._prepare_output(runcard)

    @property
    def platform(self):
        return self.runcard.platform

    @property
    def backend(self):
        return self.runcard.backend

    @property
    def qubits(self):
        if self.platform is not None:
            return allocate_qubits(self.platform, self.runcard.qubits)

        return self.runcard.qubits

    def _prepare_output(self, runcard):
        self.platform.dump(self.folder / PLATFORM)
        shutil.copy(runcard, self.folder / RUNCARD)

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

        with open(self.folder / META, "w") as file:
            yaml.dump(meta, file)

    def run(self):
        self.executor = Executor.load(
            self.runcard,
            self.folder,
            self.platform,
            self.qubits,
            self.update,
        )
        if self.platform is not None:
            self.platform.connect()
            self.platform.setup()
            self.platform.start()

        self.executor.run()

        if self.platform is not None:
            self.platform.stop()
            self.platform.disconnect()

    def dump_report(self):
        """Dump report."""
        from qibocal.web.report import create_report

        # update end time
        meta = yaml.safe_load((self.folder / META).read_text())
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(self.folder / META, "w") as file:
            yaml.dump(meta, file)

        create_report(self.folder, self.executor.history)

    def dump_platform_runcard(self):
        """Dump platform runcard."""
        if self.platform is not None:
            self.platform.dump(self.folder / UPDATED_PLATFORM)


class ReportBuilder:
    def __init__(self, path: Path, history: History):
        # FIXME: currently the title of the report is the output folder
        self.path = self.title = path
        self.metadata = yaml.safe_load((path / META).read_text())
        self.runcard = Runcard.load(path / RUNCARD)
        self.qubits = self.runcard.qubits

        self.history = history

    def routine_name(self, routine, iteration):
        """Prettify routine's name for report headers."""
        name = routine.replace("_", " ").title()
        return f"{name} - {iteration}"

    def routine_qubits(self, task_uid: TaskId):
        """Get local qubits parameter from Task if available otherwise use global one."""
        local_qubits = self.history[task_uid].task.qubits
        return local_qubits if len(local_qubits) > 0 else self.qubits

    def single_qubit_plot(self, task_uid: TaskId, qubit: QubitId):
        node = self.history[task_uid]
        data = node.task.data
        figures, fitting_report = node.task.operation.report(data, node.res, qubit)
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            html_list = []
            for figure in figures:
                figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
                temp.seek(0)
                fightml = temp.read().decode("utf-8")
                html_list.append(fightml)

        all_html = "".join(html_list)
        return all_html, fitting_report

    def plot(self, task_uid: TaskId):
        node = self.history[task_uid]
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
