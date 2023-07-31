import datetime
import json
import tempfile
from pathlib import Path

import yaml
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor, History
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import TaskId
from qibocal.cli.utils import generate_output_folder
from qibocal.utils import allocate_qubits_pairs, allocate_single_qubits

META = "meta.json"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"
PLATFORM = "platform.yml"


class ActionBuilder:
    """Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
        update (bool): option to update platform after each routine.
    """

    def __init__(self, runcard, folder, force, update):
        # setting output folder
        self.folder = generate_output_folder(folder, force)
        # parse runcard
        self.runcard = Runcard.load(runcard)
        # store update option
        self.update = update
        # prepare output
        self.meta = self._prepare_output(runcard)
        # allocate executor
        self.executor = Executor.load(
            self.runcard,
            self.folder,
            self.platform,
            self.qubits,
            self.update,
        )

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

        with open(self.folder / RUNCARD, "w") as file:
            yaml.dump(runcard, file)

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

        (self.folder / META).write_text(json.dumps(meta, indent=4))

        return meta

    def run(self):
        """Execute protocols in runcard."""
        if self.platform is not None:
            self.platform.connect()
            self.platform.setup()
            self.platform.start()

        for data_time, result_time, task_id in self.executor.run():
            timing_task = {}
            timing_task["acquisition"] = data_time
            timing_task["fit"] = result_time
            timing_task["tot"] = data_time + result_time
            self.meta[task_id] = timing_task

            self.dump_report()

        if self.platform is not None:
            self.platform.stop()
            self.platform.disconnect()

    def dump_report(self):
        """Generate report as html."""
        from qibocal.web.report import create_report

        # update end time
        e = datetime.datetime.now(datetime.timezone.utc)
        self.meta["end-time"] = e.strftime("%H:%M:%S")
        (self.folder / META).write_text(json.dumps(self.meta, indent=4))

        create_report(self.folder, self.executor.history)

    def dump_platform_runcard(self):
        """Dump platform runcard."""
        if self.platform is not None:
            self.platform.dump(self.folder / UPDATED_PLATFORM)


class ReportBuilder:
    def __init__(self, path: Path, history: History):
        """Helper class to generate html report."""
        # FIXME: currently the title of the report is the output folder
        self.path = self.title = path
        self.metadata = json.loads((path / META).read_text())
        self.runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))
        self.qubits = self.runcard.qubits

        self.history = history

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
            node.data, node.results, qubit
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
        """ "Generate plot when only acquisition data are provided."""
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
