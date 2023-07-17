import datetime
import tempfile
from pathlib import Path

import yaml
from qibolab.qubits import QubitId

from qibocal.auto.execute import Executor, History
from qibocal.auto.runcard import Runcard
from qibocal.auto.task import TaskId

from .builders import ActionBuilder

META = "meta.yml"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"


class AutoCalibrationBuilder(ActionBuilder):
    def __init__(self, runcard, folder, force, update):
        super().__init__(runcard, folder, force, update)
        # TODO: modify folder in Path in ActionBuilder
        self.folder = Path(self.folder)
        self.executor = Executor.load(
            self.runcard,
            self.folder,
            self.platform,
            self.qubits,
            update,
        )

    def run(self):
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


class AutoCalibrationReportBuilder:
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

    def routine_qubits(self, task: TaskId):
        """Get local qubits parameter from Task if available otherwise use global one."""
        local_qubits = self.history[task].task.qubits
        return local_qubits if len(local_qubits) > 0 else self.qubits

    def plot(self, task: TaskId, qubit: QubitId):
        node = self.history[task]
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
