import datetime
import os
import tempfile
from pathlib import Path

import yaml

from qibocal.auto.execute import Executor, History
from qibocal.auto.runcard import Runcard

from .builders import ActionBuilder, load_yaml

META = "meta.yml"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"


class AutoCalibrationBuilder(ActionBuilder):
    def __init__(self, runcard, folder=None, force=False, update=True):
        super().__init__(runcard, folder, force)
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
        from qibocal.web.report import create_autocalibration_report

        # update end time
        meta = yaml.safe_load((self.folder / META).read_text())
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(self.folder / META, "w") as file:
            yaml.dump(meta, file)

        create_autocalibration_report(self.folder, self.executor.history)

    def dump_platform_runcard(self):
        if self.platform is not None:
            self.platform.dump(self.folder / UPDATED_PLATFORM)


class AutoCalibrationReportBuilder:
    def __init__(self, path: Path, history: History):
        # FIXME: currently the title of the report is the output folder
        self.path = self.title = path
        self.metadata = yaml.safe_load((path / META).read_text())
        self.runcard = Runcard.load(path / RUNCARD)
        self.format = self.runcard.format
        self.qubits = self.runcard.qubits

        self.history = history

    def routine_name(self, routine, iteration):
        """Prettify routine's name for report headers."""
        name = routine.replace("_", " ").title()
        return f"{name} - {iteration}"

    def routine_qubits(self, routine_name, iteration):
        """Get local qubits parameter from Task if available otherwise use global one."""
        local_qubits = self.history[(routine_name, iteration)].task.parameters.qubits
        return local_qubits if local_qubits else self.qubits

    def single_qubit_plot(self, routine_name, iteration, qubit):
        node = self.history[(routine_name, iteration)]
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

    def plot(self, routine_name, iteration):
        node = self.history[(routine_name, iteration)]
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
