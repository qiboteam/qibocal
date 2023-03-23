import datetime
import os
import tempfile
from pathlib import Path

import yaml

from qibocal.auto.execute import Executor, History

from .builders import ActionBuilder, load_yaml

META = "meta.yml"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"


class AutoCalibrationBuilder(ActionBuilder):
    def __init__(self, runcard, folder=None, force=False):
        super().__init__(runcard, folder, force)
        # TODO: modify folder in Path in ActionBuilder
        self.folder = Path(self.folder)
        self.executor = Executor.load(
            self.runcard, self.platform, self.qubits, self.folder
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
        self.platform.dump(self.folder / UPDATED_PLATFORM)


class AutoCalibrationReportBuilder:
    def __init__(self, path: Path, history: History):
        self.path = path
        self.metadata = yaml.safe_load((path / META).read_text())

        # find proper path title
        base, self.title = os.getcwd() / path, ""
        while self.title in ("", "."):
            base, self.title = os.path.split(base)

        self.runcard = yaml.safe_load((path / RUNCARD).read_text())
        self.format = self.runcard.get("format")
        self.qubits = self.runcard.get("qubits")

        self.history = history

    def routine_name(self, routine, iteration):
        """Prettify routine's name for report headers."""
        name = routine.replace("_", " ").title()
        return f"{name} - {iteration}"

    def plot(self, routine_name, iteration, qubit):
        node = self.history[(routine_name, iteration)]
        data = node.task.operation.data_type.load_data(
            self.path, "data", f"{routine_name}_{iteration}", "csv", "data"
        )
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
