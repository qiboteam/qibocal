import datetime
import os

import yaml

from qibocal.auto.execute import Executor
from qibocal.config import raise_error

from .builders import ActionBuilder, load_yaml


class AutoCalibrationBuilder(ActionBuilder):
    def __init__(self, runcard, folder=None, force=False):
        super().__init__(runcard, folder, force)
        self.executor = Executor.load(self.runcard)

    def run(self):
        self.executor.run(self.qubits, self.platform, self.folder)

    def dump_report(self):
        from qibocal.web.report import create_autocalibration_report

        # update end time
        meta = load_yaml(f"{self.folder}/meta.yml")
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(f"{self.folder}/meta.yml", "w") as file:
            yaml.dump(meta, file)

        create_autocalibration_report(self.folder, self.executor.history)


class AutoCalibrationReportBuilder:
    def __init__(self, path, history):
        self.path = path
        self.metadata = load_yaml(os.path.join(path, "meta.yml"))

        # find proper path title
        base, self.title = os.path.join(os.getcwd(), path), ""
        while self.title in ("", "."):
            base, self.title = os.path.split(base)

        self.runcard = load_yaml(os.path.join(path, "runcard.yml"))
        self.format = self.runcard.get("format")
        self.qubits = self.runcard.get("qubits")

        self.history = history

    def get_routine_name(self, routine, iteration):
        """Prettify routine's name for report headers."""
        return routine.replace("_", " ").title() + f" {iteration}"

    def plot(self, routine_name, iteration, qubit):
        import tempfile

        node = self.history[(routine_name, iteration)]
        data = node.task.operation.data_type.load_data(
            self.path, "data", f"{routine_name}_{iteration}", "csv", "data"
        )
        figures = node.task.operation.report(data, node.res, qubit)
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            html_list = []
            for figure in figures:
                figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
                temp.seek(0)
                fightml = temp.read().decode("utf-8")
                html_list.append(fightml)

        all_html = "".join(html_list)
        return all_html
