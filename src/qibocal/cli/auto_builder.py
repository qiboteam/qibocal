import os

from qibocal.auto.execute import Executor
from qibocal.config import raise_error

from .builders import ActionBuilder, load_yaml


class AutoCalibrationBuilder(ActionBuilder):
    def __init__(self, runcard, folder=None, force=False):
        super().__init__(runcard, folder, force)
        self.executor = Executor.load(self.runcard)

    def run(self):
        self.executor.run(self.platform, self.qubits, self.folder)
        print(self.executor.history)


class AutoCalibrationReportBuilder:
    def __init__(self, path, actions=None):
        self.path = path
        self.metadata = load_yaml(os.path.join(path, "meta.yml"))

        # find proper path title
        base, self.title = os.path.join(os.getcwd(), path), ""
        while self.title in ("", "."):
            base, self.title = os.path.split(base)

        self.runcard = load_yaml(os.path.join(path, "runcard.yml"))
        self.format = self.runcard.get("format")
        self.qubits = self.runcard.get("qubits")

        # create calibration routine objects
        # (could be incorporated to :meth:`qibocal.cli.builders.ActionBuilder._build_single_action`)
        self.routines = []
        if actions is None:
            actions = self.runcard.get("actions")

        for action in actions:
            if hasattr(hardware, action):
                routine = getattr(hardware, action)
            elif hasattr(gateset.niGSC, action):
                routine = niGSCactionParser(self.runcard, self.path, action)
                routine.load_plot()
            else:
                raise_error(ValueError, f"Undefined action {action} in report.")

            if not hasattr(routine, "plots"):
                routine.plots = []
            self.routines.append(routine)
