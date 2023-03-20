from qibocal.auto.execute import Executor

from .builders import ActionBuilder


class AutoCalibrationBuilder(ActionBuilder):
    def __init__(self, runcard, folder=None, force=False):
        super().__init__(runcard, folder, force)
        self.executor = Executor.load(self.runcard)

    def run(self):
        self.executor.run(self.platform, self.qubits, self.folder)
        print(self.executor.history)
