from qibocal.auto.execute import Executor
from qibocal.cli.report import report

executor = Executor.create(name="myexec", platform="dummy")

from myexec import close, init, single_shot_classification

init("test_x", force=True)

completed = single_shot_classification(nshots=1000)

close()
report(executor.path, executor.history)
