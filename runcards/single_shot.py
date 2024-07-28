from qibocal.auto.execute import Executor
from qibocal.cli.report import report

with Executor.open("myexec", path="test_x", platform="dummy", force=True) as executor:
    from myexec import single_shot_classification

    completed = single_shot_classification(nshots=1000)

report(executor.path, executor.history)
