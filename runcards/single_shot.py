from pathlib import Path

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

executor = Executor.create(name="myexec", platform="dummy")
path = Path("test_x")

from myexec import close, init, single_shot_classification

output, meta = init(path, force=True)

completed = single_shot_classification(nshots=1000)

close(path, output, meta)
report(path, executor.history)
