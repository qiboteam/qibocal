from pathlib import Path

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

executor = Executor.create(name="myexec", platform="dummy")

from myexec import close, init, single_shot_classification

path, output, meta, platform = init(Path("test_x"), force=True)

completed = single_shot_classification(nshots=1000)

history = close(path, output, meta, platform)

report(path, history)
