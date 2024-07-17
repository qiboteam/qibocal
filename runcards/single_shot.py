from pathlib import Path

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

executor = Executor.create(name="myexec", platform="dummy")

from myexec import init, single_shot_classification

path, output, meta, platform = init(Path("test_x"), force=True)

completed = single_shot_classification(nshots=1000)

meta.end()

# stop and disconnect platform
platform.disconnect()

history = executor.history
# dump history, metadata, and updated platform
output.history = history
output.dump(path)

report(path, history)
