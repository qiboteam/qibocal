from pathlib import Path

from qibo.backends import construct_backend

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.auto.output import Metadata, Output
from qibocal.cli.report import report

folder = Path("test_x")
force = True

backend = construct_backend(backend="qibolab", platform="dummy")
platform = backend.platform
if platform is None:
    raise ValueError("Qibocal requires a Qibolab platform to run.")

executor = Executor(name="myexec", history=History(), platform=platform, targets=[0])

# generate output folder
path = Output.mkdir(folder, force)

# generate meta
meta = Metadata.generate(path.name, backend)
output = Output(History(), meta, platform)
output.dump(path)

from myexec import single_shot_classification

# connect and initialize platform
platform.connect()

# run
meta.start()

completed = single_shot_classification(nshots=1000)

meta.end()

# stop and disconnect platform
platform.disconnect()

history = executor.history
# dump history, metadata, and updated platform
output.history = history
output.dump(path)

report(path, history)
