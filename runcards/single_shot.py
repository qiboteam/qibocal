from pathlib import Path

from qibo.backends import construct_backend

from qibocal.auto.history import History
from qibocal.auto.output import Metadata, Output
from qibocal.cli.report import report
from qibocal.routines import single_shot_classification

folder = Path("test_x")
force = True

backend = construct_backend(backend="qibolab", platform="qw11q")
platform = backend.platform

# generate output folder
path = Output.mkdir(folder, force)

# generate meta
meta = Metadata.generate(path.name, backend)
output = Output(History(), meta, platform)
output.dump(path)

# connect and initialize platform
platform.connect()

# run
meta.start()

completed = single_shot_classification(nshots=1000)

# stop and disconnect platform
platform.disconnect()

history = executor.history
# dump history, metadata, and updated platform
output.history = history
output.dump(path)

report(path, history)
