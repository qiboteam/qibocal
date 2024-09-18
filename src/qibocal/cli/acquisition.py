from pathlib import Path

from qibo.backends import construct_backend

from ..auto.history import History
from ..auto.mode import ExecutionMode
from ..auto.output import Metadata, Output
from ..auto.runcard import Runcard


def acquire(runcard: Runcard, folder: Path, force: bool):
    """Data acquisition.

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    # rename for brevity
    backend = construct_backend(backend=runcard.backend, platform=runcard.platform)
    platform = backend.platform
    if platform is None:
        raise ValueError("Qibocal requires a Qibolab platform to run.")

    # generate output folder
    path = Output.mkdir(folder, force)

    # dump action runcard
    runcard.dump(path)

    # generate meta
    meta = Metadata.generate(path.name, backend)
    meta.targets = runcard.targets
    output = Output(History(), meta, platform)
    output.dump(path)

    # connect and initialize platform
    platform.connect()

    # run
    meta.start()
    history = runcard.run(output=path, platform=platform, mode=ExecutionMode.ACQUIRE)
    meta.end()

    # TODO: implement iterative dump of report...

    # stop and disconnect platform
    platform.disconnect()

    # dump history, metadata, and updated platform
    output.history = history
    output.dump(path)
