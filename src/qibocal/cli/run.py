from pathlib import Path

from qibo.backends import construct_backend

from ..auto.history import History
from ..auto.mode import AUTOCALIBRATION
from ..auto.output import Metadata, Output
from ..auto.runcard import Runcard
from .report import report


def protocols_execution(runcard: Runcard, folder: Path, force, update):
    """Autocalibration.

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
    history = runcard.run(
        output=path,
        platform=platform,
        mode=AUTOCALIBRATION,
        update=update,
    )
    meta.end()

    # TODO: implement iterative dump of report...

    # stop and disconnect platform
    platform.disconnect()

    # dump history, metadata, and updated platform
    output.history = history
    output.dump(path)

    report(path, history)
