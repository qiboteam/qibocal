from pathlib import Path

from qibo.backends import construct_backend

from ..auto.execute import run
from ..auto.history import History
from ..auto.mode import AUTOCALIBRATION
from ..auto.output import Metadata, Output
from ..auto.runcard import Runcard
from .report import report


def autocalibrate(runcard: Runcard, folder: Path, force, update):
    """Autocalibration.

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    # rename for brevity
    backend = construct_backend(backend=runcard.backend, platform=runcard.platform)
    platform = backend.platform
    # generate output folder
    path = Output.mkdir(folder, force)

    # generate meta
    meta = Metadata.generate(path.name, backend, str(platform))
    output = Output(History(), meta, platform)
    output.dump(path)

    # dump action runcard
    runcard.dump(path)

    # connect and initialize platform
    if platform is not None:
        platform.connect()

    # run
    history = run(output=path, runcard=runcard, mode=AUTOCALIBRATION, update=update)

    # TODO: implement iterative dump of report...

    # stop and disconnect platform
    if platform is not None:
        platform.disconnect()

    meta.end()
    output.dump(path)

    report(path, history)
