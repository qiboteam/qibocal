import datetime
import json
from pathlib import Path

from qibo.backends import set_backend
from qibolab.serialize import dump_platform

from ..auto.execute import run
from ..auto.mode import AUTOCALIBRATION
from ..auto.output import Metadata, Output
from ..auto.runcard import Runcard
from .report import report


def autocalibrate(runcard: Runcard, folder: Path, force, update):
    """Autocalibration.

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    set_backend(backend=runcard.backend, platform=runcard.platform)
    # rename for brevity
    backend = runcard.backend_obj
    platform = runcard.platform_obj
    # generate output folder
    path = Output.mkdir(folder, force)

    # generate meta
    meta = Metadata.generate(path.name, backend, str(platform))

    # dump platform
    if backend.name == "qibolab":
        (path / PLATFORM).mkdir(parents=True, exist_ok=True)
        dump_platform(platform, path / PLATFORM)

    # dump action runcard
    runcard.dump(folder)
    # dump meta
    (path / META).write_text(json.dumps(meta, indent=4))

    # connect and initialize platform
    if platform is not None:
        platform.connect()

    # run
    history = run(output=path, runcard=runcard, mode=AUTOCALIBRATION, update=update)

    # TODO: implement iterative dump of report...

    # stop and disconnect platform
    if platform is not None:
        platform.disconnect()

    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")
    meta = add_timings_to_meta(meta, history)
    (path / META).write_text(json.dumps(meta, indent=4))

    report(path, history)

    # dump updated runcard
    if platform is not None:
        (path / UPDATED_PLATFORM).mkdir(parents=True, exist_ok=True)
        dump_platform(platform, path / UPDATED_PLATFORM)
