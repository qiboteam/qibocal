import datetime
import json
from dataclasses import asdict

import yaml
from qibo.backends import GlobalBackend, set_backend
from qibolab.serialize import dump_platform

from ..auto.execute import run
from ..auto.history import add_timings_to_meta
from ..auto.mode import ExecutionMode
from .utils import META, PLATFORM, RUNCARD, generate_meta, generate_output_folder


def acquire(runcard, folder, force):
    """Data acquisition

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """

    set_backend(backend=runcard.backend, platform=runcard.platform)
    backend = GlobalBackend()
    platform = backend.platform
    # generate output folder
    path = generate_output_folder(folder, force)

    # generate meta
    meta = generate_meta(backend, platform, path)
    # dump platform
    if backend.name == "qibolab":
        (path / PLATFORM).mkdir(parents=True, exist_ok=True)
        dump_platform(platform, path / PLATFORM)

    # dump action runcard
    (path / RUNCARD).write_text(yaml.safe_dump(asdict(runcard)))
    # dump meta
    (path / META).write_text(json.dumps(meta, indent=4))

    # connect and initialize platform
    if platform is not None:
        platform.connect()

    history = run(output=path, runcard=runcard, mode=ExecutionMode.ACQUIRE)

    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")

    # stop and disconnect platform
    if platform is not None:
        platform.disconnect()

    # dump updated meta
    meta = add_timings_to_meta(meta, history)
    (path / META).write_text(json.dumps(meta, indent=4))
