import datetime
import json

import yaml
from qibo.backends import GlobalBackend
from qibolab.serialize import dump_runcard

from ..auto.execute import Executor
from ..auto.history import add_timings_to_meta
from ..auto.mode import ExecutionMode
from ..auto.runcard import Runcard
from .utils import META, RUNCARD, UPDATED_PLATFORM, create_qubits_dict


def fit(path, update):
    """Post-processing analysis

    Arguments:

    - FOLDER: input folder.

    """
    # load path, meta, runcard and executor
    meta = yaml.safe_load((path / META).read_text())
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    GlobalBackend.set_backend(backend=meta["backend"], platform=meta["platform"])
    backend = GlobalBackend()
    platform = backend.platform
    qubits = create_qubits_dict(qubits=runcard.qubits, platform=platform)
    executor = Executor.load(
        runcard, path, update=update, platform=platform, qubits=qubits
    )

    # perform post-processing
    list(executor.run(mode=ExecutionMode.fit))

    # dump updated runcard
    if platform is not None and update:
        dump_runcard(platform, path / UPDATED_PLATFORM)

    # update time in meta.yml
    meta = add_timings_to_meta(meta, executor.history)
    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")
    (path / META).write_text(json.dumps(meta, indent=4))
