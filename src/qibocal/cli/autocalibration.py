import datetime
import json
from dataclasses import asdict

import yaml
from qibolab.serialize import dump_runcard

from ..auto.execute import Executor
from ..auto.history import add_timings_to_meta
from ..auto.mode import ExecutionMode
from ..auto.runcard import Runcard
from .utils import (
    META,
    PLATFORM,
    RUNCARD,
    UPDATED_PLATFORM,
    create_qubits_dict,
    generate_meta,
    generate_output_folder,
)


def autocalibrate(card, folder, force, update):
    """Autocalibration

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """
    # load and initialize Runcard from file
    runcard = Runcard.load(card)

    # generate output folder
    path = generate_output_folder(folder, force)
    # generate meta
    meta = generate_meta(runcard, path)

    # dump platform
    if runcard.backend == "qibolab":
        dump_runcard(runcard.platform_obj, path / PLATFORM)
    # dump action runcard
    (path / RUNCARD).write_text(yaml.safe_dump(asdict(runcard)))
    # dump meta
    (path / META).write_text(json.dumps(meta, indent=4))

    # allocate qubits, runcard and executor
    qubits = create_qubits_dict(runcard)
    platform = runcard.platform_obj
    executor = Executor.load(runcard, path, platform, qubits, update)

    # connect and initialize platform
    if platform is not None:
        platform.connect()
        platform.setup()
        platform.start()

    # run protocols
    executor.run(mode=ExecutionMode.autocalibration)

    # stop and disconnect platform
    if platform is not None:
        platform.stop()
        platform.disconnect()

    # dump updated runcard
    if platform is not None:
        dump_runcard(platform, path / UPDATED_PLATFORM)

    # dump updated meta
    meta = add_timings_to_meta(meta, executor.history)
    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")
    (path / META).write_text(json.dumps(meta, indent=4))
