import datetime
import json
from dataclasses import asdict

import yaml
from qibolab.serialize import dump_runcard

from ..auto.execute import Executor
from ..auto.history import add_timings_to_meta
from ..auto.mode import ExecutionMode
from ..cli.report import ReportBuilder
from .utils import (
    META,
    PLATFORM,
    RUNCARD,
    UPDATED_PLATFORM,
    create_qubits_dict,
    generate_meta,
    generate_output_folder,
)


def autocalibrate(runcard, folder, force, update):
    """Autocalibration

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """

    # rename for brevity
    backend = runcard.backend_obj
    platform = runcard.platform_obj
    # generate output folder
    path = generate_output_folder(folder, force)

    # allocate qubits
    qubits = create_qubits_dict(qubits=runcard.qubits, platform=platform)

    # generate meta
    meta = generate_meta(backend, platform, path)
    # dump platform
    if backend.name == "qibolab":
        dump_runcard(platform, path / PLATFORM)

    # dump action runcard
    (path / RUNCARD).write_text(yaml.safe_dump(asdict(runcard)))
    # dump meta
    (path / META).write_text(json.dumps(meta, indent=4))

    # allocate executor
    executor = Executor.load(runcard, path, platform, qubits, update)

    # connect and initialize platform
    if platform is not None:
        platform.connect()
        platform.setup()
        platform.start()

    # run protocols
    for _ in executor.run(mode=ExecutionMode.autocalibration):
        report = ReportBuilder(path, qubits, executor, meta, executor.history)
        report.run(path)

    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")

    # stop and disconnect platform
    if platform is not None:
        platform.stop()
        platform.disconnect()

    # dump updated runcard
    if platform is not None:
        dump_runcard(platform, path / UPDATED_PLATFORM)

    # dump updated meta
    meta = add_timings_to_meta(meta, executor.history)
    (path / META).write_text(json.dumps(meta, indent=4))
