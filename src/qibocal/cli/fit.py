import datetime
import json
import pathlib
import shutil

import yaml
from qibolab.serialize import dump_runcard

from qibocal.config import log, raise_error

from ..auto.execute import run
from ..auto.history import add_timings_to_meta
from ..auto.mode import ExecutionMode
from ..auto.operation import RESULTSFILE
from ..auto.runcard import Runcard
from .utils import META, RUNCARD, UPDATED_PLATFORM


def fit(input_path, update, output_path, force):
    """Post-processing analysis

    Arguments:

    - input_path: input folder.
    - update: perform platform update
    - output_path: new folder with data and fit
    """

    if output_path is not None:
        if output_path.exists():
            if force is False:
                raise_error(RuntimeError, f"Directory {output_path} already exists.")
            # overwrite output_path
            log.warning(f"Deleting previous directory {output_path}.")
            shutil.rmtree(pathlib.Path.cwd() / output_path)
        path = shutil.copytree(input_path, output_path)
    else:
        if len(list(input_path.glob(f"**/{RESULTSFILE}.json"))) > 0:
            if force:
                log.warning(f"Overwriting fit in {input_path}.")
            else:
                raise_error(
                    RuntimeError, f"Directory {input_path} contains fitting results."
                )
        path = input_path

    meta = json.loads((path / META).read_text())
    # load runcard
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    # run
    history = run(
        output=path,
        runcard=runcard,
        mode=ExecutionMode.FIT,
    )

    # update time in meta
    meta = add_timings_to_meta(meta, history)
    e = datetime.datetime.now(datetime.timezone.utc)
    meta["end-time"] = e.strftime("%H:%M:%S")

    # dump updated runcard
    if runcard.platform_obj is not None and update:  # pragma: no cover
        # cannot test update since dummy may produce wrong values and trigger errors
        (path / UPDATED_PLATFORM).mkdir(parents=True, exist_ok=True)
        dump_runcard(runcard.platform_obj, path / UPDATED_PLATFORM)

    # dump json

    (path / META).write_text(json.dumps(meta, indent=4))
