import shutil
from pathlib import Path
from typing import Optional

import yaml

from ..auto.execute import run
from ..auto.mode import ExecutionMode
from ..auto.operation import RESULTSFILE
from ..auto.output import Output
from ..auto.runcard import RUNCARD, Runcard
from ..config import log, raise_error


def mkoutput(input: Path, output: Optional[Path], force: bool):
    if output is not None:
        if output.exists():
            if not force:
                raise_error(RuntimeError, f"Directory {output} already exists.")
            # overwrite output_path
            log.warning(f"Deleting previous directory {output}.")
            shutil.rmtree(output)
        return shutil.copytree(input, output)

    if len(list(input.glob(f"**/{RESULTSFILE}.json"))) > 0:
        if force:
            log.warning(f"Overwriting fit in {input}.")
        else:
            raise_error(RuntimeError, f"Directory {input} contains fitting results.")
    return input


def fit(input_path, update, output_path, force):
    """Post-processing analysis.

    Arguments:

    - input_path: input folder.
    - update: perform platform update
    - output_path: new folder with data and fit
    """
    path = mkoutput(input_path, output_path, force)

    output = Output.load(path)
    # load runcard
    runcard = Runcard.load(yaml.safe_load((path / RUNCARD).read_text()))

    # run
    output.history = run(
        output=path,
        runcard=runcard,
        mode=ExecutionMode.FIT,
        platform=output.platform,
        update=update,
    )

    # update time in meta
    output.dump(path)
