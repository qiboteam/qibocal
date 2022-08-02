# -*- coding: utf-8 -*-
"""Adds global CLI options."""
import inspect
import os
import shutil

import click
import yaml

from qcvv import calibrations
from qcvv.config import log, raise_error

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("runcard", metavar="RUNCARD", type=click.Path(exists=True))
@click.argument("folder", type=click.Path())
@click.option(
    "--force", is_flag=True, help="Use --force option to overwrite the output folder."
)
def command(runcard, folder, force=None):

    """qcvv: Quantum Calibration Verification and Validation using Qibo.

    Arguments:

     - FOLDER: output folder.

     - RUNCARD: runcard with declarative inputs.
    """

    action_builder = ActionBuilder(runcard, folder, force)
    action_builder.execute()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("path", metavar="DATA_FOLDER", type=click.Path())
def live_plot(path):
    """Real time plotting of calibration data on a dash server.

    DATA_FOLDER is the path to the folder that contains the
    data to be plotted.
    """
    from qcvv.live import app, serve_layout

    # Hack to pass data path to the layout
    app.layout = lambda: serve_layout(path)
    app.run_server(debug=True)


class ActionBuilder:
    """ "Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already."""

    def __init__(self, runcard, folder, force):
        # Creating output folder
        if os.path.exists(folder) and not force:
            raise_error(
                RuntimeError, "Calibration folder with the same name already exists."
            )
        elif os.path.exists(folder) and force:
            log.info(f"Overwriting folder {folder}.")
            shutil.rmtree(os.path.join(os.getcwd(), folder))

        path = os.path.join(os.getcwd(), folder)
        log.info(f"Creating directory {path}.")
        os.makedirs(path)

        self.folder = folder
        self.force = force
        self.runcard = self.load_runcard(runcard)
        self._allocate_platform(self.runcard["platform"])
        self.qubit = self.runcard["qubit"]
        self.format = self.runcard["format"]

        # Saving runcard
        self.save_runcards(path, runcard)

    def _allocate_platform(self, platform_name):
        """Allocate the platform using Qibolab."""
        from qibo.backends import construct_backend

        self.platform = construct_backend("qibolab", platform=platform_name).platform

    def save_runcards(self, path, runcard):
        """Save the output runcards."""
        import datetime

        import qibo
        import qibolab
        from qibolab.paths import qibolab_folder

        import qcvv

        platform_runcard = (
            qibolab_folder / "runcards" / f"{self.runcard['platform']}.yml"
        )
        shutil.copy(platform_runcard, f"{path}/platform.yml")

        e = datetime.datetime.now()
        meta = {}
        meta["date"] = e.strftime("%Y-%m-%d %H:%M:%S")
        meta["versions"] = {
            "qibo": qibo.__version__,
            "qibolab": qibolab.__version__,
            "qcvv": qcvv.__version__,
        }
        with open(f"{path}/meta.yml", "w") as f:
            yaml.dump(meta, f)

        shutil.copy(runcard, f"{path}/runcard.yml")

    def _build_single_action(self, name):
        """Helper method to parse the actions in the runcard."""
        f = getattr(calibrations, name)
        if hasattr(f, "prepare"):
            self.output = f.prepare(name=f.__name__, folder=self.folder)
        sig = inspect.signature(f)
        params = self.runcard["actions"][name]
        for param in list(sig.parameters)[2:-1]:
            if param not in params:
                raise_error(AttributeError, f"Missing parameter {param} in runcard.")
        return f, params

    @staticmethod
    def load_runcard(path):
        """Method to load the runcard."""
        with open(path, "r") as file:
            runcard = yaml.safe_load(file)
        return runcard

    def execute(self):
        """Method to execute sequentially all the actions in the runcard."""
        self.platform.connect()
        self.platform.setup()
        self.platform.start()
        for action in self.runcard["actions"]:
            routine, args = self._build_single_action(action)
            results = self.get_result(routine, args)
        self.platform.stop()
        self.platform.disconnect()

    def get_result(self, routine, arguments):
        """Method to execute a single action and retrieving the results."""
        results = routine(self.platform, self.qubit, **arguments)
        if hasattr(routine, "final_action"):
            return routine.final_action(results, self.output, self.format)
        return results
