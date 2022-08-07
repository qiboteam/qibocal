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
@click.option(
    "folder",
    "-o",
    type=click.Path(),
    help="Output folder. If not provided a standard name will generated.",
)
@click.option(
    "force",
    "-f",
    is_flag=True,
    help="Use --force option to overwrite the output folder.",
)
def command(runcard, folder, force=None):

    """qcvv: Quantum Calibration Verification and Validation using Qibo.

    Arguments:

     - RUNCARD: runcard with declarative inputs.
    """

    action_builder = ActionBuilder(runcard, folder, force)
    action_builder.execute()


@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "port",
    "-p",
    "--port",
    default=8050,
    type=int,
    help="Localhost port to launch dash server.",
)
def live_plot(port):
    """Real time plotting of calibration data on a dash server."""
    import socket

    from qcvv.web.app import app

    # change port if it is already used
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                break
        port += 1

    app.run_server(debug=True, port=port)


class ActionBuilder:
    """ "Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already."""

    def __init__(self, runcard, folder=None, force=False):

        path, self.folder = self._generate_output_folder(folder, force)
        self.runcard = self.load_runcard(runcard)
        self._allocate_platform(self.runcard["platform"])
        self.qubit = self.runcard["qubit"]
        self.format = self.runcard["format"]

        # Saving runcard
        self.save_runcards(path, runcard)

    @staticmethod
    def _generate_output_folder(folder, force):
        """Static method for generating the output folder.

        Args:
            folder (path): path for the output folder. If None it will be created a folder automatically
            force (bool): option to overwrite the output folder if it exists already.
        """
        if folder is None:
            import datetime
            import getpass

            e = datetime.datetime.now()
            user = getpass.getuser().replace(".", "-")
            date = e.strftime("%Y-%m-%d")
            folder = f"{date}-{'000'}-{user}"
            num = 0
            while os.path.exists(folder):
                log.warning(f"Directory {folder} already exists.")
                num += 1
                folder = f"{date}-{str(num).rjust(3, '0')}-{user}"
                log.warning(f"Trying to create directory {folder}")
        elif os.path.exists(folder) and not force:
            raise_error(RuntimeError, f"Directory {folder} already exists.")
        elif os.path.exists(folder) and force:
            log.warning(f"Deleting previous directory {folder}.")
            shutil.rmtree(os.path.join(os.getcwd(), folder))

        path = os.path.join(os.getcwd(), folder)
        log.info(f"Creating directory {folder}.")
        os.makedirs(path)
        return path, folder

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
