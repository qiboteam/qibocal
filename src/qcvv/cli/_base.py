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
@click.argument("platform", metavar="PLATFORM_NAME")
@click.argument("action_runcard", metavar="ACTION_CARD", type=click.Path(exists=True))
@click.argument("folder", type=click.Path())
@click.option("--force", is_flag=True)
def command(platform, action_runcard, folder, force=None):

    """qcvv: Quantum Calibration Verification and Validation using Qibo."""
    from qibo.backends import construct_backend

    platform = construct_backend("qibolab", platform=platform).platform

    if os.path.exists(folder) and not force:
        raise_error(
            RuntimeError, "Calibration folder with the same name already exists."
        )
    else:
        from qibolab.paths import qibolab_folder

        runcard = qibolab_folder / "runcards" / f"{platform}.yml"
        path = os.path.join(os.getcwd(), folder)
        if not force:
            log.info(f"Creating directory {path}.")
            os.makedirs(path)
        shutil.copy(runcard, f"{path}/")

    platform.connect()
    platform.setup()
    platform.start()
    action_builder = ActionBuilder(platform, action_runcard, folder)
    action_builder.execute_action()

    platform.stop()
    platform.disconnect()


class ActionBuilder:
    def __init__(self, platform, path, folder):
        self.platform = platform
        self.runcard = self.load_action_runcard(path)
        self.folder = folder

    def _build_single_action(self, name):
        """This private method finds the correct function in the qcvv and
        checks if any of the arguments are missing in the runcard"""
        f = getattr(calibrations, name)
        if hasattr(f, "prepare"):
            self.output = f.prepare(name=f.__name__, folder=self.folder)
        sig = inspect.signature(f)
        params = self.runcard[name]
        for param in list(sig.parameters)[1:]:
            if param not in params:
                raise_error(AttributeError, f"Missing parameter {param} in runcard.")
        return f, params

    def load_action_runcard(self, path):
        """Loading action runcard"""
        with open(path, "r") as file:
            action_settings = yaml.safe_load(file)
        return action_settings

    def execute_action(self):
        """Method to obtain the calibration routine with the arguments
        checked"""
        for action in self.runcard:
            routine, args = self._build_single_action(action)
            results = self.get_result(routine, args)

    def get_result(self, routine, arguments):
        """Method to execute the routine and saving data through
        final action"""
        results = routine(self.platform, **arguments)
        if hasattr(routine, "final_action"):
            return routine.final_action(results, self.output)
        return results


@click.command(context_settings=CONTEXT_SETTINGS)
@click.argument("path", metavar="PLOT_PATH", type=click.Path())
def live_plot(path):
    from qcvv.live import app
    app.path = path # Bad hack to pass data path to all dash callbacks
    app.run_server(debug=True)
