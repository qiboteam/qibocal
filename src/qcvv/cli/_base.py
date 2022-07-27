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
@click.argument("action_runcard", metavar="ACTION_CARD", type=click.Path(exists=True))
@click.argument("folder", type=click.Path())
@click.option("--force", is_flag=True)
def command(action_runcard, folder, force=None):

    """qcvv: Quantum Calibration Verification and Validation using Qibo."""

    action_builder = ActionBuilder(action_runcard, folder, force)
    action_builder.execute()
    action_builder.save_runcard()


class ActionBuilder:
    def __init__(self, action_runcard, folder, force):
        if os.path.exists(folder) and not force:
            raise_error(
                RuntimeError, "Calibration folder with the same name already exists."
            )
        path = os.path.join(os.getcwd(), folder)
        # if not self.force:
        log.info(f"Creating directory {path}.")
        os.makedirs(path)
        self.folder = folder
        self.force = force
        self.runcard = self.load_action_runcard(action_runcard)
        self._allocate_platform(self.runcard["platform"])
        self.qubit = self.runcard["qubit"]

        shutil.copy(action_runcard, f"{path}/runcard.yml")
        self.save_runcard(path)

    def _allocate_platform(self, platform_name):
        from qibo.backends import construct_backend

        self.platform = construct_backend("qibolab", platform=platform_name).platform

    def save_runcard(self, path):
        from qibolab.paths import qibolab_folder

        runcard = qibolab_folder / "runcards" / f"{self.runcard['platform']}.yml"
        shutil.copy(runcard, f"{path}/platform.yml")

    def _build_single_action(self, name):
        """This private method finds the correct function in the qcvv and
        checks if any of the arguments are missing in the runcard"""
        f = getattr(calibrations, name)
        if hasattr(f, "prepare"):
            self.output = f.prepare(name=f.__name__, folder=self.folder)
        sig = inspect.signature(f)
        params = self.runcard["actions"][name]
        for param in list(sig.parameters)[2:]:
            if param not in params:
                raise_error(AttributeError, f"Missing parameter {param} in runcard.")
        return f, params

    def load_action_runcard(self, path):
        """Loading action runcard"""
        with open(path, "r") as file:
            action_settings = yaml.safe_load(file)
        return action_settings

    def execute(self):
        """Method to obtain the calibration routine with the arguments
        checked"""
        self.platform.connect()
        self.platform.setup()
        self.platform.start()
        for action in self.runcard["actions"]:
            routine, args = self._build_single_action(action)
            results = self.get_result(routine, args)
        self.platform.stop()
        self.platform.disconnect()

    def get_result(self, routine, arguments):
        """Method to execute the routine and saving data through
        final action"""
        results = routine(self.platform, self.qubit, **arguments)
        if hasattr(routine, "final_action"):
            return routine.final_action(results, self.output)
        return results
