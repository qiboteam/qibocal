# -*- coding: utf-8 -*-
import datetime
import inspect
import os
import shutil

import yaml

from qcvv import calibrations
from qcvv.config import log, raise_error


class ActionBuilder:
    """Class for parsing and executing runcards.

    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
    """

    def __init__(self, runcard, folder=None, force=False):

        path, self.folder = self._generate_output_folder(folder, force)
        self.runcard = self.load_runcard(runcard)
        self._allocate_platform(self.runcard["platform"])
        self.qubits = self.runcard["qubits"]
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
        import qibo
        import qibolab
        from qibolab.paths import qibolab_folder

        import qcvv

        platform_runcard = (
            qibolab_folder / "runcards" / f"{self.runcard['platform']}.yml"
        )
        shutil.copy(platform_runcard, f"{path}/platform.yml")

        e = datetime.datetime.now(datetime.timezone.utc)
        meta = {}
        meta["date"] = e.strftime("%Y-%m-%d")
        meta["start-time"] = e.strftime("%H:%M:%S")
        meta["end-time"] = e.strftime("%H:%M:%S")
        meta["versions"] = {
            "qibo": qibo.__version__,
            "qibolab": qibolab.__version__,
            "qcvv": qcvv.__version__,
        }
        with open(f"{path}/meta.yml", "w") as file:
            yaml.dump(meta, file)

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
            self._execute_single_action(routine, args)
        self.platform.stop()
        self.platform.disconnect()
        self.dump_report()

    def _execute_single_action(self, routine, arguments):
        """Method to execute a single action and retrieving the results."""
        for qubit in self.qubits:
            results = routine(self.platform, qubit, **arguments)
            if hasattr(routine, "final_action"):
                routine.final_action(results, self.output, self.format)

    def dump_report(self):
        from qcvv.web.report import create_report

        # update end time
        with open(f"{self.folder}/meta.yml", "r") as file:
            meta = yaml.safe_load(file)
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(f"{self.folder}/meta.yml", "w") as file:
            yaml.dump(meta, file)

        create_report(self.folder)


class Metadata:
    def __init__(self, metadata):
        self.date = metadata.get("date")
        self.start_time = metadata.get("start-time")
        self.end_time = metadata.get("end-time")
        self.versions = metadata.get("versions")


class ReportBuilder:
    def __init__(self, path):
        # TODO: Create a ``load_yaml`` method and use it here and in
        # ``ActionBuilder``
        with open(os.path.join(path, "runcard.yml"), "r") as file:
            self.runcard = yaml.safe_load(file)

        with open(os.path.join(path, "meta.yml"), "r") as file:
            metadata = yaml.safe_load(file)

        self.metadata = Metadata(metadata)

        self.format = self.runcard.get("format")
        self.qubits = self.runcard.get("qubits")
        self.routines = self._create_routines()

    def _create_routines(self):
        routines = []
        for name in self.runcard.get("actions").keys():
            routine = getattr(calibrations, name)
            routine.name = routine.__name__
            routine.pretty_name = routine.name.replace("_", " ").title()
            if hasattr(routine, "plots"):
                routine.plots = routine.plots[::-1]
            else:
                routine.plots = []
            routines.append(routine)
        return routines
