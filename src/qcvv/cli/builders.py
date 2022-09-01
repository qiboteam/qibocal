# -*- coding: utf-8 -*-
import datetime
import inspect
import os
import shutil

import yaml

from qcvv import calibrations
from qcvv.config import log, raise_error
from qcvv.data import Data


def load_yaml(path):
    """Load yaml file from disk."""
    with open(path, "r") as file:
        data = yaml.safe_load(file)
    return data


class ActionBuilder:
    """Class for parsing and executing runcards.

    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
    """

    def __init__(self, runcard, folder=None, force=False):
        path, self.folder = self._generate_output_folder(folder, force)
        self.runcard = load_yaml(runcard)
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
        path = os.path.join(self.folder, f"data/{name}/")
        os.makedirs(path)
        sig = inspect.signature(f)
        params = self.runcard["actions"][name]
        for param in list(sig.parameters)[2:-1]:
            if param not in params:
                raise_error(AttributeError, f"Missing parameter {param} in runcard.")
        return f, params, path

    def execute(self):
        """Method to execute sequentially all the actions in the runcard."""
        self.platform.connect()
        self.platform.setup()
        self.platform.start()
        for action in self.runcard["actions"]:
            routine, args, path = self._build_single_action(action)
            self._execute_single_action(routine, args, path)
        self.platform.stop()
        self.platform.disconnect()

    def _execute_single_action(self, routine, arguments, path):
        """Method to execute a single action and retrieving the results."""
        for qubit in self.qubits:
            results = routine(self.platform, qubit, **arguments)
            if self.format is None:
                raise_error(
                    ValueError, f"Cannot store data using {self.format} format."
                )
            for data in results:
                getattr(data, f"to_{self.format}")(path)

            self.update_platform_runcard(qubit, routine.__name__)

    def update_platform_runcard(self, qubit, routine):

        data_fit = Data.load_data(self.folder, routine, self.format, f"fit_q{qubit}")
        params = [i for i in list(data_fit.df.keys()) if "fit" not in i]
        settings = load_yaml(f"{self.folder}/platform.yml")

        for param in params:
            settings["characterization"]["single_qubit"][qubit][param] = int(
                data_fit.df[param][0]
            )

        with open(f"{self.folder}/data/{routine}/platform.yml", "a+") as file:
            yaml.dump(
                settings, file, sort_keys=False, indent=4, default_flow_style=None
            )

    def dump_report(self):
        from qcvv.web.report import create_report

        # update end time
        meta = load_yaml(f"{self.folder}/meta.yml")
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(f"{self.folder}/meta.yml", "w") as file:
            yaml.dump(meta, file)

        create_report(self.folder)


class ReportBuilder:
    """Parses routines and plots to report and live plotting page.

    Args:
        path (str): Path to the data folder to generate report for.
    """

    def __init__(self, path):
        self.path = path
        self.metadata = load_yaml(os.path.join(path, "meta.yml"))

        # find proper path title
        base, self.title = os.path.join(os.getcwd(), path), ""
        while self.title in ("", "."):
            base, self.title = os.path.split(base)

        self.runcard = load_yaml(os.path.join(path, "runcard.yml"))
        self.format = self.runcard.get("format")
        self.qubits = self.runcard.get("qubits")

        # create calibration routine objects
        # (could be incorporated to :meth:`qcvv.cli.builders.ActionBuilder._build_single_action`)
        self.routines = []
        for action in self.runcard.get("actions"):
            routine = getattr(calibrations, action)
            if not hasattr(routine, "plots"):
                routine.plots = []
            self.routines.append(routine)

    def get_routine_name(self, routine):
        """Prettify routine's name for report headers."""
        return routine.__name__.replace("_", " ").title()

    def get_figure(self, routine, method, qubit):
        """Get html figure for report.

        Args:
            routine (Callable): Calibration method.
            method (Callable): Plot method.
            qubit (int): Qubit id.
        """
        import tempfile

        figure = method(self.path, routine.__name__, qubit, self.format)
        with tempfile.NamedTemporaryFile() as temp:
            figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
            fightml = temp.read().decode("utf-8")
        return fightml

    def get_live_figure(self, routine, method, qubit):
        """Get url to dash page for live plotting.

        This url is used by :meth:`qcvv.web.app.get_graph`.

        Args:
            routine (Callable): Calibration method.
            method (Callable): Plot method.
            qubit (int): Qubit id.
        """
        return os.path.join(
            method.__name__,
            self.path,
            routine.__name__,
            str(qubit),
            self.format,
        )
