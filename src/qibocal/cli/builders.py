import datetime
import shutil
import tempfile
from pathlib import Path

import yaml

from qibocal.auto.execute import Executor, History
from qibocal.auto.runcard import Runcard
from qibocal.cli.utils import generate_output_folder, load_yaml
from qibocal.config import raise_error
from qibocal.utils import allocate_qubits

META = "meta.yml"
RUNCARD = "runcard.yml"
UPDATED_PLATFORM = "new_platform.yml"
PLATFORM = "platform.yml"


class ActionBuilder:
    """Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
        update (bool): option to
    """

    def __init__(self, runcard, folder, force, update):
        # setting output folder
        self.folder = generate_output_folder(folder, force)
        # parse runcard
        self.runcard = Runcard.load(Path(runcard))
        self.update = update

        # backend and platform allocation
        # backend_name = self.runcard.get("backend", "qibolab")
        # platform_name = self.runcard.get("platform", "dummy")
        # platform_runcard = self.runcard.get("runcard", None)
        # self.backend, self.platform = self._allocate_backend(
        #     backend_name, platform_name, platform_runcard, update
        # )

        # self.qubits = allocate_qubits(self.platform, self.runcard.get("qubits", []))

        # Saving runcard
        # shutil.copy(self.runcard, f"{self.folder}/ RUNCARD")
        # self.save_meta()

        # self.folder = Path(self.folder)
        self._prepare_output(runcard)

    @property
    def platform(self):
        return self.runcard.platform

    @property
    def backend(self):
        return self.runcard.backend

    @property
    def qubits(self):
        if self.platform is not None:
            return allocate_qubits(self.platform, self.runcard.qubits)

        return self.runcard.qubits

    def _prepare_output(self, runcard):
        self.platform.dump(self.folder / PLATFORM)
        shutil.copy(runcard, self.folder / RUNCARD)

        import qibocal

        e = datetime.datetime.now(datetime.timezone.utc)
        meta = {}
        meta["title"] = self.folder.name
        meta["backend"] = str(self.backend)
        meta["platform"] = str(self.backend.platform)
        meta["date"] = e.strftime("%Y-%m-%d")
        meta["start-time"] = e.strftime("%H:%M:%S")
        meta["end-time"] = e.strftime("%H:%M:%S")
        meta["versions"] = self.backend.versions  # pylint: disable=E1101
        meta["versions"]["qibocal"] = qibocal.__version__

        with open(self.folder / META, "w") as file:
            yaml.dump(meta, file)

    def run(self):
        self.executor = Executor.load(
            self.runcard,
            self.folder,
            self.platform,
            self.qubits,
            self.update,
        )
        if self.platform is not None:
            self.platform.connect()
            self.platform.setup()
            self.platform.start()

        self.executor.run()

        if self.platform is not None:
            self.platform.stop()
            self.platform.disconnect()

    def _allocate_backend(self, backend_name, platform_name, platform_runcard, update):
        """Allocate the platform using Qibolab."""
        from qibo.backends import GlobalBackend, set_backend
        from qibolab import dummy

        if backend_name == "qibolab":
            if platform_name == dummy.NAME:
                platform = dummy.create_dummy()
                platform.dump(f"{self.folder}/platform.yml")
                if update:
                    updated_runcard = f"{self.folder}/new_platform.yml"
                    platform.dump(updated_runcard)
                if platform_runcard is not None:
                    raise_error(
                        ValueError, "Dummy platform doesn't support custom runcards."
                    )
            else:
                if platform_runcard is None:
                    from qibolab import get_platforms_path

                    original_runcard = get_platforms_path() / f"{platform_name}.yml"
                else:
                    original_runcard = platform_runcard
                # copy of the original runcard that will stay unmodified
                shutil.copy(original_runcard, f"{self.folder}/platform.yml")
                if update:
                    # copy of the original runcard that will be modified during calibration
                    updated_runcard = f"{self.folder}/new_platform.yml"
                    shutil.copy(original_runcard, updated_runcard)
                else:
                    updated_runcard = original_runcard

        # allocate backend with updated_runcard
        set_backend(
            backend=backend_name,
            platform=platform_name,
            runcard=updated_runcard if update else None,
        )
        backend = GlobalBackend()
        return backend, backend.platform

    def dump_report(self):
        """Dump report."""
        from qibocal.web.report import create_autocalibration_report

        # update end time
        meta = yaml.safe_load((self.folder / META).read_text())
        e = datetime.datetime.now(datetime.timezone.utc)
        meta["end-time"] = e.strftime("%H:%M:%S")
        with open(self.folder / META, "w") as file:
            yaml.dump(meta, file)

        create_autocalibration_report(self.folder, self.executor.history)

    def dump_platform_runcard(self):
        """Dump platform runcard."""
        if self.platform is not None:
            self.platform.dump(self.folder / UPDATED_PLATFORM)

    def save_meta(self):
        import qibocal

        e = datetime.datetime.now(datetime.timezone.utc)
        meta = {}
        meta["title"] = self.folder
        meta["backend"] = str(self.backend)
        meta["platform"] = str(self.backend.platform)
        meta["date"] = e.strftime("%Y-%m-%d")
        meta["start-time"] = e.strftime("%H:%M:%S")
        meta["end-time"] = e.strftime("%H:%M:%S")
        meta["versions"] = self.backend.versions  # pylint: disable=E1101
        meta["versions"]["qibocal"] = qibocal.__version__

        with open(f"{self.folder}/meta.yml", "w") as file:
            yaml.dump(meta, file)


class ReportBuilder:
    def __init__(self, path: Path, history: History):
        # FIXME: currently the title of the report is the output folder
        self.path = self.title = path
        self.metadata = yaml.safe_load((path / META).read_text())
        self.runcard = Runcard.load(path / RUNCARD)
        self.format = self.runcard.format
        self.qubits = self.runcard.qubits

        self.history = history

    def routine_name(self, routine, iteration):
        """Prettify routine's name for report headers."""
        name = routine.replace("_", " ").title()
        return f"{name} - {iteration}"

    def routine_qubits(self, routine_name, iteration):
        """Get local qubits parameter from Task if available otherwise use global one."""
        local_qubits = self.history[(routine_name, iteration)].task.qubits
        return local_qubits if len(local_qubits) > 0 else self.qubits

    def single_qubit_plot(self, routine_name, iteration, qubit):
        node = self.history[(routine_name, iteration)]
        data = node.task.data
        figures, fitting_report = node.task.operation.report(data, node.res, qubit)
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            html_list = []
            for figure in figures:
                figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
                temp.seek(0)
                fightml = temp.read().decode("utf-8")
                html_list.append(fightml)

        all_html = "".join(html_list)
        return all_html, fitting_report

    def plot(self, routine_name, iteration):
        node = self.history[(routine_name, iteration)]
        data = node.task.data
        figures, fitting_report = node.task.operation.report(data)
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            html_list = []
            for figure in figures:
                figure.write_html(temp.name, include_plotlyjs=False, full_html=False)
                temp.seek(0)
                fightml = temp.read().decode("utf-8")
                html_list.append(fightml)

        all_html = "".join(html_list)
        return all_html, fitting_report
