import datetime
import shutil

import yaml

from qibocal.cli.utils import generate_output_folder, load_yaml
from qibocal.config import raise_error
from qibocal.utils import allocate_qubits


class ActionBuilder:
    """Class for parsing and executing runcards.
    Args:
        runcard (path): path containing the runcard.
        folder (path): path for the output folder.
        force (bool): option to overwrite the output folder if it exists already.
    """

    def __init__(self, runcard, folder, force, update):
        # setting output folder
        self.folder = generate_output_folder(folder, force)

        # parse runcard
        self.runcard = load_yaml(runcard)

        # backend and platform allocation
        backend_name = self.runcard.get("backend", "qibolab")
        platform_name = self.runcard.get("platform", "dummy")
        platform_runcard = self.runcard.get("runcard", None)
        self.backend, self.platform = self._allocate_backend(
            backend_name, platform_name, platform_runcard, update
        )

        self.qubits = allocate_qubits(self.platform, self.runcard.get("qubits", []))

        # Setting format. If absent csv is used.
        self.format = self.runcard.get("format", "csv")
        # Saving runcard
        shutil.copy(runcard, f"{self.folder}/runcard.yml")
        self.save_meta()

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
