import pathlib
from dataclasses import dataclass
from typing import Callable

from qibocal.auto.history import History
from qibocal.auto.serialize import _nested_list_to_tuples
from qibocal.auto.task import TaskId

WEB_DIR = pathlib.Path(__file__).parent
STYLES = WEB_DIR / "static" / "styles.css"
SCRIPT = WEB_DIR / "script.js"
TEMPLATES = WEB_DIR / "templates"


def report_css_styles(path: pathlib.Path):
    """HTML string containing path of css file."""
    return f"<style>\n{pathlib.Path(path).read_text()}\n</style>"


def report_script(path: pathlib.Path):
    """HTML string containing path of js file."""
    return f"<script>\n{pathlib.Path(path).read_text()}\n</script>"


@dataclass
class Report:
    """Report generation class."""

    path: pathlib.Path
    """Path with calibration data."""
    history: History
    """History of protocols."""
    meta: dict
    """Meta data."""
    plotter: Callable
    """Plotting function to generate html."""

    @staticmethod
    def routine_name(routine: TaskId):
        """Prettify routine's name for report headers."""
        return routine.id.title()

    def routine_targets(self, task_id: TaskId):
        """Extract local targets parameter from Task.

        If not available use the global ones.
        """
        local_targets = self.history[task_id].task.targets
        targets = local_targets if len(local_targets) > 0 else self.meta["targets"]
        return _nested_list_to_tuples(targets)
