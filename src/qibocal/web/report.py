import pathlib
from dataclasses import dataclass
from typing import Callable

from qibocal.auto.history import History
from qibocal.auto.task import TaskId

WEB_DIR = pathlib.Path(__file__).parent
STYLES = WEB_DIR / "static" / "styles.css"
TEMPLATES = WEB_DIR / "templates"


@dataclass
class Report:
    """Report generation class."""

    path: pathlib.Path
    """Path with calibration data."""
    targets: list
    """Global targets."""
    history: History
    """History of protocols."""
    meta: dict
    """Meta data."""
    plotter: Callable
    """Plotting function to generate html."""

    @staticmethod
    def routine_name(routine):
        """Prettify routine's name for report headers."""
        return routine.title()

    def routine_targets(self, task_id: TaskId):
        """Get local targets parameter from Task if available otherwise use global one."""
        local_targets = self.history[task_id].task.targets
        return local_targets if len(local_targets) > 0 else self.targets
