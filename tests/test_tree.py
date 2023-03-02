import pathlib

import pytest

from qibocal.tree.execute import Executor
from qibocal.tree.pending import Queue
from qibocal.tree.task import Task

runcards = pathlib.Path(__file__) / "runcards"


def test_execution():
    executor = Executor.load(tasks)

    with pytest.raises(
        RuntimeError, match="Execution completed but tasks still pending "
    ):
        executor.run()
