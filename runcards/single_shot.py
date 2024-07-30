from pathlib import Path

from qibocal.cli.report import report
from qibocal.routines import close, init, single_shot_classification

path = Path("test_x")

init(path=path, force=True)
single_shot_classification(nshots=1000)
close()

report(path)
