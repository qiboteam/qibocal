"""Minimal Qibocal script example.

In this example, the default Qibocal executor is used. Additional
configurations can still be passed through the `init()` call, but there
is no explicit API to access the execution details.

If more fine grained control is needed, refer to the `rx_calibration.py` example.

.. note::

    though simple, this example is not limited to single protocol execution, but
    multiple protocols can be added as well, essentially in the same fashion of a plain
    runcard - still with the advantage of handling execution and results
    programmatically
"""

from pathlib import Path

from qibocal.cli.report import report
from qibocal.routines import close, init, single_shot_classification

path = Path("test_x")

init(path=path, force=True)
single_shot_classification(nshots=1000)
close()

report(path)
