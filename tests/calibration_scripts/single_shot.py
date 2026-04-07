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

from qibocal import Executor
from qibocal.cli.report import report

# ADD HERE PLATFORM AND PATH
# platform = "mock"
# targets = [0]
# path = Path("my_path")

with Executor.open(
    "myexec",
    path=path,
    platform=platform,
    targets=targets,
    update=True,
    force=True,
) as e:
    ssc = e.single_shot_classification(nshots=1000)
    print("\nfidelities:\n", ssc.results.readout_fidelity, "\n")

report(path)
