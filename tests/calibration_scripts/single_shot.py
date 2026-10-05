"""Minimal Qibocal script example.

In this example, a protocol is passed directly to the Qibocal executor's
acquisition, fit and update methods. Reports are not needed for calibration.
Direct execution does not populate executor history or save data and results.

If more fine grained control is needed, refer to the `rx_calibration.py` example.

.. note::

    though simple, this example is not limited to single protocol execution, but
    multiple protocols can be added as well, essentially in the same fashion of a plain
    runcard - still with the advantage of handling execution and results
    programmatically
"""

from qibocal import Executor
from qibocal.protocols import single_shot_classification

# ADD HERE PLATFORM AND PATH
# platform = "mock"
# targets = [0]
# path = Path("my_path")

with Executor.open(
    path=path,
    platform=platform,
    targets=targets,
    update=False,
    force=True,
) as e:
    bound = single_shot_classification(nshots=1000)
    data = e.acquire(bound)
    results = e.fit(data)
    e.update(results)
    print("\nfidelities:\n", results.results.fidelity, "\n")
