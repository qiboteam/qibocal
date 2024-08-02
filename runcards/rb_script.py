from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.cli.report import report

TARGET = "D4"
"""Target qubit where RB will be executed."""
PLATFORM = create_platform("qw11q")
"""Platform where RB will be executed."""
OUTPUT = "test_rb_script"
"""Output folder name."""


# setup
targets = [TARGET]
executor = Executor(
    name="myexec", history=History(), platform=PLATFORM, targets=targets, update=False
)
from myexec import close, init, standard_rb

for amplitude in [0.07, 0.08]:

    executor.platform.qubits[TARGET].native_gates.RX.amplitude = amplitude

    # start experiment
    init(f"{OUTPUT}_amp_{amplitude}", force=True, targets=targets)

    # launch rb
    rb_output = standard_rb(depths=[1, 5, 10, 20], niter=5, nshots=100)

    # retrieve result
    print(f"Fidelity: {rb_output.results.fidelity[TARGET]}")
    print(f"Pulse Fidelity: {rb_output.results.pulse_fidelity[TARGET]}")

    # disconnect
    close()

    # generate report
    report(executor.path, executor.history)
    # reset history
    executor.history = History()
