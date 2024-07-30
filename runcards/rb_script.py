from qibolab import create_platform

from qibocal.auto.execute import Executor
from qibocal.cli.report import report

TARGET = 0
"""Target qubit where RB will be executed."""
PLATFORM = create_platform("dummy")
"""Platform where RB will be executed."""
OUTPUT = "test_rb_script"
"""Output folder name."""


# setup
targets = [TARGET]
executor = Executor(
    name="myexec", history=History(), platform=PLATFORM, targets=targets, update=False
)
from myexec import close, init, standard_rb

# start experiment
init(OUTPUT, force=True, targets=targets)

# launch rb
rb_output = standard_rb(depths=[1, 5, 10, 20, 50, 100], niter=20, nshots=100)

# retrive result
print(f"Fidelity: {rb_output.results.fidelity[TARGET]}")
print(f"Pulse Fidelity: {rb_output.results.pulse_fidelity[TARGET]}")

# disconnect
close()

# generate report
report(executor.path, executor.history)
