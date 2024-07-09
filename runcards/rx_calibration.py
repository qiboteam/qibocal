from pathlib import Path

from qibo.backends import construct_backend

from qibocal.auto.execute import Executor
from qibocal.auto.history import History
from qibocal.auto.output import Metadata, Output
from qibocal.cli.report import report

target = "D4"
folder = Path("test_rx_calibration")
force = True

backend = construct_backend(backend="qibolab", platform="qw11q")
platform = backend.platform
if platform is None:
    raise ValueError("Qibocal requires a Qibolab platform to run.")

executor = Executor(
    name="myexec", history=History(), platform=platform, targets=[target]
)

# generate output folder
path = Output.mkdir(folder, force)

# generate meta
meta = Metadata.generate(path.name, backend)
output = Output(History(), meta, platform)
output.dump(path)

from myexec import drag_tuning, rabi_amplitude, ramsey

# connect and initialize platform
platform.connect()

# run
meta.start()


rabi_output = rabi_amplitude(
    min_amp_factor=0.5,
    max_amp_factor=1.5,
    step_amp_factor=0.01,
    pulse_length=platform.qubits[target].native_gates.RX.duration,
    nshots=2048,
)

# update only if chi2 is satisfied
if rabi_output.results.chi2[target][0] > 2:
    raise (
        f"Rabi fit has chi2 {rabi_output.results.chi2[target][0]} greater than 2. Stopping."
    )
else:
    rabi_output.update_platform(platform)

ramsey_output = ramsey(
    delay_between_pulses_start=10,
    delay_between_pulses_end=5000,
    delay_between_pulses_step=100,
    detuning=1_000_000,
    nshots=2048,
)

if ramsey_output.results.chi2[target][0] > 2:
    raise (
        f"Ramsey fit has chi2 {ramsey_output.results.chi2[target][0]} greater than 2. Stopping."
    )
elif ramsey_output.results.delta_phys[target][0] < 1e4:
    print(
        f"Ramsey frequency not updated, correctio to small { ramsey_output.results.delta_phys[target][0]}"
    )
else:
    ramsey_output.update_platform(platform)

rabi_output_2 = rabi_amplitude(
    min_amp_factor=0.5,
    max_amp_factor=1.5,
    step_amp_factor=0.01,
    pulse_length=platform.qubits[target].native_gates.RX.duration,
    nshots=2048,
)


# update only if chi2 is satisfied
if rabi_output_2.results.chi2[target][0] > 2:
    raise (
        f"Rabi fit has chi2 {rabi_output_2.results.chi2[target][0]} greater than 2. Stopping."
    )
else:
    rabi_output_2.update_platform(platform)

drag_output = drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)

if drag_output.results.chi2[target][0] > 2:
    raise (
        f"Drag fit has chi2 {drag_output.results.chi2[target][0]} greater than 2. Stopping."
    )
else:
    drag_output.update_platform(platform)

rabi_output_3 = rabi_amplitude(
    min_amp_factor=0.5,
    max_amp_factor=1.5,
    step_amp_factor=0.01,
    pulse_length=platform.qubits[target].native_gates.RX.duration,
    nshots=2048,
)


# update only if chi2 is satisfied
if rabi_output_3.results.chi2[target][0] > 2:
    raise (
        f"Rabi fit has chi2 {rabi_output_3.results.chi2[target][0]} greater than 2. Stopping."
    )
else:
    rabi_output_3.update_platform(platform)

meta.end()

# stop and disconnect platform
platform.disconnect()

history = executor.history
# dump history, metadata, and updated platform
output.history = history
output.dump(path)

report(path, history)
