from qibocal.auto.execute import Executor
from qibocal.cli.report import report

# platform = 'my_platform'  # Specify platform name
# targets = [] # Specify list of targets
# path = "path" Specify output path

with Executor.open(
    path=path,
    platform=platform,
    targets=[target],
    update=True,
    force=True,
) as e:
    rabi_output = e.rabi_amplitude(
        min_amp=0.0,
        max_amp=1,
        step_amp=0.01,
        pulse_length=e.platform.natives.single_qubit[target].RX[0][1].duration,
    )

    ramsey_output = e.ramsey(
        delay_between_pulses_start=10,
        delay_between_pulses_end=5000,
        delay_between_pulses_step=100,
        detuning=1_000_000,
        update=False,
    )

    rabi_output_2 = e.rabi_amplitude(
        min_amp=0,
        max_amp=0.2,
        step_amp=0.01,
        pulse_length=e.platform.natives.single_qubit[target].RX[0][1].duration,
    )

    rabi_output_3 = e.rabi_amplitude(
        min_amp=0,
        max_amp=0.2,
        step_amp=0.01,
        pulse_length=e.platform.natives.single_qubit[target].RX[0][1].duration,
    )


report(e.path, e.history)
