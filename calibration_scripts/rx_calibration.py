from qibocal.auto.execute import Executor
from qibocal.cli.report import report

target = 0
with Executor.open(
    "myexec",
    path="test_rx_calibration",
    platform="dummy",
    targets=[target],
    update=True,
    force=True,
) as e:
    e.platform.settings.nshots = 2000

    rabi_output = e.rabi_amplitude(
        min_amp=0.0,
        max_amp=2,
        step_amp=0.01,
        pulse_length=e.platform.natives.single_qubit[target].RX[0][1].duration,
    )
    # update only if chi2 is satisfied
    if rabi_output.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_output.results.chi2[target][0]} greater than 2. Stopping."
        )

    ramsey_output = e.ramsey(
        delay_between_pulses_start=10,
        delay_between_pulses_end=5000,
        delay_between_pulses_step=100,
        detuning=1_000_000,
        update=False,
    )
    if ramsey_output.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Ramsey fit has chi2 {ramsey_output.results.chi2[target][0]} greater than 2. Stopping."
        )
    if ramsey_output.results.delta_phys[target][0] < 1e4:
        print(
            f"Ramsey frequency not updated, correction too small { ramsey_output.results.delta_phys[target][0]}"
        )
    else:
        ramsey_output.update_platform(e.platform)

    rabi_output_2 = e.rabi_amplitude(
        min_amp=0,
        max_amp=0.2,
        step_amp=0.01,
        pulse_length=e.platform.natives.single_qubit[target].RX[0][1].duration,
    )
    # update only if chi2 is satisfied
    if rabi_output_2.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_output_2.results.chi2[target][0]} greater than 2. Stopping."
        )

    drag_output = e.drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)
    if drag_output.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Drag fit has chi2 {drag_output.results.chi2[target][0]} greater than 2. Stopping."
        )

    rabi_output_3 = e.rabi_amplitude(
        min_amp=0,
        max_amp=0.2,
        step_amp=0.01,
        pulse_length=e.platform.natives.single_qubit[target].RX[0][1].duration,
    )
    # update only if chi2 is satisfied
    if rabi_output_3.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_output_3.results.chi2[target][0]} greater than 2. Stopping."
        )

report(e.path, e.history)
