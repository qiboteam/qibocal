from qibocal.auto.execute import Executor
from qibocal.protocols import rabi_amplitude, ramsey

# platform = 'my_platform'  # Specify platform name
# target = 0  # Specify target qubit
# path = "path" Specify output path

with Executor.open(
    path=path,
    platform=platform,
    targets=[target],
    update=False,
    force=True,
) as e:
    pulse_length = e.platform.natives.single_qubit[target].RX[0][1].duration

    rabi_output = e.fit(
        e.acquire(
            rabi_amplitude(
                min_amp=0.0, max_amp=1, step_amp=0.01, pulse_length=pulse_length
            )
        )
    )
    # update only if chi2 is satisfied
    if rabi_output.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_output.results.chi2[target][0]} greater than 2. Stopping."
        )
    e.update(rabi_output)

    ramsey_output = e.fit(
        e.acquire(
            ramsey(
                delay_between_pulses_start=10,
                delay_between_pulses_end=5000,
                delay_between_pulses_step=100,
                detuning=1_000_000,
            )
        )
    )
    if ramsey_output.results.delta_phys[target][0] < 1e4:
        print(
            f"Ramsey frequency not updated, correction too small {ramsey_output.results.delta_phys[target][0]}"
        )
    else:
        e.update(ramsey_output)

    rabi_output_2 = e.fit(
        e.acquire(
            rabi_amplitude(
                min_amp=0, max_amp=0.2, step_amp=0.01, pulse_length=pulse_length
            )
        )
    )
    # update only if chi2 is satisfied
    if rabi_output_2.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_output_2.results.chi2[target][0]} greater than 2. Stopping."
        )
    e.update(rabi_output_2)

    rabi_output_3 = e.fit(
        e.acquire(
            rabi_amplitude(
                min_amp=0, max_amp=0.2, step_amp=0.01, pulse_length=pulse_length
            )
        )
    )
    # update only if chi2 is satisfied
    if rabi_output_3.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_output_3.results.chi2[target][0]} greater than 2. Stopping."
        )
    e.update(rabi_output_3)
