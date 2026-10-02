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
    rabi_parameters = {
        "min_amp": 0.0,
        "max_amp": 1,
        "step_amp": 0.01,
        "pulse_length": e.platform.natives.single_qubit[target].RX[0][1].duration,
    }
    rabi_data = e.acquire(rabi_amplitude(**rabi_parameters))
    rabi_completed = e.fit(rabi_data)
    rabi_results = rabi_completed.results
    # update only if chi2 is satisfied
    if rabi_results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_results.chi2[target][0]} greater than 2. Stopping."
        )
    e.update(rabi_completed)

    ramsey_parameters = {
        "delay_between_pulses_start": 10,
        "delay_between_pulses_end": 5000,
        "delay_between_pulses_step": 100,
        "detuning": 1_000_000,
    }
    ramsey_data = e.acquire(ramsey(**ramsey_parameters))
    ramsey_completed = e.fit(ramsey_data)
    ramsey_results = ramsey_completed.results
    if ramsey_results.delta_phys[target][0] < 1e4:
        print(
            f"Ramsey frequency not updated, correction too small {ramsey_results.delta_phys[target][0]}"
        )
    else:
        e.update(ramsey_completed)

    rabi_parameters_2 = {
        "min_amp": 0,
        "max_amp": 0.2,
        "step_amp": 0.01,
        "pulse_length": e.platform.natives.single_qubit[target].RX[0][1].duration,
    }
    rabi_data_2 = e.acquire(rabi_amplitude(**rabi_parameters_2))
    rabi_completed_2 = e.fit(rabi_data_2)
    rabi_results_2 = rabi_completed_2.results
    # update only if chi2 is satisfied
    if rabi_results_2.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_results_2.chi2[target][0]} greater than 2. Stopping."
        )
    e.update(rabi_completed_2)

    rabi_parameters_3 = {
        "min_amp": 0,
        "max_amp": 0.2,
        "step_amp": 0.01,
        "pulse_length": e.platform.natives.single_qubit[target].RX[0][1].duration,
    }
    rabi_data_3 = e.acquire(rabi_amplitude(**rabi_parameters_3))
    rabi_completed_3 = e.fit(rabi_data_3)
    rabi_results_3 = rabi_completed_3.results
    # update only if chi2 is satisfied
    if rabi_results_3.chi2[target][0] > 2:
        raise RuntimeError(
            f"Rabi fit has chi2 {rabi_results_3.chi2[target][0]} greater than 2. Stopping."
        )
    e.update(rabi_completed_3)
